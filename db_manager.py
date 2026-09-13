#!/usr/bin/env python3
"""
db_manager.py
- PostgreSQL database manager for the Local LLM Service Platform.
- Manages:
  1. Hugging Face model catalog synchronization (hf_model_catalog)
  2. Model download history & status tracking (model_downloads)
  3. Server runtime lifecycle and error logging (server_runtime_logs)
  4. LiteLLM user, API key, and spend log statistics (LiteLLM tables)
- Features zero host-dependency execution via docker exec with psql,
  with automatic docker group permission handling.
"""

import sys
import os
import json
import shlex
import subprocess
from datetime import datetime

CONTAINER_NAME = os.environ.get("POSTGRES_CONTAINER", "litellm-postgres")
DB_USER = os.environ.get("POSTGRES_USER", "llm_admin")
DB_NAME = os.environ.get("POSTGRES_DB", "litellm")


class DBManager:
    def __init__(self, container_name=CONTAINER_NAME, user=DB_USER, dbname=DB_NAME):
        self.container_name = container_name
        self.user = user
        self.dbname = dbname
        self._use_sg_docker = False
        self._check_docker_permission()

    def _check_docker_permission(self):
        """Checks if docker requires sg docker wrapper for current session."""
        try:
            res = subprocess.run(["docker", "info"], capture_output=True, text=True, timeout=3)
            if res.returncode != 0 and "permission denied" in (res.stderr + res.stdout).lower():
                self._use_sg_docker = True
        except Exception:
            pass

    def _run_cmd(self, cmd_list, input_data=None, timeout=10):
        """Runs a command, automatically wrapping with sg docker if needed."""
        if self._use_sg_docker:
            quoted = " ".join(shlex.quote(str(x)) for x in cmd_list)
            cmd = ["sg", "docker", "-c", quoted]
        else:
            cmd = cmd_list
        return subprocess.run(cmd, input=input_data, capture_output=True, text=True, timeout=timeout)

    def is_available(self):
        """Checks if the PostgreSQL container is running and accepting connections."""
        try:
            res = self._run_cmd([
                "docker", "exec", self.container_name,
                "pg_isready", "-U", self.user, "-d", self.dbname
            ], timeout=5)
            return res.returncode == 0
        except Exception:
            return False

    def ensure_schema(self):
        """Ensures custom tables (hf_model_catalog, model_downloads, server_runtime_logs) exist."""
        if not self.is_available():
            return False
        try:
            res = self.query_json("SELECT (to_regclass('public.hf_model_catalog') IS NOT NULL) AS has_table;")
            if res and res[0].get("has_table"):
                return True
            init_sql = Path(__file__).resolve().parent / "init_db.sql"
            if init_sql.exists():
                with open(init_sql, "r", encoding="utf-8") as f:
                    content = f.read()
                ok, _, err = self.execute(content)
                return ok
        except Exception:
            pass
        return False

    def execute(self, sql, timeout=10):
        """Executes a SQL command inside the PostgreSQL container."""
        try:
            cmd = [
                "docker", "exec", "-i", self.container_name,
                "psql", "-U", self.user, "-d", self.dbname,
                "-v", "ON_ERROR_STOP=1", "-q"
            ]
            res = self._run_cmd(cmd, input_data=sql, timeout=timeout)
            return res.returncode == 0, res.stdout, res.stderr
        except Exception as e:
            return False, "", str(e)

    def query_json(self, sql, timeout=10):
        """Executes a SELECT query and returns the results as Python dict/list via JSON."""
        clean_sql = sql.strip().rstrip(";")
        json_sql = f"SELECT coalesce(json_agg(t), '[]'::json) FROM ({clean_sql}) t;"
        try:
            cmd = [
                "docker", "exec", "-i", self.container_name,
                "psql", "-U", self.user, "-d", self.dbname,
                "-t", "-A", "-c", json_sql
            ]
            res = self._run_cmd(cmd, timeout=timeout)
            if res.returncode == 0 and res.stdout.strip():
                return json.loads(res.stdout.strip())
            return []
        except Exception as e:
            sys.stderr.write(f"DB Query Error: {e}\n")
            return []

    def execute_scalar(self, sql, timeout=5):
        """Executes SQL and returns the first scalar result (e.g. for RETURNING id)."""
        try:
            cmd = [
                "docker", "exec", "-i", self.container_name,
                "psql", "-U", self.user, "-d", self.dbname,
                "-t", "-A", "-q"
            ]
            res = self._run_cmd(cmd, input_data=sql, timeout=timeout)
            if res.returncode == 0 and res.stdout.strip():
                val = res.stdout.strip().split("\n")[0].strip()
                return val
        except Exception:
            pass
        return None

    # --- 1. Hugging Face Catalog Synchronization ----------------------------

    def sync_catalog(self, catalog_list):
        """Upserts a list of model dicts into hf_model_catalog."""
        if not self.is_available():
            return False

        values = []
        for m in catalog_list:
            mid = m.get("id", "").replace("'", "''")
            repo = m.get("repo", "").replace("'", "''")
            name = m.get("name", "").replace("'", "''")
            fn = m.get("filename", "").replace("'", "''")
            quant = m.get("quant", "").replace("'", "''")
            size_gb = float(m.get("size_gb", 0))
            size_bytes = int(size_gb * (1024 ** 3))
            vram = float(m.get("recommended_vram_gb", m.get("min_vram_gb", 8.0)))
            cat = m.get("category", "").replace("'", "''")
            desc = m.get("desc", "").replace("'", "''")
            args = m.get("args", "").replace("'", "''")

            values.append(
                f"('{mid}', '{repo}', '{name}', '{fn}', '{quant}', {size_bytes}, {size_gb}, {vram}, '{cat}', '{desc}', '{args}')"
            )

        if not values:
            return False

        val_str = ",\n".join(values)
        sql = f"""
        INSERT INTO hf_model_catalog (id, repo_id, model_name, filename, quantization, size_bytes, size_gb, min_vram_gb, category, description, extra_args)
        VALUES {val_str}
        ON CONFLICT (id) DO UPDATE SET
            model_name = EXCLUDED.model_name,
            filename = EXCLUDED.filename,
            size_gb = EXCLUDED.size_gb,
            description = EXCLUDED.description,
            extra_args = EXCLUDED.extra_args,
            updated_at = CURRENT_TIMESTAMP;
        """
        ok, _, err = self.execute(sql)
        if not ok:
            sys.stderr.write(f"sync_catalog failed: {err}\n")
        return ok

    # --- 2. Model Downloads Tracking -----------------------------------------

    def log_download_start(self, model_id, filename, local_path, download_url, total_bytes):
        """Records the start of a download and returns the record ID."""
        if not self.is_available():
            return None

        mid = f"'{model_id}'" if model_id else "NULL"
        fn = filename.replace("'", "''")
        lp = local_path.replace("'", "''")
        url = download_url.replace("'", "''")

        sql = f"""
        INSERT INTO model_downloads (model_id, filename, local_path, download_url, status, total_bytes, started_at)
        VALUES ({mid}, '{fn}', '{lp}', '{url}', 'DOWNLOADING', {int(total_bytes)}, CURRENT_TIMESTAMP)
        RETURNING id;
        """
        val = self.execute_scalar(sql)
        return int(val) if val and val.isdigit() else None

    def log_download_complete(self, download_id, downloaded_bytes, speed_mb=0, duration_sec=0):
        """Updates download record as COMPLETED."""
        if not self.is_available() or not download_id:
            return False
        sql = f"""
        UPDATE model_downloads
        SET status = 'COMPLETED',
            downloaded_bytes = {int(downloaded_bytes)},
            download_speed_mb = {float(speed_mb):.2f},
            duration_seconds = {int(duration_sec)},
            completed_at = CURRENT_TIMESTAMP
        WHERE id = {int(download_id)};
        """
        ok, _, _ = self.execute(sql)
        return ok

    def log_download_failed(self, download_id, error_message):
        """Updates download record as FAILED with error message."""
        if not self.is_available() or not download_id:
            return False
        err = str(error_message).replace("'", "''")
        sql = f"""
        UPDATE model_downloads
        SET status = 'FAILED',
            error_message = '{err}',
            completed_at = CURRENT_TIMESTAMP
        WHERE id = {int(download_id)};
        """
        ok, _, _ = self.execute(sql)
        return ok

    # --- 3. Server Runtime Logs ----------------------------------------------

    def log_server_start(self, container_name, model_id, filename, port=8080, hw_profile="100% GPU", gpu_name="NVIDIA"):
        """Records container launch event."""
        if not self.is_available():
            return None
        cname = container_name.replace("'", "''")
        mid = f"'{model_id}'" if model_id else "NULL"
        fn = f"'{filename.replace("'", "''")}'" if filename else "NULL"
        hw = hw_profile.replace("'", "''")
        gpu = gpu_name.replace("'", "''")

        sql = f"""
        INSERT INTO server_runtime_logs (container_name, model_id, filename, port, hardware_profile, gpu_name, status, started_at)
        VALUES ('{cname}', {mid}, {fn}, {int(port)}, '{hw}', '{gpu}', 'HEALTHY', CURRENT_TIMESTAMP)
        RETURNING id;
        """
        val = self.execute_scalar(sql)
        return int(val) if val and val.isdigit() else None

    def log_server_stop(self, container_name, exit_code=0):
        """Marks running server log as STOPPED."""
        if not self.is_available():
            return False
        cname = container_name.replace("'", "''")
        sql = f"""
        UPDATE server_runtime_logs
        SET status = 'STOPPED',
            exit_code = {int(exit_code)},
            stopped_at = CURRENT_TIMESTAMP
        WHERE container_name = '{cname}' AND status IN ('HEALTHY', 'STARTING');
        """
        ok, _, _ = self.execute(sql)
        return ok

    # --- 4. Statistics & Dashboard Queries -----------------------------------

    def get_download_history(self, limit=10):
        """Returns the latest model downloads."""
        sql = f"""
        SELECT d.id, coalesce(m.model_name, d.filename) as name, d.status,
               round(d.total_bytes / 1073741824.0, 2) as size_gb,
               d.download_speed_mb as speed_mb, d.duration_seconds,
               to_char(d.started_at, 'YYYY-MM-DD HH24:MI:SS') as started_at
        FROM model_downloads d
        LEFT JOIN hf_model_catalog m ON d.model_id = m.id
        ORDER BY d.started_at DESC
        LIMIT {int(limit)}
        """
        return self.query_json(sql)

    def get_runtime_history(self, limit=10):
        """Returns the latest container runtime instances."""
        sql = f"""
        SELECT id, container_name, filename, hardware_profile, status,
               to_char(started_at, 'YYYY-MM-DD HH24:MI:SS') as started_at,
               to_char(stopped_at, 'YYYY-MM-DD HH24:MI:SS') as stopped_at,
               round(EXTRACT(EPOCH FROM (coalesce(stopped_at, CURRENT_TIMESTAMP) - started_at)) / 60.0, 1) as uptime_min
        FROM server_runtime_logs
        ORDER BY started_at DESC
        LIMIT {int(limit)}
        """
        return self.query_json(sql)

    def get_users_summary(self):
        """Returns LiteLLM users and their verification tokens/keys."""
        exists = self.query_json("SELECT (to_regclass('\"LiteLLM_UserTable\"') IS NOT NULL) as has_table")
        if not exists or not exists[0].get("has_table"):
            return []

        sql = """
        SELECT u.user_id, u.user_email, u.user_role, u.max_budget, u.spend,
               coalesce(count(t.token), 0) as active_keys,
               to_char(u.created_at, 'YYYY-MM-DD HH24:MI:SS') as created_at
        FROM "LiteLLM_UserTable" u
        LEFT JOIN "LiteLLM_VerificationToken" t ON u.user_id = t.user_id
        GROUP BY u.user_id, u.user_email, u.user_role, u.max_budget, u.spend, u.created_at
        ORDER BY u.created_at DESC
        """
        return self.query_json(sql)

    def get_token_usage_stats(self):
        """Returns overall token usage and call count from LiteLLM_SpendLogs."""
        exists = self.query_json("SELECT (to_regclass('\"LiteLLM_SpendLogs\"') IS NOT NULL) as has_table")
        if not exists or not exists[0].get("has_table"):
            return {"total_requests": 0, "total_tokens": 0, "prompt_tokens": 0, "completion_tokens": 0, "models": []}

        sql_summary = """
        SELECT count(*) as total_requests,
               coalesce(sum("total_tokens"), 0) as total_tokens,
               coalesce(sum("prompt_tokens"), 0) as prompt_tokens,
               coalesce(sum("completion_tokens"), 0) as completion_tokens
        FROM "LiteLLM_SpendLogs";
        """
        summary = self.query_json(sql_summary)

        sql_by_model = """
        SELECT model, count(*) as requests,
               coalesce(sum("total_tokens"), 0) as tokens,
               round(avg(EXTRACT(EPOCH FROM ("endTime" - "startTime")) * 1000.0), 1) as avg_latency_ms
        FROM "LiteLLM_SpendLogs"
        GROUP BY model
        ORDER BY tokens DESC;
        """
        by_model = self.query_json(sql_by_model)

        res = summary[0] if summary else {"total_requests": 0, "total_tokens": 0, "prompt_tokens": 0, "completion_tokens": 0}
        res["models"] = by_model
        return res


def main():
    db = DBManager()
    if len(sys.argv) < 2:
        print(f"PostgreSQL Status: {'🟢 Available' if db.is_available() else '🔴 Offline'}")
        sys.exit(0)

    cmd = sys.argv[1]
    if cmd == "status":
        print("ONLINE" if db.is_available() else "OFFLINE")
    elif cmd == "downloads":
        print(json.dumps(db.get_download_history(), indent=2, ensure_ascii=False))
    elif cmd == "runtimes":
        print(json.dumps(db.get_runtime_history(), indent=2, ensure_ascii=False))
    elif cmd == "users":
        print(json.dumps(db.get_users_summary(), indent=2, ensure_ascii=False))
    elif cmd == "stats":
        print(json.dumps(db.get_token_usage_stats(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
