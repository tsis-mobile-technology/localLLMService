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
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
BACKUPS_DIR = SCRIPT_DIR / "backups"

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

    # --- 5. Backup & Restore Operations --------------------------------------

    def backup_database(self, backup_dir=None, prefix="litellm_backup"):
        """
        Dumps the PostgreSQL database to a timestamped .sql file and updates the latest.sql copy.
        Returns: (success: bool, latest_file: Path, timestamp_file: Path, size_bytes: int, message: str)
        """
        if not self.is_available():
            return False, None, None, 0, "PostgreSQL 컨테이너가 실행 중이지 않거나 준비되지 않았습니다."

        target_dir = Path(backup_dir) if backup_dir else BACKUPS_DIR
        target_dir.mkdir(parents=True, exist_ok=True)

        now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        ts_file = target_dir / f"{prefix}_{now_str}.sql"
        latest_file = target_dir / f"{prefix}_latest.sql"

        cmd = [
            "docker", "exec", self.container_name,
            "pg_dump", "-U", self.user, "-d", self.dbname,
            "--clean", "--if-exists", "--no-owner", "--no-privileges"
        ]

        try:
            res = self._run_cmd(cmd, timeout=90)
            if res.returncode != 0:
                err_msg = res.stderr.strip() or "pg_dump 실행 중 에러가 발생했습니다."
                return False, None, None, 0, f"백업 실패: {err_msg}"

            sql_output = res.stdout
            if not sql_output.strip():
                return False, None, None, 0, "백업 결과 데이터가 비어 있습니다."

            with open(ts_file, "w", encoding="utf-8") as f:
                f.write(sql_output)

            shutil.copy2(ts_file, latest_file)
            size = ts_file.stat().st_size

            return True, latest_file, ts_file, size, "백업이 성공적으로 완료되었습니다."
        except Exception as e:
            return False, None, None, 0, f"백업 중 예외 발생: {e}"

    def restore_database(self, backup_file=None):
        """
        Restores the database from a given SQL dump file. Defaults to backups/litellm_backup_latest.sql.
        Returns: (success: bool, restored_tables: int, message: str)
        """
        if not self.is_available():
            return False, 0, "PostgreSQL 컨테이너가 실행 중이지 않거나 준비되지 않았습니다."

        target_file = Path(backup_file) if backup_file else (BACKUPS_DIR / "litellm_backup_latest.sql")
        if not target_file.exists():
            return False, 0, f"백업 파일을 찾을 수 없습니다: {target_file}"

        if target_file.stat().st_size == 0:
            return False, 0, f"백업 파일 크기가 0바이트입니다: {target_file}"

        try:
            with open(target_file, "r", encoding="utf-8") as f:
                sql_content = f.read()

            cmd = [
                "docker", "exec", "-i", self.container_name,
                "psql", "-U", self.user, "-d", self.dbname,
                "-v", "ON_ERROR_STOP=0", "-q"
            ]
            res = self._run_cmd(cmd, input_data=sql_content, timeout=120)

            tables_res = self.query_json("SELECT count(*) as count FROM information_schema.tables WHERE table_schema='public';")
            count = int(tables_res[0].get("count", 0)) if tables_res else 0

            if count > 0:
                return True, count, f"데이터베이스가 성공적으로 복원되었습니다 (총 {count}개 테이블)."
            else:
                return False, 0, f"복원은 완료되었으나 복원된 테이블이 없습니다: {res.stderr}"
        except Exception as e:
            return False, 0, f"복원 중 예외 발생: {e}"

    def _has_table(self, table_name):
        """Helper to check if a specific table exists."""
        try:
            res = self.query_json(
                f"SELECT (to_regclass('public.\"{table_name}\"') IS NOT NULL OR to_regclass('public.{table_name}') IS NOT NULL) as exists;"
            )
            return bool(res and res[0].get("exists"))
        except Exception:
            return False

    def is_database_empty(self):
        """
        Checks if the database has no user/application data.
        Returns True if empty or structure-only without data.
        """
        if not self.is_available():
            return False
        try:
            tables = self.query_json("SELECT count(*) as count FROM information_schema.tables WHERE table_schema='public';")
            cnt = int(tables[0].get("count", 0)) if tables else 0
            if cnt == 0:
                return True

            c_count = 0
            if self._has_table("hf_model_catalog"):
                res = self.query_json("SELECT count(*) as count FROM hf_model_catalog;")
                c_count = int(res[0].get("count", 0)) if res else 0

            u_count = 0
            if self._has_table("LiteLLM_UserTable"):
                res = self.query_json('SELECT count(*) as count FROM "LiteLLM_UserTable";')
                u_count = int(res[0].get("count", 0)) if res else 0

            return (c_count == 0 and u_count == 0)
        except Exception:
            return True

    def auto_restore_if_empty(self, backup_file=None):
        """
        Restores database only if current DB is empty and a valid backup file exists.
        Returns: (success: bool, status_code: str)
        """
        if not self.is_available():
            return False, "DB_NOT_AVAILABLE"

        target_file = Path(backup_file) if backup_file else (BACKUPS_DIR / "litellm_backup_latest.sql")
        if not target_file.exists() or target_file.stat().st_size == 0:
            return False, "NO_BACKUP_FILE"

        if not self.is_database_empty():
            return True, "SKIPPED_ALREADY_HAS_DATA"

        ok, count, msg = self.restore_database(target_file)
        if ok:
            return True, f"AUTO_RESTORED_{count}_TABLES"
        return False, f"RESTORE_FAILED: {msg}"


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
    elif cmd == "backup":
        out_dir = sys.argv[2] if len(sys.argv) > 2 else None
        ok, latest, ts, size, msg = db.backup_database(out_dir)
        if ok:
            size_mb = size / (1024 * 1024)
            print(f"✅ 백업 완료: {latest} ({size_mb:.2f} MB, 타임스탬프: {ts.name})")
            sys.exit(0)
        else:
            print(f"❌ 백업 실패: {msg}", file=sys.stderr)
            sys.exit(1)
    elif cmd == "restore":
        bfile = sys.argv[2] if len(sys.argv) > 2 else None
        ok, count, msg = db.restore_database(bfile)
        if ok:
            print(f"✅ 복구 완료: {count}개 테이블 복원됨 ({msg})")
            sys.exit(0)
        else:
            print(f"❌ 복구 실패: {msg}", file=sys.stderr)
            sys.exit(1)
    elif cmd == "is_empty":
        print("EMPTY" if db.is_database_empty() else "POPULATED")
    elif cmd == "auto_restore":
        ok, status = db.auto_restore_if_empty()
        print(f"AUTO_RESTORE_RESULT: {status}")
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

