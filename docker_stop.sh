#!/bin/bash
# ============================================================================
# Stop local LLM service containers & update database runtime logs & auto-backup
# ============================================================================
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# If --all, -a, or --down option is passed, run full teardown via docker_down.sh
if [ "${1:-}" = "--all" ] || [ "${1:-}" = "-a" ] || [ "${1:-}" = "--down" ]; then
    exec "$SCRIPT_DIR/docker_down.sh"
fi

# 1. Record server stop in PostgreSQL if available
if command -v python3 &>/dev/null && [ -f "$SCRIPT_DIR/db_manager.py" ]; then
    python3 -c "
from db_manager import DBManager
db = DBManager()
if db.is_available():
    db.log_server_stop('llama-server', exit_code=0)
" 2>/dev/null || true
fi

# 2. Auto-backup PostgreSQL DB if container is running
if [ "${1:-}" != "--skip-backup" ] && [ -f "$SCRIPT_DIR/db_manager.py" ] && docker ps --format '{{.Names}}' 2>/dev/null | grep -q "^litellm-postgres$"; then
    echo "💾 PostgreSQL 데이터 자동 백업 중..."
    python3 "$SCRIPT_DIR/db_manager.py" backup || true
fi

# 3. Stop and remove litellm-proxy and llama-server containers
docker stop litellm-proxy 2>/dev/null || true
docker rm -f litellm-proxy 2>/dev/null || true

docker stop llama-server 2>/dev/null || true
docker rm -f llama-server 2>/dev/null || true

echo ""
echo "✓ LLaMA.cpp 및 LiteLLM 컨테이너가 정상 종료되었습니다."
if [ -f "$SCRIPT_DIR/backups/litellm_backup_latest.sql" ]; then
    size=$(du -h "$SCRIPT_DIR/backups/litellm_backup_latest.sql" 2>/dev/null | awk '{print $1}')
    echo "💾 PostgreSQL DB가 'backups/litellm_backup_latest.sql' ($size)에 최신 백업되었습니다."
fi
echo "ℹ️ PostgreSQL DB(litellm-postgres)까지 완전히 종료하려면:"
echo "  • './docker_stop.sh --all' 또는 './docker_down.sh' (DB 백업 후 완전 종료)"
echo "  • 또는 'docker compose down'"
echo "  (재시작 시 './docker_run.sh' 또는 'docker compose up'에서 백업된 데이터가 자동 복구됩니다)"
