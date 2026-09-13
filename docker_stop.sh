#!/bin/bash
# ============================================================================
# Stop local LLM service containers & update database runtime logs
# ============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Record server stop in PostgreSQL if available
if command -v python3 &>/dev/null && [ -f "$SCRIPT_DIR/db_manager.py" ]; then
    python3 -c "
from db_manager import DBManager
db = DBManager()
if db.is_available():
    db.log_server_stop('llama-server', exit_code=0)
" 2>/dev/null || true
fi

# Stop and remove litellm-proxy and llama-server containers
docker stop litellm-proxy 2>/dev/null || true
docker rm -f litellm-proxy 2>/dev/null || true

docker stop llama-server 2>/dev/null || true
docker rm -f llama-server 2>/dev/null || true

echo "✓ LLaMA.cpp 및 LiteLLM 컨테이너가 정상 종료되었습니다."
echo "ℹ️ PostgreSQL DB(litellm-postgres)는 데이터 보존을 위해 상시 유지됩니다."
echo "  (DB까지 완전히 종료하려면 'docker compose down'을 실행하세요)"
