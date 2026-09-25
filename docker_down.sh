#!/bin/bash
# ============================================================================
# Safely Backup PostgreSQL DB & Stop all local LLM services (docker compose down)
# ============================================================================
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🛑 Local LLM Service 전체 종료 (DB 안전 백업 포함)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# 1. PostgreSQL DB 백업 수행
if [ -f "$SCRIPT_DIR/db_manager.py" ] && docker ps --format '{{.Names}}' 2>/dev/null | grep -q "^litellm-postgres$"; then
    echo "💾 PostgreSQL 데이터 백업 진행 중..."
    if python3 "$SCRIPT_DIR/db_manager.py" backup; then
        echo "✓ DB 백업이 성공적으로 생성되었습니다."
    else
        echo "⚠️ DB 백업 중 경고가 발생했으나 프로세스를 계속 진행합니다."
    fi
else
    echo "ℹ️ PostgreSQL 컨테이너가 실행 중이지 않아 백업을 건너뜁니다."
fi

# 2. LLaMA.cpp 서버 런타임 로그 기록 및 컨테이너 정리
if command -v python3 &>/dev/null && [ -f "$SCRIPT_DIR/db_manager.py" ]; then
    python3 -c "
from db_manager import DBManager
db = DBManager()
if db.is_available():
    db.log_server_stop('llama-server', exit_code=0)
" 2>/dev/null || true
fi

docker stop litellm-proxy llama-server 2>/dev/null || true
docker rm -f litellm-proxy llama-server 2>/dev/null || true

# 3. Docker Compose Down 실행
echo "📦 PostgreSQL 및 Compose 서비스 종료 중 (docker compose down)..."
(cd "$SCRIPT_DIR" && docker compose down)

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ 모든 서비스(PostgreSQL DB 포함)가 안전하게 종료되었습니다."
if [ -f "$SCRIPT_DIR/backups/litellm_backup_latest.sql" ]; then
    size=$(du -h "$SCRIPT_DIR/backups/litellm_backup_latest.sql" 2>/dev/null | awk '{print $1}')
    echo "💾 최신 백업 파일: backups/litellm_backup_latest.sql ($size)"
    echo "🔄 다시 구동할 때 ('./docker_run.sh' 또는 'docker compose up') 백업된 데이터가 자동 복구됩니다."
fi
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
