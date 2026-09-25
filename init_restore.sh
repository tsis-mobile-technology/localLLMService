#!/bin/bash
set -e

BACKUP_FILE="/backups/litellm_backup_latest.sql"

if [ -f "$BACKUP_FILE" ] && [ -s "$BACKUP_FILE" ]; then
    echo "=========================================================="
    echo "🔄 [Auto-Restore] 백업 파일 발견: $BACKUP_FILE"
    echo "   PostgreSQL 데이터베이스 복구를 시작합니다..."
    echo "=========================================================="
    psql -v ON_ERROR_STOP=0 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" -f "$BACKUP_FILE"
    echo "=========================================================="
    echo "✅ [Auto-Restore] 데이터베이스 복구가 완료되었습니다."
    echo "=========================================================="
else
    echo "ℹ️ [Auto-Restore] 백업 파일이 없습니다. 기본 init_db.sql로 초기화됩니다."
fi
