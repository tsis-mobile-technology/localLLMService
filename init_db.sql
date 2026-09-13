-- ============================================================================
-- Local LLM Service Platform - Initial Database Schema
-- ============================================================================

-- 1. 허깅페이스 모델 카탈로그 테이블
CREATE TABLE IF NOT EXISTS hf_model_catalog (
    id VARCHAR(64) PRIMARY KEY,
    repo_id VARCHAR(128) NOT NULL,
    model_name VARCHAR(128) NOT NULL,
    filename VARCHAR(128) NOT NULL,
    quantization VARCHAR(32) NOT NULL,
    size_bytes BIGINT NOT NULL,
    size_gb NUMERIC(6, 2) NOT NULL,
    min_vram_gb NUMERIC(4, 1) NOT NULL,
    category VARCHAR(64),
    description TEXT,
    extra_args TEXT,
    downloads_count BIGINT DEFAULT 0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 2. 모델 다운로드 이력 관리 테이블
CREATE TABLE IF NOT EXISTS model_downloads (
    id SERIAL PRIMARY KEY,
    model_id VARCHAR(64) REFERENCES hf_model_catalog(id) ON DELETE SET NULL,
    filename VARCHAR(128) NOT NULL,
    local_path TEXT NOT NULL,
    download_url TEXT NOT NULL,
    status VARCHAR(32) NOT NULL, -- 'PENDING', 'DOWNLOADING', 'COMPLETED', 'FAILED'
    downloaded_bytes BIGINT DEFAULT 0,
    total_bytes BIGINT DEFAULT 0,
    download_speed_mb NUMERIC(6, 2),
    duration_seconds INTEGER,
    error_message TEXT,
    started_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP WITH TIME ZONE
);

-- 3. 서버 런타임 및 수명주기 로그 테이블
CREATE TABLE IF NOT EXISTS server_runtime_logs (
    id SERIAL PRIMARY KEY,
    container_name VARCHAR(64) NOT NULL,
    model_id VARCHAR(64),
    filename VARCHAR(128),
    port INTEGER NOT NULL,
    hardware_profile VARCHAR(32), -- '100% GPU', 'GPU+CPU', 'CPU-Only'
    gpu_name VARCHAR(128),
    vram_used_mb INTEGER,
    status VARCHAR(32) NOT NULL, -- 'STARTING', 'HEALTHY', 'STOPPED', 'ERROR'
    exit_code INTEGER,
    error_log TEXT,
    started_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    stopped_at TIMESTAMP WITH TIME ZONE
);

-- 인덱스 생성
CREATE INDEX IF NOT EXISTS idx_downloads_status ON model_downloads(status);
CREATE INDEX IF NOT EXISTS idx_downloads_started ON model_downloads(started_at DESC);
CREATE INDEX IF NOT EXISTS idx_runtime_status ON server_runtime_logs(status);
CREATE INDEX IF NOT EXISTS idx_runtime_started ON server_runtime_logs(started_at DESC);
