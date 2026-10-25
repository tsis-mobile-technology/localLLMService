#!/bin/bash
if [ -z "${BASH_VERSION:-}" ]; then
    exec bash "$0" "$@"
fi

# Auto-apply docker group if user is in docker group but current session has not loaded it
if ! docker info &>/dev/null 2>&1; then
    if ! groups | grep -qw "docker" && getent group docker 2>/dev/null | grep -qw "$USER"; then
        cmd=$(printf "%q " "$0" "$@")
        exec sg docker -c "exec bash $cmd"
    fi
fi

set -euo pipefail

# ============================================================================
# Unsloth GGUF Hardware-Optimized Docker Model Launcher + PostgreSQL Platform
# ============================================================================
# 1. PC 환경(CPU, RAM, GPU VRAM, 저장공간) 자동 감지
# 2. Hugging Face unsloth GGUF 모델 중 PC 사양에 최적화된 TOP 10 모델 추천
# 3. 모델 선택 시 자동 다운로드 (이어받기/진행률) 후 Docker 컨테이너 구동
# 4. PostgreSQL 연동: 모델 메타데이터, 다운로드 이력, 서버 런타임 로그,
#    LiteLLM 사용자/관리자 계정 및 발급된 API Key, 토큰 소비 통계 영구 보존
# ============================================================================

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Load environment configuration from .env if present
if [ -f "$SCRIPT_DIR/.env" ]; then
    set -a
    # shellcheck disable=SC1091
    source "$SCRIPT_DIR/.env"
    set +a
fi

readonly MODELS_DIR="${MODELS_DIR:-$HOME/Projects/models}"
export MODELS_DIR
mkdir -p "$MODELS_DIR"

readonly CONTAINER_NAME="llama-server"
readonly DOCKER_IMAGE="ghcr.io/ggml-org/llama.cpp:server-cuda"
readonly PORT=8080
readonly RECOMMENDER="$SCRIPT_DIR/hf_recommender.py"
readonly DB_MANAGER="$SCRIPT_DIR/db_manager.py"
readonly DOCKER_NETWORK="local-llm-net"

# PostgreSQL Integration
readonly POSTGRES_CONTAINER="litellm-postgres"
readonly POSTGRES_PORT=5432
readonly DATABASE_URL="${DATABASE_URL:-postgresql://llm_admin:llm_secret_pass@litellm-postgres:5432/litellm}"

# liteLLM Integration
readonly LITELLM_CONTAINER="litellm-proxy"
readonly LITELLM_PORT=4000
readonly LITELLM_IMAGE="ghcr.io/berriai/litellm:main-latest"
readonly LITELLM_CONFIG="$MODELS_DIR/litellm_config.yaml"
readonly LITELLM_MASTER_KEY="${LITELLM_MASTER_KEY:-sk-local-master}"

# Laya Decision Engine Integration (System 1)
readonly LAYA_CONTAINER="laya-engine"
readonly LAYA_PORT=8001
readonly LAYA_IMAGE="local-laya:latest"
LAYA_ENABLED="${LAYA_ENABLED:-true}"
LAYA_CURRENT_TIER="bypass"

# Docker GPU Support Flag
DOCKER_GPU_SUPPORT=true

# --- Error Analysis Helper ---------------------------------------------------

analyze_docker_error() {
    local err="$1"
    if echo "$err" | grep -qi "failed to discover GPU vendor from CDI\|could not select device driver\|nvidia-container-cli"; then
        echo -e "【원인 분석】\nNVIDIA 그래픽카드는 감지되었으나, Docker 데몬에 'NVIDIA Container Toolkit'이 설치/연동되어 있지 않습니다.\n(에러: failed to discover GPU vendor from CDI)\n\n【해결 명령어 (Ubuntu 24.04)】\n1) curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg\n2) curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list\n3) sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit\n4) sudo nvidia-ctk runtime configure --runtime=docker\n5) sudo systemctl restart docker"
    elif echo "$err" | grep -qi "port is already allocated\|address already in use"; then
        echo -e "【원인 분석】\n포트가 이미 다른 프로세스에서 사용 중입니다.\n\n【해결 방법】\n'sudo lsof -i :$PORT' 명령어로 포트를 점유 중인 프로세스를 확인하고 종료해주세요."
    elif echo "$err" | grep -qi "Conflict. The container name"; then
        echo -e "【원인 분석】\n동일한 이름의 컨테이너가 이미 존재합니다.\n\n【해결 방법】\n'docker rm -f $CONTAINER_NAME' 명령어로 기존 컨테이너를 삭제해주세요."
    else
        echo -e "【원인 분석】\nDocker 실행 파라미터 또는 데몬 상태를 점검해주세요."
    fi
}

# --- Infrastructure & Prerequisites Check ------------------------------------

ensure_infrastructure() {
    # 1. Ensure docker network exists
    if ! docker network inspect "$DOCKER_NETWORK" &>/dev/null; then
        docker network create "$DOCKER_NETWORK" &>/dev/null || true
    fi

    # 2. Ensure PostgreSQL is running
    local was_down=false
    if ! docker ps --format '{{.Names}}' 2>/dev/null | grep -q "^${POSTGRES_CONTAINER}$"; then
        was_down=true
        echo "📦 PostgreSQL 서비스 기동 중 (docker compose up -d postgres)..."
        (cd "$SCRIPT_DIR" && docker compose up -d postgres &>/dev/null || true)
        local retries=15
        while [ $retries -gt 0 ]; do
            if docker exec "$POSTGRES_CONTAINER" pg_isready -U llm_admin -d litellm &>/dev/null; then
                break
            fi
            sleep 1
            retries=$((retries - 1))
        done
    fi

    # 3. Check for automatic restore if database is empty but backup exists
    if [ -f "$DB_MANAGER" ]; then
        local restore_res
        restore_res=$(python3 "$DB_MANAGER" auto_restore 2>/dev/null || echo "")
        if echo "$restore_res" | grep -q "AUTO_RESTORED"; then
            echo "✅ 백업된 PostgreSQL 데이터를 성공적으로 복구했습니다."
        fi
    fi
}

check_gpu_docker_support() {
    if command -v nvidia-smi &>/dev/null; then
        local test_err
        test_err=$(docker run --rm --gpus all hello-world 2>&1 || true)
        if echo "$test_err" | grep -qi "failed to discover GPU vendor from CDI\|could not select device driver"; then
            DOCKER_GPU_SUPPORT=false
            if [ -t 0 ] && [ -t 1 ] && [ -n "${TERM:-}" ] && [ "$TERM" != "dumb" ] && command -v whiptail &>/dev/null; then
                if whiptail --title "⚠️ NVIDIA Container Toolkit 미설치 감지" \
                    --yesno "현재 PC에 NVIDIA GPU(RTX 3060)가 장착되어 있지만, Docker에서 GPU를 사용하기 위한 'nvidia-container-toolkit'이 설치되어 있지 않습니다.\n\n에러: failed to discover GPU vendor from CDI\n\n• [Yes (권장)]: 툴킷 설치 명령어를 확인하고 종료합니다.\n• [No (임시)]: GPU 가속 없이 'CPU 모드'로 임시 구동합니다." \
                    16 75 \
                    --yes-button "설치 가이드 확인" --no-button "CPU 모드로 계속"; then

                    clear
                    print_nvidia_toolkit_guide
                    exit 0
                else
                    DOCKER_GPU_SUPPORT=false
                fi
            else
                print_nvidia_toolkit_guide
                DOCKER_GPU_SUPPORT=false
            fi
        fi
    fi
}

print_nvidia_toolkit_guide() {
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "🛠️  NVIDIA Container Toolkit 설치 가이드 (Ubuntu 24.04 LTS)"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "Docker에서 RTX 3060 GPU 가속을 사용하려면 아래 명령을 실행해주세요:"
    echo ""
    echo "# 1. NVIDIA 리포지토리 추가"
    echo "curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg && \\"
    echo "curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \\"
    echo "  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \\"
    echo "  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list"
    echo ""
    echo "# 2. 툴킷 설치"
    echo "sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit"
    echo ""
    echo "# 3. Docker 런타임 연동"
    echo "sudo nvidia-ctk runtime configure --runtime=docker"
    echo ""
    echo "# 4. Docker 서비스 재시작"
    echo "sudo systemctl restart docker"
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "설치 완료 후 './docker_run.sh'를 실행하시면 100% GPU 가속이 활성화됩니다."
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
}

check_prerequisites() {
    if ! command -v python3 &>/dev/null; then
        echo "❌ Error: python3 is required."
        exit 1
    fi

    if [ ! -f "$RECOMMENDER" ]; then
        echo "❌ Error: $RECOMMENDER not found."
        exit 1
    fi

    if ! command -v whiptail &>/dev/null; then
        echo "Error: whiptail not found. Please install whiptail: sudo apt install whiptail"
        exit 1
    fi

    if ! command -v docker &>/dev/null; then
        whiptail --title "❌ Error" \
            --msgbox "Docker command not found in PATH!" \
            8 50
        exit 1
    fi

    if ! docker info &>/dev/null 2>&1; then
        local docker_err
        docker_err=$(docker info 2>&1 || true)
        if echo "$docker_err" | grep -qi "permission denied"; then
            whiptail --title "❌ Permission Denied" \
                --msgbox "Docker 데몬 접근 권한이 없습니다 (Permission denied).\n\n아래 명령을 실행하여 현재 셸에 그룹 권한을 갱신해주세요:\n  newgrp docker\n\n또는:\n  sudo usermod -aG docker \$USER\n후 로그아웃/로그인해주세요." \
                13 65
        else
            whiptail --title "❌ Error" \
                --msgbox "Docker daemon is not running.\n\nPlease start Docker and try again:\n  sudo systemctl start docker" \
                11 55
        fi
        exit 1
    fi

    check_gpu_docker_support
    ensure_infrastructure
}

# --- Container Status Functions ----------------------------------------------

get_postgres_status() {
    if docker ps --format '{{.Names}}' 2>/dev/null | grep -q "^${POSTGRES_CONTAINER}$"; then
        echo "🟢 DB-UP"
    else
        echo "🔴 DB-DOWN"
    fi
}

get_container_status() {
    if docker ps --format '{{.Names}}' 2>/dev/null | grep -q "^${CONTAINER_NAME}$"; then
        echo "🟢 RUNNING"
    elif docker ps -a --format '{{.Names}}' 2>/dev/null | grep -q "^${CONTAINER_NAME}$"; then
        echo "🔴 STOPPED"
    else
        echo "⚫ NOT RUNNING"
    fi
}

get_litellm_status() {
    if docker ps --format '{{.Names}}' 2>/dev/null | grep -q "^${LITELLM_CONTAINER}$"; then
        echo "🟢 RUNNING"
    elif docker ps -a --format '{{.Names}}' 2>/dev/null | grep -q "^${LITELLM_CONTAINER}$"; then
        echo "🔴 STOPPED"
    else
        echo "⚫ NOT RUNNING"
    fi
}

get_laya_status() {
    if docker ps --format '{{.Names}}' 2>/dev/null | grep -q "^${LAYA_CONTAINER}$"; then
        local dev
        dev=$(docker inspect "$LAYA_CONTAINER" --format '{{range .Config.Env}}{{println .}}{{end}}' 2>/dev/null | grep "^LAYA_DEVICE=" | cut -d= -f2 || echo "auto")
        if [ "$dev" = "cuda" ]; then
            echo "🟢 GPU"
        elif [ "$dev" = "cpu" ]; then
            echo "🟢 CPU"
        else
            echo "🟢 UP"
        fi
    elif docker ps -a --format '{{.Names}}' 2>/dev/null | grep -q "^${LAYA_CONTAINER}$"; then
        echo "🔴 STOPPED"
    else
        echo "⚫ OFF"
    fi
}

stop_existing_container() {
    local status litellm_status
    litellm_status=$(get_litellm_status)
    status=$(get_container_status)

    if [[ "$litellm_status" != "⚫ NOT RUNNING" ]]; then
        docker stop "$LITELLM_CONTAINER" 2>/dev/null || true
        docker rm -f "$LITELLM_CONTAINER" 2>/dev/null || true
    fi

    if docker ps -a --format '{{.Names}}' 2>/dev/null | grep -q "^${LAYA_CONTAINER}$"; then
        docker stop "$LAYA_CONTAINER" 2>/dev/null || true
        docker rm -f "$LAYA_CONTAINER" 2>/dev/null || true
    fi

    if [[ "$status" != "⚫ NOT RUNNING" ]]; then
        # Record stop in DB
        python3 -c "from db_manager import DBManager; db=DBManager(); db.log_server_stop('$CONTAINER_NAME')" 2>/dev/null || true
        docker stop "$CONTAINER_NAME" 2>/dev/null || true
        docker rm -f "$CONTAINER_NAME" 2>/dev/null || true
    fi

    docker rm -f "$CONTAINER_NAME" &>/dev/null || true
}

# --- liteLLM & Laya Configuration --------------------------------------------

ensure_laya_image() {
    if ! docker image inspect "$LAYA_IMAGE" &>/dev/null; then
        echo "📦 Laya 결정 엔진 Docker 이미지 빌드 중 ($LAYA_IMAGE)..."
        (cd "$SCRIPT_DIR" && docker build -t "$LAYA_IMAGE" -f Dockerfile.laya .)
    fi
}

launch_laya() {
    local tier="${1:-gpu}"
    LAYA_CURRENT_TIER="$tier"
    if [ "$tier" = "bypass" ] || [ "$LAYA_ENABLED" = "false" ]; then
        echo "ℹ️  Laya 결정 엔진: 바이패스 모드 (구동 생략)"
        return 0
    fi

    ensure_laya_image

    echo "🚀 Laya 결정 엔진 컨테이너 시작 중 (Tier: $tier)..."
    docker stop "$LAYA_CONTAINER" &>/dev/null || true
    docker rm -f "$LAYA_CONTAINER" &>/dev/null || true

    local laya_gpu_args=()
    if [ "$tier" = "gpu" ] && [ "$DOCKER_GPU_SUPPORT" = "true" ]; then
        laya_gpu_args+=(--gpus all)
        laya_gpu_args+=(-e "LAYA_DEVICE=cuda")
        echo "⚡ Laya 가속 모드: 100% GPU (CUDA, ~30ms 속도)"
    else
        laya_gpu_args+=(-e "LAYA_DEVICE=cpu")
        laya_gpu_args+=(-e "OMP_NUM_THREADS=4")
        echo "⚡ Laya 가속 모드: CPU + 78GB RAM (VRAM 0GB 사용, GPU OOM 방지)"
    fi

    local custom_model_dir="${SCRIPT_DIR}/custom_laya_model"
    local laya_model_args=()
    if [ -d "$custom_model_dir" ] && [ -f "$custom_model_dir/model.safetensors" ]; then
        echo "🎯 Laya 커스텀 파인튜닝 모델 감지: $custom_model_dir"
        laya_model_args+=(-v "${custom_model_dir}:/root/custom_laya_model")
        laya_model_args+=(-e 'LAYA_EXTRA_MODELS={"custom": "/root/custom_laya_model"}')
        laya_model_args+=(-e "LAYA_DEFAULT_MODEL=custom")
        laya_model_args+=(-e "LAYA_MODELS=custom,multilingual,english")
    else
        laya_model_args+=(-e "LAYA_MODELS=multilingual,english")
    fi

    local laya_out
    if ! laya_out=$(docker run -d --name "$LAYA_CONTAINER" \
        --network "$DOCKER_NETWORK" \
        ${laya_gpu_args[@]+"${laya_gpu_args[@]}"} \
        ${laya_model_args[@]+"${laya_model_args[@]}"} \
        -p ${LAYA_PORT}:8000 \
        -v "${HOME}/.cache/huggingface:/root/.cache/huggingface" \
        -e "LAYA_HOST=0.0.0.0" \
        -e "LAYA_PORT=8000" \
        -e "LAYA_PRELOAD=1" \
        -e "HF_HOME=/root/.cache/huggingface" \
        "$LAYA_IMAGE" 2>&1); then

        echo "⚠️ Laya 시작 실패: $laya_out"
        return 1
    fi

    # Laya 헬스체크 대기 (최대 15초)
    local retries=15
    while [ $retries -gt 0 ]; do
        if curl -s "http://localhost:${LAYA_PORT}/health" &>/dev/null; then
            echo "🟢 Laya 결정 엔진 정상 가동 확인 (포트 $LAYA_PORT)"
            break
        fi
        sleep 1
        retries=$((retries - 1))
    done
}

generate_litellm_config() {
    local tier="${1:-gpu}"
    local callback_entry=""
    if [ "$tier" != "bypass" ] && [ "$LAYA_ENABLED" != "false" ]; then
        callback_entry="  callbacks: [\"laya_interceptor.proxy_handler_instance\"]"
    fi

    cat > "$LITELLM_CONFIG" <<YAML
model_list:
  - model_name: "gpt-4o"
    litellm_params:
      model: "openai/local"
      api_base: "http://host.docker.internal:8080"
      api_key: "sk-local"
      max_input_tokens: 32768
      max_tokens: 8192
    model_info:
      max_input_tokens: 32768
      max_tokens: 8192
  - model_name: "local"
    litellm_params:
      model: "openai/local"
      api_base: "http://host.docker.internal:8080"
      api_key: "sk-local"
      max_input_tokens: 32768
      max_tokens: 8192
    model_info:
      max_input_tokens: 32768
      max_tokens: 8192

litellm_settings:
  drop_params: true
  cache: true
  cache_params:
    type: "local"
    supported_call_types: ["completion", "text_completion"]
$callback_entry

general_settings:
  master_key: "$LITELLM_MASTER_KEY"
  database_url: "os.environ/DATABASE_URL"
  store_model_in_db: true
YAML
}

launch_litellm() {
    local tier="${1:-gpu}"
    echo "🚀 liteLLM Proxy 컨테이너 시작 중 (PostgreSQL DB 연동 및 Laya 인터셉터)..."
    docker rm -f "$LITELLM_CONTAINER" &>/dev/null || true

    local laya_flag="$LAYA_ENABLED"
    if [ "$tier" = "bypass" ]; then
        laya_flag="false"
    fi

    local litellm_out
    if ! litellm_out=$(docker run -d --name "$LITELLM_CONTAINER" \
        --network "$DOCKER_NETWORK" \
        --add-host host.docker.internal:host-gateway \
        -p ${LITELLM_PORT}:4000 \
        -v "${LITELLM_CONFIG}:/app/config.yaml" \
        -v "${SCRIPT_DIR}/laya_interceptor.py:/app/laya_interceptor.py:ro" \
        -v "${SCRIPT_DIR}/laya_questions.yaml:/app/laya_questions.yaml:ro" \
        -e "DATABASE_URL=${DATABASE_URL}" \
        -e "LITELLM_MASTER_KEY=${LITELLM_MASTER_KEY}" \
        -e "LAYA_API_URL=http://laya-engine:8000/v1/systemone" \
        -e "LAYA_ENABLED=${laya_flag}" \
        "$LITELLM_IMAGE" \
        --config /app/config.yaml \
        --port 4000 2>&1); then

        echo "⚠️ liteLLM 시작 실패: $litellm_out"
        whiptail --title "⚠️  liteLLM Warning" \
            --msgbox "Failed to launch liteLLM container:\n\n$litellm_out\n\nllama.cpp는 포트 $PORT 에서 정상 작동합니다." \
            14 70
        return 1
    fi

    # LiteLLM 초기 구동 안정화 대기 및 생존 검증
    sleep 3
    if ! docker ps --format '{{.Names}}' 2>/dev/null | grep -q "^${LITELLM_CONTAINER}$"; then
        local crash_log
        crash_log=$(docker logs "$LITELLM_CONTAINER" 2>&1 | tail -n 12)
        echo "❌ liteLLM 컨테이너 비정상 종료:\n$crash_log"
        whiptail --title "⚠️  liteLLM 비정상 종료" \
            --msgbox "liteLLM 컨테이너가 시작 후 비정상 종료되었습니다:\n\n$crash_log\n\nllama.cpp(포트 $PORT)는 계속 사용 가능합니다." \
            16 75
        return 1
    fi
}

ask_litellm_option() {
    whiptail --title "Optional: liteLLM Proxy" \
        --yesno "llama.cpp와 함께 liteLLM 프록시를 실행하시겠습니까?\n\n[PostgreSQL DB 연동 장점]:\n• 다중 사용자(Users) 계정 및 개별 API Key 발급\n• 접속 로그, 토큰 사용량(SpendLogs) DB 영구 적재\n• 통합 웹 UI 대시보드 제공 (http://localhost:4000/ui)\n• 시맨틱 캐싱으로 동일 질문 초고속 응답" \
        17 75 \
        --yes-button "Start liteLLM (권장)" --no-button "Skip"
}

# --- Database Views & Submenus -----------------------------------------------

show_users_view() {
    local users_json
    users_json=$(python3 "$DB_MANAGER" users 2>/dev/null || echo "[]")
    local text
    text=$(python3 -c "
import json, sys
data = json.loads(sys.argv[1])
if not data:
    print('등록된 사용자가 없습니다. LiteLLM Web UI(localhost:4000/ui)에서 사용자를 추가하세요.')
else:
    for u in data:
        print(f'• 사용자 ID: {u[\"user_id\"][:18]}... ({u[\"user_email\"]})')
        print(f'  역할: {u[\"user_role\"]} │ 예산: \${u[\"max_budget\"]} │ 활성 API키: {u[\"active_keys\"]}개 │ 등록: {u[\"created_at\"]}')
        print('')
" "$users_json")

    whiptail --title "👥 LiteLLM 사용자 및 API Key 목록" \
        --msgbox "$text\n[안내] 신규 사용자 추가 및 개별 API 키 발급은 http://localhost:4000/ui 에서 가능합니다." \
        18 80
}

show_stats_view() {
    local stats_json
    stats_json=$(python3 "$DB_MANAGER" stats 2>/dev/null || echo "{}")
    local text
    text=$(python3 -c "
import json, sys
s = json.loads(sys.argv[1])
print(f'• 총 요청 건수: {s.get(\"total_requests\", 0):,} 회')
print(f'• 누적 소비 토큰: {s.get(\"total_tokens\", 0):,} 토큰 (프롬프트: {s.get(\"prompt_tokens\", 0):,}, 생성: {s.get(\"completion_tokens\", 0):,})')
print('')
print('【모델별 호출 통계】')
for m in s.get('models', []):
    print(f'  - {m[\"model\"]}: {m[\"requests\"]}회 ({m[\"tokens\"]} 토큰, 평균 지연시간 {m[\"avg_latency_ms\"]}ms)')
if not s.get('models'):
    print('  (아직 API 호출 기록이 없습니다)')
" "$stats_json")

    whiptail --title "📊 LiteLLM 토큰 소비 및 요청 통계" \
        --msgbox "$text" \
        18 80
}

show_history_view() {
    local dl_json rt_json text
    dl_json=$(python3 "$DB_MANAGER" downloads 2>/dev/null || echo "[]")
    rt_json=$(python3 "$DB_MANAGER" runtimes 2>/dev/null || echo "[]")

    text=$(python3 -c "
import json, sys
dls = json.loads(sys.argv[1])
rts = json.loads(sys.argv[2])

print('【최근 모델 다운로드 이력】')
for d in dls[:4]:
    print(f'  - {d[\"name\"]} ({d[\"size_gb\"]}GB) [{d[\"status\"]}] @ {d[\"started_at\"]}')
if not dls:
    print('  (다운로드 이력이 없습니다)')

print('')
print('【최근 서버 구동 이력】')
for r in rts[:4]:
    print(f'  - {r[\"filename\"]} [{r[\"status\"]}] {r[\"hardware_profile\"]} (가동 {r[\"uptime_min\"]}분) @ {r[\"started_at\"]}')
if not rts:
    print('  (구동 이력이 없습니다)')
" "$dl_json" "$rt_json")

    whiptail --title "📜 모델 다운로드 및 서버 구동 이력" \
        --msgbox "$text" \
        18 80
}

show_backup_restore_view() {
    local bchoice
    bchoice=$(whiptail --title "💾 PostgreSQL 백업 및 복원 관리" \
        --menu "원하는 작업을 선택하세요:" 15 65 3 \
        "1" "💾 지금 즉시 데이터베이스 백업 생성" \
        "2" "🔄 최신 백업본(litellm_backup_latest.sql) 복구" \
        "3" "취소 (메인 메뉴로 돌아가기)" \
        3>&1 1>&2 2>&3) || return 0

    if [ "$bchoice" = "1" ]; then
        local b_out
        b_out=$(python3 "$DB_MANAGER" backup 2>&1 || true)
        whiptail --title "💾 DB 백업 결과" --msgbox "$b_out" 12 70
    elif [ "$bchoice" = "2" ]; then
        if whiptail --title "⚠️ DB 복원 확인" \
            --yesno "최신 백업 파일로부터 데이터베이스를 복구하시겠습니까?\n\n기존 데이터가 백업 시점의 상태로 덮어씌워질 수 있습니다." 12 65; then
            local r_out
            r_out=$(python3 "$DB_MANAGER" restore 2>&1 || true)
            whiptail --title "🔄 DB 복원 결과" --msgbox "$r_out" 12 70
        fi
    fi
}

show_laya_view() {
    local lchoice
    lchoice=$(whiptail --title "🧠 Laya System 1 초고속 결정 엔진 관리" \
        --menu "원하는 작업을 선택하세요:" 17 75 5 \
        "1" "📊 실시간 누적 판별 통계 확인 (성공/차단/지연/신뢰도)" \
        "2" "📋 최근 판별 및 라우팅 이력 조회 (최근 20건)" \
        "3" "💾 오분류/판별 로그 데이터셋 추출 (Export to CSV)" \
        "4" "🚀 커스텀 파인튜닝 실행 (laya_train.py)" \
        "5" "취소 (메인 메뉴로 돌아가기)" \
        3>&1 1>&2 2>&3) || return 0

    if [ "$lchoice" = "1" ]; then
        local s_out
        s_out=$(python3 "$DB_MANAGER" laya-stats 2>&1 || true)
        whiptail --title "📊 Laya 실시간 누적 통계" --msgbox "$s_out" 22 80
    elif [ "$lchoice" = "2" ]; then
        local l_out
        l_out=$(python3 "$DB_MANAGER" laya-logs 20 2>&1 || true)
        whiptail --title "📋 Laya 최근 판별 이력" --msgbox "$l_out" 22 105
    elif [ "$lchoice" = "3" ]; then
        local exp_path="data/laya_train_exported.csv"
        local e_out
        e_out=$(python3 "$DB_MANAGER" laya-export "$exp_path" all 0.0 2>&1 || true)
        whiptail --title "💾 데이터셋 추출 결과" --msgbox "$e_out\n\n저장 경로: $SCRIPT_DIR/$exp_path" 14 75
    elif [ "$lchoice" = "4" ]; then
        whiptail --title "🚀 Laya 로컬 파인튜닝" \
            --msgbox "Laya 모델 파인튜닝을 시작합니다.\n\n터미널에서 실시간 학습 진행률(에포크, 손실값)이 표시됩니다.\n[확인]을 누르면 기본 데이터셋(data/laya_sample_train.csv)으로 학습을 시작합니다." 14 75
        python3 "$SCRIPT_DIR/laya_train.py" --data "$SCRIPT_DIR/data/laya_sample_train.csv"
        echo ""
        read -rp "학습이 완료되었습니다. 계속하려면 Enter를 누르세요..." _
    fi
}

show_hf_account_view() {
    local hf_json text
    hf_json=$(python3 "$RECOMMENDER" hf-status 2>/dev/null || echo "{}")
    text=$(python3 -c "
import json, sys
data = json.loads(sys.argv[1])
if data.get('authenticated'):
    pro_str = ' (PRO 유료 플랜)' if data.get('is_pro') else ' (무료 플랜)'
    print(f'🟢 계정 인증 상태 : 인증 성공 (@{data.get(\"username\")})')
    if data.get('fullname'):
        print(f'• 사용자 이름   : {data.get(\"fullname\")}')
    if data.get('email'):
        print(f'• 등록된 이메일 : {data.get(\"email\")}')
    print(f'• 구독 플랜     : {data.get(\"plan\")}{pro_str}')
    print('• 고속 CDN 전용 가속 및 Gated 모델 다운로드 권한 활성화됨')
elif data.get('configured'):
    print(f'❌ 계정 인증 실패: {data.get(\"message\")}')
    print('• .env 파일의 HF_TOKEN 값을 올바르게 설정해주세요.')
else:
    print('ℹ️ Hugging Face 익명 다운로드 모드')
    print('• .env 파일에 HF_TOKEN을 설정하시면 더 빠른 전용 CDN 속도와')
    print('  Gated 모델(라이선스 동의 모델) 다운로드가 가능합니다.')
    print('• 토큰 발급 링크 : https://huggingface.co/settings/tokens (Read 권한 권장)')
" "$hf_json")

    whiptail --title "🤗 Hugging Face API 연동 상태" \
        --msgbox "$text\n\n[설정 파일]: $SCRIPT_DIR/.env" \
        18 80
}

show_reload_view() {
    local scan_res text
    scan_res=$(python3 "$RECOMMENDER" scan-models 2>/dev/null || echo "{}")
    text=$(python3 -c "
import json, sys
data = json.loads(sys.argv[1])
total = data.get('total_local_files', 0)
new_count = data.get('newly_discovered_count', 0)
models_dir = data.get('models_dir', '')
models = data.get('local_models', [])

print(f'📁 대상 저장소 : {models_dir}')
print(f'✅ 감지된 로컬 GGUF 모델 : 총 {total}개 (새로 발견된 커스텀/Unsloth: {new_count}개)')
print('')
print('【보유 중인 로컬 모델 목록】')
for m in models:
    size = m.get('size_gb', 0)
    tag = m.get('offload_tag', '100% GPU')
    cat = m.get('category', '')
    print(f'  • {m[\"name\"]} ({size}GB) [{tag}] - {cat}')
" "$scan_res")

    whiptail --title "🔄 모델 저장소 스캔 및 새로고침 완료" \
        --msgbox "$text\n\n[확인] 메인 메뉴 목록이 최신 상태로 갱신되었습니다." \
        20 85
}

# --- UI & Model Selection ----------------------------------------------------

show_model_menu() {
    local container_status litellm_status pg_status laya_status status_line env_summary
    container_status=$(get_container_status)
    litellm_status=$(get_litellm_status)
    pg_status=$(get_postgres_status)
    laya_status=$(get_laya_status)
    status_line="PG: $pg_status │ Laya: $laya_status │ llama.cpp: $container_status │ liteLLM: $litellm_status"
    env_summary=$(python3 "$RECOMMENDER" env-summary)

    if [ "$DOCKER_GPU_SUPPORT" = "false" ]; then
        status_line="$status_line  │  ⚠️ CPU모드"
    fi

    # Fetch menu items JSON from recommender
    local menu_json
    menu_json=$(python3 "$RECOMMENDER" whiptail-menu)

    local menu_items=()
    mapfile -t menu_items < <(python3 -c "import json, sys; [print(x) for x in json.loads(sys.argv[1])]" "$menu_json")

    # Add management actions
    menu_items+=("R" "[🔄 새로고침] 모델 저장소 폴더 재스캔 (Reload)")
    menu_items+=("K" "[🤗 계정] Hugging Face API 토큰 및 계정 연동 상태 확인")
    menu_items+=("U" "[👥 사용자] LiteLLM 사용자 계정 및 API Key 목록 조회")
    menu_items+=("S" "[📊 통계] LiteLLM 토큰 소비 및 요청 로그 통계")
    menu_items+=("L" "[🧠 Laya] System 1 결정 엔진 통계, 이력 및 파인튜닝")
    menu_items+=("H" "[📜 이력] 모델 다운로드 및 서버 실행 이력 조회")
    menu_items+=("B" "[💾 백업/복원] PostgreSQL DB 수동 백업 및 복구 관리")


    local choice
    choice=$(whiptail \
        --title "🦙 Unsloth GGUF 모델 추천 & 실행 (PostgreSQL 통합 플랫폼)" \
        --menu "서비스 상태: $status_line\n$env_summary\n\n모델을 선택하세요 ([✓ 보유] 즉시구동, [⬇ 다운] 선택 시 자동 다운로드 후 구동):" \
        33 118 16 \
        "${menu_items[@]}" \
        3>&1 1>&2 2>&3) || return 1

    echo "$choice"
}

confirm_and_prepare_model() {
    local choice="$1"
    local model_info
    model_info=$(python3 "$RECOMMENDER" get-model "$choice")

    local name filename size_gb is_dl offload_tag desc lora_file
    name=$(python3 -c "import json, sys; d=json.loads(sys.argv[1]); print(d['name'])" "$model_info")
    filename=$(python3 -c "import json, sys; d=json.loads(sys.argv[1]); print(d['filename'])" "$model_info")
    size_gb=$(python3 -c "import json, sys; d=json.loads(sys.argv[1]); print(d['size_gb'])" "$model_info")
    is_dl=$(python3 -c "import json, sys; d=json.loads(sys.argv[1]); print(str(d['is_downloaded']).lower())" "$model_info")
    offload_tag=$(python3 -c "import json, sys; d=json.loads(sys.argv[1]); print(d['offload_tag'])" "$model_info")
    desc=$(python3 -c "import json, sys; d=json.loads(sys.argv[1]); print(d['desc'])" "$model_info")
    lora_file=$(python3 -c "import json, sys; d=json.loads(sys.argv[1]); print(d.get('lora_file', ''))" "$model_info")

    if [ "$is_dl" = "true" ]; then
        local lora_info=""
        if [ -n "$lora_file" ]; then
            lora_info="\n• LoRA 어댑터: $lora_file (동시 결합 로드)"
        fi
        whiptail --title "🚀 모델 구동 확인" \
            --yesno "선택한 모델이 로컬에 이미 준비되어 있습니다.\n\n• 모델명: $name\n• 파일명: $filename${lora_info}\n• 크기: ${size_gb} GB\n• 적합도: [$offload_tag]\n• 저장 경로: $MODELS_DIR/$filename\n• 서비스 포트: http://localhost:$PORT\n\n이 모델로 컨테이너를 구동하시겠습니까?" \
            17 75
        return $?
    else
        if whiptail --title "📥 모델 다운로드 및 구동 확인" \
            --yesno "선택한 모델을 Hugging Face(Unsloth)에서 다운로드하여 구동합니다.\n\n• 모델명: $name\n• 파일명: $filename\n• 예상 크기: 약 ${size_gb} GB\n• 하드웨어 적합도: [$offload_tag]\n• 저장 경로: $MODELS_DIR\n• 설명: $desc\n\n[다운로드 완료 후 자동으로 Docker 컨테이너가 구동되며 PostgreSQL에 이력이 저장됩니다]\n다운로드를 진행하시겠습니까?" \
            18 80; then

            echo ""
            echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
            echo "📥 Unsloth GGUF 모델 다운로드 시작: $name"
            echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

            if ! python3 "$RECOMMENDER" download "$choice"; then
                whiptail --title "❌ 다운로드 실패" \
                    --msgbox "모델 다운로드 중 오류가 발생했습니다.\n인터넷 연결 및 디스크 여유 공간을 확인해주세요." \
                    10 65
                return 1
            fi

            echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
            echo "✅ 다운로드 완료! 컨테이너 구동 단계로 진행합니다."
            echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
            sleep 1
            return 0
        else
            return 1
        fi
    fi
}

launch_model() {
    local choice="$1"
    local model_info
    model_info=$(python3 "$RECOMMENDER" get-model "$choice")

    local mid filename extra_args offload_tag lora_file
    mid=$(python3 -c "import json, sys; d=json.loads(sys.argv[1]); print(d.get('id', ''))" "$model_info")
    filename=$(python3 -c "import json, sys; d=json.loads(sys.argv[1]); print(d['filename'])" "$model_info")
    extra_args=$(python3 -c "import json, sys; d=json.loads(sys.argv[1]); print(d.get('args', ''))" "$model_info")
    offload_tag=$(python3 -c "import json, sys; d=json.loads(sys.argv[1]); print(d.get('offload_tag', ''))" "$model_info")
    lora_file=$(python3 -c "import json, sys; d=json.loads(sys.argv[1]); print(d.get('lora_file', ''))" "$model_info")

    local gpu_info
    gpu_info=$(python3 -c "import json, sys; d=json.loads(sys.argv[1]); print(f\"{d['gpu']['count']},{d['gpu']['total_vram_gb']},{d['gpu']['name']}\")" "$(python3 "$RECOMMENDER" env)")
    local gpu_count="${gpu_info%%,*}"
    local gpu_name="${gpu_info##*,}"

    local shm_size="16g"
    local memory_limit="24g"
    local threads="8"

    if [ "$offload_tag" = "100% GPU" ]; then
        shm_size="16g"
        memory_limit="24g"
        threads="8"
    else
        shm_size="32g"
        memory_limit="56g"
        threads="10"
    fi

    local gpu_args=()
    if [ "$gpu_count" -ge 1 ] && [ "$DOCKER_GPU_SUPPORT" = "true" ]; then
        gpu_args+=(--gpus all)
        gpu_args+=(-e "CUDA_VISIBLE_DEVICES=0")
        gpu_args+=(-e "NVIDIA_TF32=1")
        gpu_args+=(-e "NVIDIA_DISABLE_MPS=0")
    else
        extra_args=$(echo "$extra_args" | sed -E 's/-ngl [0-9]+//g')
        threads=$(nproc 2>/dev/null || echo "8")
        echo "ℹ️  NVIDIA Container Toolkit 미설치로 인해 CPU 모드로 구동합니다."
    fi

    local hf_token_env=()
    if [ -n "${HF_TOKEN:-}" ]; then
        hf_token_env+=(-e "HF_TOKEN=$HF_TOKEN")
    fi

    local model_arg="-m /models/$filename"
    if [ -n "$lora_file" ]; then
        model_arg="$model_arg --lora /models/$lora_file"
        echo "🔗 LoRA 어댑터 결합 로드: /models/$lora_file"
    fi

    echo "🚀 Docker 컨테이너 실행 중 ($CONTAINER_NAME)..."
    local docker_output
    if ! docker_output=$(docker run -d --name "$CONTAINER_NAME" \
        --network "$DOCKER_NETWORK" \
        ${gpu_args[@]+"${gpu_args[@]}"} \
        ${hf_token_env[@]+"${hf_token_env[@]}"} \
        --cap-add IPC_LOCK \
        --cap-add SYS_ADMIN \
        --ulimit memlock=-1:-1 \
        --ulimit stack=67108864 \
        --shm-size "$shm_size" \
        --memory "$memory_limit" \
        -p ${PORT}:8080 \
        -v "${MODELS_DIR}:/models" \
        -e "LLAMA_CACHE=/models" \
        "$DOCKER_IMAGE" \
        $model_arg \
        --host 0.0.0.0 \
        --threads "$threads" \
        --parallel 1 \
        $extra_args 2>&1); then

        echo ""
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "❌ [Docker Error] 컨테이너 구동 실패 상세:"
        echo "$docker_output"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

        docker rm -f "$CONTAINER_NAME" &>/dev/null || true

        local guide
        guide=$(analyze_docker_error "$docker_output")

        whiptail --title "❌ Docker 컨테이너 구동 실패" \
            --msgbox "Docker 실행 중 아래 오류가 발생했습니다:\n\n$docker_output\n\n$guide" \
            20 80
        return 1
    fi

    # 컨테이너 시작 직후 비정상 종료 검증
    sleep 2
    if ! docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
        echo "❌ llama.cpp 컨테이너가 시작 직후 비정상 종료되었습니다."
        echo "【컨테이너 에러 로그】"
        docker logs "$CONTAINER_NAME"
        docker rm -f "$CONTAINER_NAME" &>/dev/null || true
        return 1
    fi

    # Record server launch in PostgreSQL
    local log_fn="$filename"
    if [ -n "$lora_file" ]; then
        log_fn="$filename (+LoRA: $lora_file)"
    fi
    python3 -c "
from db_manager import DBManager
db = DBManager()
if db.is_available():
    db.log_server_start('$CONTAINER_NAME', '$mid', '$log_fn', $PORT, '$offload_tag', '$gpu_name')
" 2>/dev/null || true
}

show_launch_success() {
    local choice="$1"
    local tier="${2:-$LAYA_CURRENT_TIER}"
    local model_info
    model_info=$(python3 "$RECOMMENDER" get-model "$choice")
    local name filename lora_file
    name=$(python3 -c "import json, sys; d=json.loads(sys.argv[1]); print(d['name'])" "$model_info")
    filename=$(python3 -c "import json, sys; d=json.loads(sys.argv[1]); print(d['filename'])" "$model_info")
    lora_file=$(python3 -c "import json, sys; d=json.loads(sys.argv[1]); print(d.get('lora_file', ''))" "$model_info")

    local litellm_status litellm_msg
    litellm_status=$(get_litellm_status)

    if [[ "$litellm_status" == "🟢 RUNNING" ]]; then
        litellm_msg="\nliteLLM Proxy: http://localhost:$LITELLM_PORT (클라이언트 연동용)\nliteLLM UI:   http://localhost:$LITELLM_PORT/ui (마스터 키: $LITELLM_MASTER_KEY)"
    else
        litellm_msg=""
    fi

    local laya_msg=""
    local laya_st
    laya_st=$(get_laya_status)
    if [[ "$laya_st" != "⚫ OFF" && "$laya_st" != "🔴 STOPPED" ]]; then
        laya_msg="\n• Laya 엔진:   🟢 $laya_st 모드 (포트 $LAYA_PORT, 사전 가드레일/의도분류 가속)"
    elif [ "$tier" = "bypass" ]; then
        laya_msg="\n• Laya 엔진:   ⚪ 바이패스 모드 (비활성)"
    fi

    local mode_msg=""
    if [ "$DOCKER_GPU_SUPPORT" = "false" ]; then
        mode_msg="\n[모드]: CPU 모드 (GPU 툴킷 미설치)"
    else
        mode_msg="\n[모드]: 100% GPU 가속 (CUDA)"
    fi

    local lora_msg=""
    if [ -n "$lora_file" ]; then
        lora_msg="\n• 베이스 모델: $filename\n• LoRA 어댑터: $lora_file (결합 구동)"
    fi

    whiptail --title "✅ 실행 성공 (PostgreSQL & Laya 연동 완료)" \
        --yesno "LLaMA.cpp 및 서비스가 성공적으로 실행되었습니다!\n\n• 모델:       $name${lora_msg}\n• llama.cpp:   http://localhost:$PORT\n• PostgreSQL: 🟢 DB 저장 연동 완료 (포트 $POSTGRES_PORT)${laya_msg}${mode_msg}${litellm_msg}\n\n컨테이너 실시간 로그를 확인하시겠습니까?\n(종료하려면 Ctrl+C)" \
        22 80 \
        --yes-button "Tail Logs" --no-button "Exit"
}

# --- Main Entry Point --------------------------------------------------------

main() {
    check_prerequisites

    # CLI pre-parsing
    while [ $# -gt 0 ]; do
        case "$1" in
            --no-laya)
                export LAYA_ENABLED=false
                echo "ℹ️  Laya 결정 엔진 비활성화 (--no-laya)"
                shift
                ;;
            --laya-status)
                echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
                echo "🧠 Laya 결정 엔진 (System 1) 상태 점검:"
                echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
                local l_st
                l_st=$(get_laya_status)
                echo "• 컨테이너 상태: $l_st (포트: $LAYA_PORT)"
                if docker ps --format '{{.Names}}' 2>/dev/null | grep -q "^${LAYA_CONTAINER}$"; then
                    echo "• 헬스체크 응답:"
                    curl -s "http://localhost:${LAYA_PORT}/health" | jq . 2>/dev/null || curl -s "http://localhost:${LAYA_PORT}/health"
                else
                    echo "• 현재 가동 중이지 않습니다."
                fi
                echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
                exit 0
                ;;
            *)
                break
                ;;
        esac
    done

    # CLI options
    if [ "${1:-}" = "--reload" ] || [ "${1:-}" = "--scan" ]; then
        python3 "$RECOMMENDER" scan-summary
        exit 0
    fi

    if [ "${1:-}" = "--hf-check" ]; then
        python3 "$RECOMMENDER" hf-check
        exit 0
    fi

    if [ "${1:-}" = "--env" ]; then
        python3 "$RECOMMENDER" env-detail
        exit 0
    fi

    if [ "${1:-}" = "--list" ]; then
        python3 "$RECOMMENDER" env-detail
        echo ""
        echo "📋 추천 GGUF 모델 전체 목록 (하드웨어 적합도 순):"
        python3 "$RECOMMENDER" recommend | jq -r 'to_entries[] | "  \(.key + 1). \(.value.name) (\(.value.size_gb)GB) [\(.value.offload_tag)] - \(.value.category): \(.value.desc) (다운로드 여부: \(.value.is_downloaded))"'
        exit 0
    fi

    if [ "${1:-}" = "--users" ]; then
        echo "👥 LiteLLM 사용자 계정 및 API Key 목록 (PostgreSQL):"
        python3 "$DB_MANAGER" users
        exit 0
    fi

    if [ "${1:-}" = "--stats" ]; then
        echo "📊 LiteLLM 토큰 소비 및 요청 로그 통계 (PostgreSQL):"
        python3 "$DB_MANAGER" stats
        exit 0
    fi

    if [ "${1:-}" = "--history" ]; then
        echo "📜 모델 다운로드 이력:"
        python3 "$DB_MANAGER" downloads
        echo ""
        echo "🖥️ 서버 런타임 이력:"
        python3 "$DB_MANAGER" runtimes
        exit 0
    fi

    if [ "${1:-}" = "--backup" ]; then
        python3 "$DB_MANAGER" backup "${2:-}"
        exit 0
    fi

    if [ "${1:-}" = "--restore" ]; then
        python3 "$DB_MANAGER" restore "${2:-}"
        exit 0
    fi

    if [ "${1:-}" = "--laya-stats" ]; then
        python3 "$DB_MANAGER" laya-stats
        exit 0
    fi

    if [ "${1:-}" = "--laya-logs" ]; then
        python3 "$DB_MANAGER" laya-logs "${2:-20}"
        exit 0
    fi

    if [ "${1:-}" = "--laya-export" ]; then
        python3 "$DB_MANAGER" laya-export "${2:-data/laya_train_exported.csv}" "${3:-all}" "${4:-0.0}"
        exit 0
    fi

    if [ "${1:-}" = "--train-laya" ]; then
        python3 "$SCRIPT_DIR/laya_train.py" "${@:2}"
        exit 0
    fi


    # Direct launch by number (e.g. ./docker_run.sh 1)
    if [ $# -gt 0 ] && [[ "$1" =~ ^[0-9]+$ ]]; then
        local direct_choice="$1"
        python3 "$RECOMMENDER" env-detail
        echo ""
        echo "선택 모델 ID: $direct_choice"
        if ! python3 "$RECOMMENDER" get-model "$direct_choice" &>/dev/null; then
            echo "❌ 잘못된 모델 번호입니다."
            exit 1
        fi

        local direct_model_info is_dl
        direct_model_info=$(python3 "$RECOMMENDER" get-model "$direct_choice")
        is_dl=$(python3 -c "import json, sys; d=json.loads(sys.argv[1]); print(str(d.get('is_downloaded', False)).lower())" "$direct_model_info")

        if [ "$is_dl" != "true" ]; then
            python3 "$RECOMMENDER" download "$direct_choice"
        fi

        # Determine Laya Smart Tier
        local tier_json laya_tier laya_reason
        tier_json=$(python3 "$RECOMMENDER" laya-tier "$direct_choice")
        laya_tier=$(python3 -c "import json, sys; d=json.loads(sys.argv[1]); print(d.get('tier', 'cpu'))" "$tier_json")
        laya_reason=$(python3 -c "import json, sys; d=json.loads(sys.argv[1]); print(d.get('reason', ''))" "$tier_json")
        echo "🧠 Laya 스마트 3-Tier 결정: [$laya_tier] $laya_reason"

        stop_existing_container
        launch_model "$direct_choice"
        launch_laya "$laya_tier" || true
        generate_litellm_config "$laya_tier"
        launch_litellm "$laya_tier" || true
        echo "✅ 모델 구동 완료! llama.cpp: http://localhost:$PORT | Laya: http://localhost:$LAYA_PORT ($laya_tier) | liteLLM: http://localhost:$LITELLM_PORT"
        exit 0
    fi

    # Interactive Whiptail Loop
    while true; do
        local choice use_litellm
        choice=$(show_model_menu) || exit 0

        # Handle management views
        if [ "$choice" = "R" ]; then
            show_reload_view
            continue
        elif [ "$choice" = "K" ]; then
            show_hf_account_view
            continue
        elif [ "$choice" = "U" ]; then
            show_users_view
            continue
        elif [ "$choice" = "S" ]; then
            show_stats_view
            continue
        elif [ "$choice" = "L" ]; then
            show_laya_view
            continue
        elif [ "$choice" = "H" ]; then
            show_history_view
            continue
        elif [ "$choice" = "B" ]; then
            show_backup_restore_view
            continue
        fi

        # Handle model launch
        if confirm_and_prepare_model "$choice"; then
            local tier_json laya_tier laya_reason
            tier_json=$(python3 "$RECOMMENDER" laya-tier "$choice")
            laya_tier=$(python3 -c "import json, sys; d=json.loads(sys.argv[1]); print(d.get('tier', 'cpu'))" "$tier_json")
            laya_reason=$(python3 -c "import json, sys; d=json.loads(sys.argv[1]); print(d.get('reason', ''))" "$tier_json")

            if ask_litellm_option; then
                use_litellm=true
            else
                use_litellm=false
            fi

            stop_existing_container
            launch_model "$choice" || exit 1

            if [[ "$use_litellm" == "true" ]]; then
                echo "🧠 Laya 스마트 3-Tier 결정: [$laya_tier] $laya_reason"
                launch_laya "$laya_tier" || true
                generate_litellm_config "$laya_tier"
                sleep 2
                launch_litellm "$laya_tier" || true
            fi

            if show_launch_success "$choice" "$laya_tier"; then
                echo ""
                echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
                echo "📋 Container logs (Ctrl+C to stop tailing): $CONTAINER_NAME"
                echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
                echo ""
                docker logs -f "$CONTAINER_NAME" || true
            fi

            exit 0
        else
            continue
        fi
    done
}

main "$@"
