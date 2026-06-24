#!/bin/bash
set -euo pipefail

# ============================================================================
# LLaMA.cpp Interactive Docker Model Launcher (Multi-Hardware Support)
# ============================================================================
# Auto-detects hardware profile (LOW/MEDIUM/HIGH) and optimizes parameters.
# Supports: < 12GB (Gemma 4 E4B), 12-24GB (Gemma 4 31B), > 24GB (Qwen 35B A3B)
# ============================================================================

# --- Constants ---------------------------------------------------------------
readonly MODELS_DIR="$HOME/Programming/models"
readonly CONTAINER_NAME="llama-server"
readonly DOCKER_IMAGE="ghcr.io/ggml-org/llama.cpp:server-cuda"
readonly PORT=8080

# Hardware profile detection (will be set by detect_hardware_profile)
HARDWARE_PROFILE=""
GPU_MEMORY_GB=0
SYSTEM_MEMORY_GB=0
GPU_COUNT=0          # Number of CUDA devices detected on this system
CUDA_DEVICES=""      # Comma-separated device list (e.g. "0" or "0,1,2,3")
GPU_TOTAL_MEMORY_GB=0 # Aggregate VRAM across all detected GPUs (e.g. 2x 96GB = 192GB)

# liteLLM Integration (optional proxy layer for caching/logging/UI)
readonly LITELLM_CONTAINER="litellm-proxy"
readonly LITELLM_PORT=4000
readonly LITELLM_IMAGE="ghcr.io/berriai/litellm:main-latest"
readonly LITELLM_CONFIG="$MODELS_DIR/litellm_config.yaml"
readonly LITELLM_MASTER_KEY="sk-local-master"

# --- Model Definitions (parallel arrays) -----------------------------------
# Optimized for NVIDIA GB10 Grace Blackwell: 128GB LPDDRX, 2x Superchips, ConnectX7
# GLM-5.2-GGUF from unsloth: https://huggingface.co/unsloth/GLM-5.2-GGUF
readonly MODEL_NAMES=(
    "GLM-5.2 Q8_0 (Recommended)"
    "GLM-5.2 BF16 (GB10 High)"
    "GLM-5.2 IQ3_XXS (LOW VRAM)"
    "Gemma 4 E4B Q4"
    "Gemma 4 E4B Q8"
    "Gemma 4 31B"
    "Gemma 4 26B A4B"
    "Qwen 3.6 35B A3B"
)

readonly MODEL_FILES=(
    "GLM-5.2-UD-Q8_0-00001-of-00017.gguf"
    "GLM-5.2-BF16-00001-of-00033.gguf"
    "GLM-5.2-UD-IQ3_XXS-00001-of-00007.gguf"
    "google_gemma-4-E4B-it-Q4_K_M.gguf"
    "google_gemma-4-E4B-it-Q8_0.gguf"
    "google_gemma-4-31B-it-Q4_K_M.gguf"
    "google_gemma-4-26B-A4B-it-Q4_K_M.gguf"
    "Qwen_Qwen3.6-35B-A3B-Q4_0.gguf"
)

readonly MODEL_DESCS=(
    "Q8_0  │ 1M ctx │ 8-bit Quant   │ 5.2B Balanced (17 files)"
    "BF16  │ 1M ctx │ Full Precision │ 5.2B Quality (33 files)"
    "IQ3_XXS │ 1M ctx │ Ultra Compact │ 5.2B Tiny (7 files)"
    "Q4_K_M  │ 131K ctx │ Full GPU    │ 4B"
    "Q8_0  │ 131K ctx │ Full GPU      │ 4B Premium"
    "Q4_K_M │ 256K ctx │ Full GPU      │ 31B High Performance"
    "Q4_K_M │ 256K ctx │ MoE GPU       │ 26B MoE Hybrid"
    "Q4_0  │ 256K ctx │ MoE GPU       │ 35B MoE Advanced"
)

# Model-specific arguments optimized per hardware profile
# Format: get_model_args <model_index> returns appropriate args for current HARDWARE_PROFILE
get_model_args() {
    local model_idx=$1

    case "$HARDWARE_PROFILE" in
        LOW)  # ≤ 12GB VRAM: q4_0 KV cache keeps context cheap; IDE-friendly sizes
            # NOTE: contexts raised to handle IDE/agent workloads (VS Code, Copilot, etc.)
            # which routinely send 40K+ token prompts. With q4_0 cache (and Gemma SWA),
            # KV growth is small — e.g. E4B Q8 @ 131072 measured ~7GB on a 12GB GPU.
            case $model_idx in
                0)  echo "--no-mmap --cache-type-k q4_0 --cache-type-v q4_0 --mlock -c 65536 -n 1024" ;;
                1)  echo "--no-mmap --cache-type-k q4_0 --cache-type-v q4_0 --mlock -c 32768 -n 1024" ;;
                2)  echo "--no-mmap --cache-type-k q4_0 --cache-type-v q4_0 --mlock -c 131072 -n 1024" ;;
                3)  echo "--no-mmap --cache-type-k q4_0 --cache-type-v q4_0 --mlock -c 131072 -n 1024" ;;
                4)  echo "--no-mmap --cache-type-k q4_0 --cache-type-v q4_0 --mlock -c 131072 -n 1024" ;;
                5)  echo "--n-gpu-layers 8 --n-cpu-moe 20 --cache-type-k q4_0 --cache-type-v q4_0 -c 65536 -n 1024" ;;
                6)  echo "--n-gpu-layers 8 --n-cpu-moe 20 --cache-type-k q4_0 --cache-type-v q4_0 -c 65536 -n 1024" ;;
                7)  echo "--n-gpu-layers 8 --n-cpu-moe 20 --cache-type-k q4_0 --cache-type-v q4_0 -c 65536 -n 1024" ;;
            esac
            ;;
        MEDIUM)  # 12-24GB VRAM: Balanced layers, moderate context
            case $model_idx in
                0)  echo "--no-mmap --cache-type-k q4_0 --cache-type-v q4_0 --mlock -c 131072 -n 4096" ;;
                1)  echo "--no-mmap --cache-type-k f16 --cache-type-v f16 --mlock -c 65536 -n 2048" ;;
                2)  echo "--no-mmap --cache-type-k q4_0 --cache-type-v q4_0 --mlock -c 65536 -n 1024" ;;
                3)  echo "--no-mmap --cache-type-k q4_0 --cache-type-v q4_0 --mlock -c 65536 -n 4096" ;;
                4)  echo "--no-mmap --cache-type-k q8_0 --cache-type-v q8_0 --mlock -c 65536 -n 4096" ;;
                5)  echo "--n-gpu-layers 32 -c 128000 --cache-type-k q4_0 --cache-type-v q4_0 -n 8192" ;;
                6)  echo "--n-gpu-layers 32 -c 128000 --cache-type-k q4_0 --cache-type-v q4_0 -n 8192" ;;
                7)  echo "--n-gpu-layers 32 -c 128000 --cache-type-k q4_0 --cache-type-v q4_0 -n 8192" ;;
            esac
            ;;
        HIGH)  # > 24GB VRAM (Grace Blackwell): Full layers, large context, high precision
            case $model_idx in
                0)  echo "--no-mmap --cache-type-k f16 --cache-type-v f16 --mlock -c 262144 -n 8192" ;;
                1)  echo "--no-mmap --cache-type-k f16 --cache-type-v f16 --mlock -c 262144 -n 8192" ;;
                2)  echo "--no-mmap --cache-type-k q4_0 --cache-type-v q4_0 --mlock -c 262144 -n 4096" ;;
                3)  echo "--no-mmap --cache-type-k q4_0 --cache-type-v q4_0 --mlock -c 131072 -n 8192" ;;
                4)  echo "--no-mmap --cache-type-k q8_0 --cache-type-v q8_0 --mlock -c 131072 -n 8192" ;;
                5)  echo "--n-gpu-layers 64 -c 256000 --cache-type-k q4_0 --cache-type-v q4_0 -n 16384" ;;
                6)  echo "--n-gpu-layers 64 -c 256000 --cache-type-k q4_0 --cache-type-v q4_0 -n 16384" ;;
                7)  echo "--n-gpu-layers 64 -c 256000 --cache-type-k q4_0 --cache-type-v q4_0 -n 16384" ;;
            esac
            ;;
    esac
}

# --- Hardware Detection Functions -------------------------------------------

# Detect available GPU memory (per-GPU total, GB) of the first device
detect_gpu_memory() {
    if command -v nvidia-smi &>/dev/null; then
        local mem_mb
        mem_mb=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -n1)
        # Guard against empty/non-numeric output (e.g. driver not ready)
        if [[ "$mem_mb" =~ ^[0-9]+$ ]]; then
            echo $((mem_mb / 1024))
        else
            echo 0
        fi
    else
        echo 0
    fi
}

# Detect aggregate GPU memory (GB) summed across all CUDA devices.
# With layer-split tensor parallelism the model is shared across GPUs,
# so combined VRAM is what determines how large a model/context can run.
detect_total_gpu_memory() {
    if command -v nvidia-smi &>/dev/null; then
        local total_mb
        total_mb=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null \
            | awk 'BEGIN{s=0} /^[0-9]+$/{s+=$1} END{print s}')
        if [[ "$total_mb" =~ ^[0-9]+$ ]] && [ "$total_mb" -gt 0 ]; then
            echo $((total_mb / 1024))
        else
            echo 0
        fi
    else
        echo 0
    fi
}

# Detect the number of CUDA-capable GPUs present on this system
detect_gpu_count() {
    if command -v nvidia-smi &>/dev/null; then
        local count
        count=$(nvidia-smi --query-gpu=index --format=csv,noheader,nounits 2>/dev/null | grep -c '^[0-9]')
        echo "${count:-0}"
    else
        echo 0
    fi
}

# Build a comma-separated CUDA device list for the detected GPU count
# e.g. 1 GPU -> "0", 2 GPUs -> "0,1", 4 GPUs -> "0,1,2,3"
build_cuda_devices() {
    local n=$1
    if [ "$n" -le 0 ]; then
        echo ""
    else
        seq -s, 0 $((n - 1))
    fi
}

# Detect system memory
detect_system_memory() {
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        free -g | awk '/^Mem:/{print $2}'
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        sysctl -n hw.memsize 2>/dev/null | awk '{print int($1/1024/1024/1024)}'
    else
        echo 0
    fi
}

# Determine hardware profile based on GPU memory
detect_hardware_profile() {
    GPU_MEMORY_GB=$(detect_gpu_memory)              # per-GPU VRAM (display)
    GPU_TOTAL_MEMORY_GB=$(detect_total_gpu_memory)  # aggregate VRAM (profile decision)
    SYSTEM_MEMORY_GB=$(detect_system_memory)
    GPU_COUNT=$(detect_gpu_count)
    CUDA_DEVICES=$(build_cuda_devices "$GPU_COUNT")

    # Profile is decided by total VRAM across all GPUs, since layer-split
    # tensor parallelism pools memory (e.g. 2x 96GB = 192GB -> HIGH).
    # Thresholds: <=12GB LOW, 13-23GB MEDIUM, >=24GB HIGH.
    if [ "$GPU_TOTAL_MEMORY_GB" -le 12 ]; then
        HARDWARE_PROFILE="LOW"
    elif [ "$GPU_TOTAL_MEMORY_GB" -lt 24 ]; then
        HARDWARE_PROFILE="MEDIUM"
    else
        HARDWARE_PROFILE="HIGH"
    fi
}

# --- Utility Functions -------------------------------------------------------

# Get current container status
get_container_status() {
    if docker ps --format '{{.Names}}' 2>/dev/null | grep -q "^${CONTAINER_NAME}$"; then
        echo "🟢 RUNNING"
    elif docker ps -a --format '{{.Names}}' 2>/dev/null | grep -q "^${CONTAINER_NAME}$"; then
        echo "🔴 STOPPED"
    else
        echo "⚫ NOT RUNNING"
    fi
}

# Get liteLLM container status
get_litellm_status() {
    if docker ps --format '{{.Names}}' 2>/dev/null | grep -q "^${LITELLM_CONTAINER}$"; then
        echo "🟢 RUNNING"
    elif docker ps -a --format '{{.Names}}' 2>/dev/null | grep -q "^${LITELLM_CONTAINER}$"; then
        echo "🔴 STOPPED"
    else
        echo "⚫ NOT RUNNING"
    fi
}

# Get file size of model or "N/A"
get_file_size() {
    local file="$MODELS_DIR/$1"
    if [ -f "$file" ]; then
        du -sh "$file" 2>/dev/null | cut -f1
    else
        echo "N/A"
    fi
}

# Check if model file exists
file_exists() {
    [ -f "$MODELS_DIR/$1" ]
}

# Check prerequisites (docker availability)
check_prerequisites() {
    if ! command -v docker &>/dev/null; then
        whiptail --title "❌ Error" \
            --msgbox "Docker command not found in PATH!" \
            8 50
        exit 1
    fi

    if ! docker info &>/dev/null 2>&1; then
        whiptail --title "❌ Error" \
            --msgbox "Docker daemon is not running.\n\nPlease start Docker and try again." \
            10 55
        exit 1
    fi
}

# Check if whiptail is available
check_whiptail() {
    if ! command -v whiptail &>/dev/null; then
        echo "Error: whiptail not found. Please install whiptail: sudo apt install whiptail"
        exit 1
    fi
}

# --- liteLLM Functions -------------------------------------------------------

# Generate liteLLM configuration file
generate_litellm_config() {
    cat > "$LITELLM_CONFIG" <<'YAML'
model_list:
  - model_name: "gpt-4o"
    litellm_params:
      model: "openai/local"
      api_base: "http://host.docker.internal:8080"
      api_key: "sk-local"

litellm_settings:
  drop_params: true
  cache: true
  cache_params:
    type: "local"
    supported_call_types: ["completion", "text_completion"]

general_settings:
  master_key: "sk-local-master"
YAML
}

# Stop liteLLM container if running
stop_litellm_container() {
    if docker ps --format '{{.Names}}' 2>/dev/null | grep -q "^${LITELLM_CONTAINER}$"; then
        docker stop "$LITELLM_CONTAINER" 2>/dev/null || true
        docker rm "$LITELLM_CONTAINER" 2>/dev/null || true
    fi
}

# Launch liteLLM proxy container
launch_litellm() {
    docker run -d --name "$LITELLM_CONTAINER" \
        --add-host host.docker.internal:host-gateway \
        -p ${LITELLM_PORT}:4000 \
        -v "${LITELLM_CONFIG}:/app/config.yaml" \
        -e "DATABASE_URL=postgresql://user:password@host.docker.internal:5433/litellm" \
        "$LITELLM_IMAGE" \
        --config /app/config.yaml \
        --port 4000 \
        2>/dev/null || {
        whiptail --title "⚠️  liteLLM Warning" \
            --msgbox "Failed to launch liteLLM container.\n\nThis is optional. llama.cpp will still run on port $PORT.\n\nCheck: Docker is running, port $LITELLM_PORT is free" \
            12 70
        return 1
    }
}

# Ask user if they want liteLLM proxy
ask_litellm_option() {
    whiptail --title "Optional: liteLLM Proxy" \
        --yesno "Start liteLLM proxy alongside llama.cpp?\n\nBenefits:\n• Unified client endpoint (always :$LITELLM_PORT)\n• Semantic caching for faster responses\n• Web UI dashboard at localhost:$LITELLM_PORT/ui\n• Request logging and monitoring\n\nNote: You can always run llama.cpp alone on port $PORT" \
        18 75 \
        --yes-button "Start liteLLM" --no-button "Skip"
}

# --- UI Functions (whiptail dialogs) -----------------------------------------

# Show model selection menu with file status and sizes
show_model_menu() {
    local container_status litellm_status status_line hardware_info
    container_status=$(get_container_status)
    litellm_status=$(get_litellm_status)
    status_line="llama.cpp: $container_status  │  liteLLM: $litellm_status"

    # Hardware profile display
    case "$HARDWARE_PROFILE" in
        LOW)  hardware_info="🔴 LOW (≤ 12GB VRAM)" ;;
        MEDIUM) hardware_info="🟡 MEDIUM (13-23GB VRAM)" ;;
        HIGH) hardware_info="🟢 HIGH (≥ 24GB VRAM)" ;;
    esac

    # Build menu items array
    local menu_items=()
    for i in "${!MODEL_NAMES[@]}"; do
        local id=$((i + 1))
        local name="${MODEL_NAMES[$i]}"
        local desc="${MODEL_DESCS[$i]}"
        local file="${MODEL_FILES[$i]}"
        local size
        size=$(get_file_size "$file")
        local exists_mark

        if file_exists "$file"; then
            exists_mark="✓"
        else
            exists_mark="✗"
        fi

        # Build item with proper spacing
        local item_text="[$exists_mark] $name  ($size) │ $desc"
        menu_items+=("$id" "$item_text")
    done

    # Show whiptail menu
    local choice
    choice=$(whiptail \
        --title "🦙 LLaMA.cpp Model Launcher (Auto-Optimized)" \
        --menu "Status: $status_line │ Hardware: $hardware_info | GPU: ${GPU_COUNT}x ${GPU_MEMORY_GB}GB = ${GPU_TOTAL_MEMORY_GB}GB total (devices: ${CUDA_DEVICES:-CPU-only})\n\nSelect a model to launch:\n[✓] File exists  [✗] File missing" \
        32 105 8 \
        "${menu_items[@]}" \
        3>&1 1>&2 2>&3) || return 1

    echo "$choice"
}

# Show launch confirmation dialog
confirm_launch() {
    local choice="$1"
    local idx=$((choice - 1))
    local name="${MODEL_NAMES[$idx]}"
    local file="${MODEL_FILES[$idx]}"
    local desc="${MODEL_DESCS[$idx]}"
    local size
    size=$(get_file_size "$file")

    # Check if file exists
    if ! file_exists "$file"; then
        whiptail --title "⚠️  File Not Found" \
            --msgbox "Model file not found:\n  $file\n\nPlease check that the file exists in:\n  $MODELS_DIR\n\nOr download the model and place it there." \
            13 70
        return 1
    fi

    # Show confirmation dialog
    whiptail --title "Confirm Launch" \
        --yesno "Launch the following model?\n\nModel:  $name\nFile:   $file\nSize:   $size\nSpecs:  $desc\nPort:   http://localhost:$PORT\n\nNote: Any running container will be stopped first." \
        16 75
}

# Show launch success and ask about log tailing
show_launch_success() {
    local choice="$1"
    local idx=$((choice - 1))
    local name="${MODEL_NAMES[$idx]}"
    local litellm_status litellm_msg
    litellm_status=$(get_litellm_status)

    if [[ "$litellm_status" == "🟢 RUNNING" ]]; then
        litellm_msg="\nliteLLM Proxy: http://localhost:$LITELLM_PORT (clients use this)\nliteLLM UI:   http://localhost:$LITELLM_PORT/ui"
    else
        litellm_msg=""
    fi

    whiptail --title "✅ Success" \
        --yesno "Containers launched successfully!\n\nModel:        $name\nllama.cpp:    http://localhost:$PORT${litellm_msg}\n\nWould you like to tail the container logs?\n(Press Ctrl+C to stop)" \
        16 75 \
        --yes-button "Tail Logs" --no-button "Exit"
}

# --- Container Management Functions ------------------------------------------

# Stop and remove existing containers
stop_existing_container() {
    local status litellm_status
    litellm_status=$(get_litellm_status)
    status=$(get_container_status)

    # Stop liteLLM first
    if [[ "$litellm_status" != "⚫ NOT RUNNING" ]]; then
        docker stop "$LITELLM_CONTAINER" 2>/dev/null || true
        docker rm "$LITELLM_CONTAINER" 2>/dev/null || true
    fi

    # Stop llama.cpp
    if [[ "$status" != "⚫ NOT RUNNING" ]]; then
        docker stop "$CONTAINER_NAME" 2>/dev/null || true
        docker rm "$CONTAINER_NAME" 2>/dev/null || true
    fi
}

# Launch the selected model with hardware-aware optimization
launch_model() {
    local choice="$1"
    local idx=$((choice - 1))
    local model_file="${MODEL_FILES[$idx]}"
    local extra_args
    extra_args=$(get_model_args "$idx")

    # Adjust resource allocation based on hardware profile
    local shm_size memory_limit threads
    case "$HARDWARE_PROFILE" in
        LOW)
            shm_size="8g"
            memory_limit="10g"
            threads="4"
            ;;
        MEDIUM)
            shm_size="16g"
            memory_limit="20g"
            threads="8"
            ;;
        HIGH)
            shm_size="100g"
            memory_limit="124g"
            threads="32"
            ;;
    esac

    # Build GPU-related docker flags based on the actual GPUs detected.
    # - 0 GPUs  : CPU-only, no --gpus / CUDA_VISIBLE_DEVICES
    # - 1 GPU   : CUDA_VISIBLE_DEVICES=0
    # - N GPUs  : CUDA_VISIBLE_DEVICES=0,1,...,N-1 (llama.cpp splits layers across them)
    local gpu_args=()
    if [ "$GPU_COUNT" -ge 1 ]; then
        gpu_args+=(--gpus all)
        gpu_args+=(-e "CUDA_VISIBLE_DEVICES=${CUDA_DEVICES}")
        gpu_args+=(-e "NVIDIA_TF32=1")
        gpu_args+=(-e "NVIDIA_DISABLE_MPS=0")
    fi

    # Enable layer-split tensor parallelism only when more than one GPU is present.
    local split_args=()
    if [ "$GPU_COUNT" -gt 1 ]; then
        split_args+=(--split-mode layer)
    fi

    docker run -d --name "$CONTAINER_NAME" \
        ${gpu_args[@]+"${gpu_args[@]}"} \
        --cap-add IPC_LOCK \
        --cap-add SYS_ADMIN \
        --ulimit memlock=-1:-1 \
        --ulimit stack=67108864 \
        --shm-size "$shm_size" \
        --memory "$memory_limit" \
        -p ${PORT}:8080 \
        -v "${MODELS_DIR}:/models" \
        "$DOCKER_IMAGE" \
        -m "/models/${model_file}" \
        --host 0.0.0.0 \
        --threads "$threads" \
        --parallel 1 \
        ${split_args[@]+"${split_args[@]}"} \
        $extra_args \
        2>/dev/null || {
        whiptail --title "❌ Docker Error" \
            --msgbox "Failed to launch docker container.\n\nPlease check:\n- Docker is running\n- No other container uses port $PORT\n- NVIDIA Container Toolkit is properly installed\n- GPU (CUDA) support is enabled\n- Detected GPUs: $GPU_COUNT (devices: ${CUDA_DEVICES:-CPU-only})\n- Sufficient VRAM for selected model (${GPU_MEMORY_GB}GB per GPU, ${GPU_TOTAL_MEMORY_GB}GB total)" \
            14 75
        return 1
    }
}

# --- Main Flow ---------------------------------------------------------------

main() {
    check_whiptail
    check_prerequisites

    # Detect hardware profile first
    detect_hardware_profile

    # Loop to handle file-not-found case
    while true; do
        local choice use_litellm
        choice=$(show_model_menu) || exit 0  # User pressed Cancel

        if confirm_launch "$choice"; then
            # Ask about liteLLM
            if ask_litellm_option; then
                use_litellm=true
            else
                use_litellm=false
            fi

            # Pre-flight check passed
            stop_existing_container

            # Generate liteLLM config if needed
            if [[ "$use_litellm" == "true" ]]; then
                generate_litellm_config
            fi

            # Launch llama.cpp
            launch_model "$choice" || exit 1

            # Launch liteLLM if requested
            if [[ "$use_litellm" == "true" ]]; then
                sleep 2  # Give llama.cpp time to start
                launch_litellm || true  # Non-fatal if liteLLM fails
            fi

            # Show success and ask about logs
            if show_launch_success "$choice"; then
                # Tail logs with visual separator
                echo ""
                echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
                echo "Container logs (Ctrl+C to stop tailing):"
                echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
                echo ""
                docker logs -f "$CONTAINER_NAME" 2>/dev/null || true
            fi

            exit 0
        else
            # User selected No in confirmation, show menu again
            continue
        fi
    done
}

# --- Entry Point -------------------------------------------------------------
main
