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
# Optimized for NVIDIA GB10 Grace Blackwell: 2x 128GB LPDDRX, 2x Superchips, ConnectX7 (256GB cluster)
# GLM-5.2 (744B, 40B active params): https://huggingface.co/unsloth/GLM-5.2-GGUF
# Quantization: IQ1_S (1-bit, 223GB) for 256GB cluster | IQ2_M (2-bit, 245GB) for dual-node | IQ3_XXS (3-bit, 110GB) for single 128GB
readonly MODEL_NAMES=(
    "GLM-5.2 IQ1_S (Recommended - 256GB Cluster)"
    "GLM-5.2 IQ2_M (128GB Dual-Node - Limited)"
    "GLM-5.2 IQ3_XXS (Single 128GB - Compact)"
    "Gemma 4 31B (Multi-Instance - Team Collab)"
    "Gemma 4 E4B Q4"
    "Gemma 4 E4B Q8"
    "Gemma 4 26B A4B"
    "Qwen 3.6 35B A3B"
    "DeepSeek-Coder-V2 (164K Context)"
)

readonly MODEL_FILES=(
    "unsloth/GLM-5.2-GGUF:UD-IQ1_S"
    "unsloth/GLM-5.2-GGUF:UD-IQ2_M"
    "unsloth/GLM-5.2-GGUF:UD-IQ3_XXS"
    "unsloth/gemma-4-31b-it-GGUF:UD-Q4_K_XL"
    "google_gemma-4-E4B-it-Q4_K_M.gguf"
    "google_gemma-4-E4B-it-Q8_0.gguf"
    "google_gemma-4-26B-A4B-it-Q4_K_M.gguf"
    "Qwen_Qwen3.6-35B-A3B-Q4_0.gguf"
    "bullerwins/DeepSeek-Coder-V2-Instruct-GGUF:DeepSeek-Coder-V2-Instruct-Q4_K_S"
)

readonly MODEL_DESCS=(
    "IQ1_S (1-bit) │ 256GB cluster │ 223GB mem │ 744B (4 files) - Ultra-Compressed"
    "IQ2_M (2-bit) │ 256GB cluster │ 245GB mem │ 744B (5 files) - High-Quality Compact"
    "IQ3_XXS (3-bit) │ 128GB single │ 110GB mem │ 744B (7 files) - Balanced Compact"
    "Q4_K_XL │ 45K ctx │ Data Parallelism │ 31B Coding (4x instances + LiteLLM) │ Best for Team"
    "Q4_K_M  │ 131K ctx │ Full GPU    │ 4B"
    "Q8_0  │ 131K ctx │ Full GPU      │ 4B Premium"
    "Q4_K_M │ 256K ctx │ Full GPU      │ 26B MoE Hybrid"
    "Q4_0  │ 256K ctx │ MoE GPU       │ 35B MoE Advanced"
    "Q4_K_S │ 164K ctx │ MoE Experts (6/6) │ 236B (21B active) │ Long-Doc Coding"
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
                8)  echo "# ERROR: DeepSeek-Coder-V2 (60-65GB) requires HIGH or MEDIUM+ profile. Use single-node 128GB with IQ3_XXS model instead." ;;
            esac
            ;;
        MEDIUM)  # 12-24GB VRAM: GLM-5.2 & DeepSeek-V2 not supported in MEDIUM
            case $model_idx in
                0)  echo "# ERROR: GLM-5.2 IQ1_S requires 256GB cluster. Use IQ3_XXS or switch to HIGH profile." ;;
                1)  echo "# ERROR: GLM-5.2 IQ2_M requires 245GB dual-node. Use IQ3_XXS or switch to HIGH profile." ;;
                2)  echo "--no-mmap --cache-type-k q4_0 --cache-type-v q4_0 --mlock -c 65536 -n 1024 --temp 1.0 --top-p 0.95 --min-p 0.01" ;;
                3)  echo "--no-mmap --cache-type-k q4_0 --cache-type-v q4_0 --mlock -c 65536 -n 4096" ;;
                4)  echo "--no-mmap --cache-type-k q8_0 --cache-type-v q8_0 --mlock -c 65536 -n 4096" ;;
                5)  echo "--n-gpu-layers 32 -c 128000 --cache-type-k q4_0 --cache-type-v q4_0 -n 8192" ;;
                6)  echo "--n-gpu-layers 32 -c 128000 --cache-type-k q4_0 --cache-type-v q4_0 -n 8192" ;;
                7)  echo "--n-gpu-layers 32 -c 128000 --cache-type-k q4_0 --cache-type-v q4_0 -n 8192" ;;
                8)  echo "# ERROR: DeepSeek-Coder-V2 (60-65GB) requires HIGH profile. Upgrade to 256GB+ cluster." ;;
            esac
            ;;
        HIGH)  # > 24GB VRAM (Grace Blackwell): 256GB cluster optimized for GLM-5.2, Gemma 4 31B, & DeepSeek-V2
            case $model_idx in
                0)  echo "-c 32768 -n 4096 --cache-type-k q4_0 --cache-type-v q4_0 --mlock --temp 1.0 --top-p 0.95 --min-p 0.01" ;;
                1)  echo "-c 16384 -n 2048 --cache-type-k q4_0 --cache-type-v q4_0 --mlock --temp 1.0 --top-p 0.95 --min-p 0.01" ;;
                2)  echo "-c 65536 -n 4096 --cache-type-k q4_0 --cache-type-v q4_0 --mlock --temp 1.0 --top-p 0.95 --min-p 0.01" ;;
                3)  echo "-c 45000 -n 4096 --cb --threads 16 --parallel 4 --temp 0.7 --top-p 0.95 --min-p 0.05" ;;
                4)  echo "--no-mmap --cache-type-k q4_0 --cache-type-v q4_0 --mlock -c 131072 -n 8192" ;;
                5)  echo "--no-mmap --cache-type-k q8_0 --cache-type-v q8_0 --mlock -c 131072 -n 8192" ;;
                6)  echo "--n-gpu-layers 64 -c 256000 --cache-type-k q4_0 --cache-type-v q4_0 -n 16384" ;;
                7)  echo "--n-gpu-layers 64 -c 256000 --cache-type-k q4_0 --cache-type-v q4_0 -n 16384" ;;
                8)  echo "-c 82000 -n 4096 --cb --threads 16 --parallel 4 --temp 0.7 --top-p 0.95 --min-p 0.05" ;;
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

# Get file size of model or "N/A" or HuggingFace model size
get_file_size() {
    local file="$1"
    if [[ "$file" == *"/"* ]] && [[ "$file" != "/"* ]]; then
        # HuggingFace hub model - return approximate size based on quantization
        case "$file" in
            *"UD-IQ1_S"*) echo "~223GB (HF)" ;;
            *"UD-IQ2_M"*) echo "~245GB (HF)" ;;
            *"UD-IQ3_XXS"*) echo "~110GB (HF)" ;;
            *"gemma-4-31b"*) echo "~20GB/inst (HF)" ;;
            *"DeepSeek-Coder-V2"*) echo "~60-65GB (HF)" ;;
            *) echo "TBD" ;;
        esac
    else
        # Local file
        local full_path="$MODELS_DIR/$file"
        if [ -f "$full_path" ]; then
            du -sh "$full_path" 2>/dev/null | cut -f1
        else
            echo "N/A"
        fi
    fi
}

# Check if model file exists (or is available on HuggingFace hub)
file_exists() {
    local file="$1"
    if [[ "$file" == *"/"* ]] && [[ "$file" != "/"* ]]; then
        # HuggingFace hub model - assume available
        return 0
    else
        # Local file
        [ -f "$MODELS_DIR/$file" ]
    fi
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

    # Special dialog for Gemma 4 31B with multi-instance option
    if [ "$idx" -eq 3 ]; then
        whiptail --title "Gemma 4 31B - Mode Selection" \
            --yesno "Model:  $name\nFile:   $file\nSize:   $size\nSpecs:  $desc\n\nThis model supports DATA PARALLELISM mode.\n\nWould you like to:\n[Yes]  Multi-Instance (4x + LiteLLM) - Best for team\n[No]   Single-Instance - Simple, single user" \
            18 75
        return $?
    fi

    # Standard confirmation dialog for other models
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

# --- Multi-Instance Mode (for Gemma 4 31B with LiteLLM) ----------------------

# Launch multiple instances of the same model (Data Parallelism)
launch_multi_instance() {
    local choice="$1"
    local idx=$((choice - 1))
    local model_file="${MODEL_FILES[$idx]}"
    local extra_args
    extra_args=$(get_model_args "$idx")

    # Only Gemma 4 31B supports multi-instance mode
    if [ "$idx" -ne 3 ]; then
        whiptail --title "❌ Not Supported" \
            --msgbox "Multi-Instance mode is only available for:\n\nGemma 4 31B (Model #4)\n\nFor other models, use single-instance mode." \
            10 60
        return 1
    fi

    # Determine instance count based on hardware
    local instance_count=2  # Default: 2 instances per workstation
    if whiptail --title "⚙️ Instance Configuration" \
        --yesno "Multi-Instance Mode Detected!\n\nDefault: 2 instances per workstation\n(4 instances total with 2 workstations)\n\nCurrent setting: $instance_count instances\n\nContinue with default configuration?" \
        12 70; then
        :
    else
        return 1
    fi

    # Stop existing containers
    docker ps -a --format '{{.Names}}' 2>/dev/null | grep -E "gemma-31b-" | while read name; do
        docker stop "$name" 2>/dev/null || true
        docker rm "$name" 2>/dev/null || true
    done

    # Launch instances
    local start_port=8081
    for i in $(seq 0 $((instance_count - 1))); do
        local port=$((start_port + i))
        local instance_name="gemma-31b-instance-$i"

        local shm_size="32g"
        local memory_limit="40g"
        local threads="8"

        docker run -d --name "$instance_name" \
            --gpus all \
            --cap-add IPC_LOCK \
            --cap-add SYS_ADMIN \
            --ulimit memlock=-1:-1 \
            --ulimit stack=67108864 \
            --shm-size "$shm_size" \
            --memory "$memory_limit" \
            -p ${port}:8080 \
            -v "${MODELS_DIR}:/models" \
            -e "LLAMA_CACHE=/models" \
            "$DOCKER_IMAGE" \
            -hf "$model_file" \
            --host 0.0.0.0 \
            $extra_args \
            2>/dev/null || {
            whiptail --title "❌ Docker Error" \
                --msgbox "Failed to launch instance $i on port $port.\n\nPlease check:\n- Docker is running\n- Port $port is available\n- NVIDIA Container Toolkit installed\n- Sufficient VRAM" \
                12 70
            return 1
        }

        echo "✅ Launched $instance_name on port $port"
        sleep 1
    done

    whiptail --title "✅ Multi-Instance Started" \
        --msgbox "Successfully launched $instance_count instances:\n\n$(for i in $(seq 0 $((instance_count - 1))); do echo "  Instance $i: http://localhost:$((start_port + i))"; done)\n\nNext: Launch LiteLLM to aggregate these instances.\n\nPorts: $start_port-$((start_port + instance_count - 1))" \
        15 70
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

    # Determine if model is from HuggingFace hub or local file
    local model_arg
    if [[ "$model_file" == *"/"* ]] && [[ "$model_file" != "/"* ]]; then
        # HuggingFace hub format (e.g., unsloth/GLM-5.2-GGUF:UD-IQ1_S)
        model_arg="-hf $model_file"
    else
        # Local file format
        model_arg="-m /models/$model_file"
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
        -e "LLAMA_CACHE=/models" \
        "$DOCKER_IMAGE" \
        $model_arg \
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
            local idx=$((choice - 1))

            # Check if Gemma 4 31B multi-instance mode
            if [ "$idx" -eq 3 ]; then
                # Multi-Instance mode
                launch_multi_instance "$choice" || continue

                # Always ask about liteLLM for multi-instance
                if ask_litellm_option; then
                    use_litellm=true
                    generate_litellm_config
                    sleep 2
                    launch_litellm || true
                fi

                whiptail --title "✅ Setup Complete" \
                    --msgbox "Multi-Instance + LiteLLM setup complete!\n\nInstances running on:\n  - Instance 0: http://localhost:8081\n  - Instance 1: http://localhost:8082\n  $([ -n "$1" ] && echo "- Instance 2: http://ws2:8083\n  - Instance 3: http://ws2:8084")\n\nLiteLLM Proxy: http://localhost:4000\n\nUse this for team collaboration!" \
                    15 70

                exit 0
            else
                # Standard single-instance mode
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
            fi
        else
            # User selected No in confirmation, show menu again
            continue
        fi
    done
}

# --- Entry Point -------------------------------------------------------------
main
