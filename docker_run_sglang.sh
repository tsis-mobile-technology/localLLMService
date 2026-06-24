#!/bin/bash
set -euo pipefail

# ============================================================================
# SGLang Interactive Docker Model Launcher
# ============================================================================
# Provides a whiptail TUI for selecting and launching different LLM models
# using SGLang with Docker. Supports model cache validation, container
# status checking, and automatic log tailing.
# ============================================================================

# --- Constants ---------------------------------------------------------------
readonly HF_CACHE_DIR="$HOME/.cache/huggingface"
readonly CONTAINER_NAME="sglang-server"
readonly DOCKER_IMAGE="lmsysorg/sglang:latest"
readonly PORT=30000

# liteLLM Integration (optional proxy layer for caching/logging/UI)
readonly LITELLM_CONTAINER="litellm-proxy"
readonly LITELLM_PORT=4000
readonly LITELLM_IMAGE="ghcr.io/berriai/litellm:main-latest"
readonly LITELLM_CONFIG="$HOME/Programming/models/litellm_config_sglang.yaml"

# --- Model Definitions (parallel arrays) -----------------------------------
# Optimized for NVIDIA GB10 Grace Blackwell: 128GB LPDDRX, 2x Superchips, ConnectX7
readonly MODEL_NAMES=(
    "GLM-5.2 (Recommended for GB10)"
    "GLM-5.2-Multi-Vision"
    "Gemma 4 E4B IT (FP8 - 100% GPU)"
    "Gemma 4 E4B IT (BF16 Full Precision)"
    "Qwen 3.6 35B A3B Instruct"
    "Gemma 2 9B IT"
    "Qwen 2.5 7B Instruct"
)

readonly MODEL_IDS=(
    "zai-org/GLM-5.2"
    "zai-org/GLM-5.2-Multi-Vision"
    "vrfai/gemma-4-E4B-it-fp8"
    "google/gemma-4-E4B-it"
    "Qwen/Qwen3.6-35B-A3B-Instruct"
    "google/gemma-2-9b-it"
    "Qwen/Qwen2.5-7B-Instruct"
)

readonly MODEL_DESCS=(
    "5.2B Dense │ Fast Inference │ 128K context │ Rec. GB10"
    "5.2B Vision │ Multimodal     │ 128K context │ Vision Capable"
    "4.5B/8B │ Text/Image/Audio │ FP8 100% GPU │ Gemma 4"
    "4.5B/8B │ Text/Image/Audio │ BF16 Full    │ Gemma 4 Premium"
    "35B/3B  │ Sparse MoE       │ 256K context │ Qwen Advanced"
    "9B      │ Dense            │ 8K context   │ Gemma 2 Light"
    "7B      │ Dense            │ 131K context │ Qwen Compact"
)

# Grace Blackwell optimization: Maximize tensor parallelism, high mem fraction
# 128GB allows aggressive offloading and large batch processing
readonly MODEL_EXTRA_ARGS=(
    "--tp 1 --dtype bfloat16 --mem-fraction-static 0.90 --max-total-tokens 131072"
    "--tp 1 --dtype bfloat16 --mem-fraction-static 0.90 --max-total-tokens 131072"
    "--tp 1 --quantization fp8 --mem-fraction-static 0.95 --max-total-tokens 131072"
    "--tp 1 --dtype bfloat16 --mem-fraction-static 0.90 --max-total-tokens 131072"
    "--tp 2 --dtype bfloat16 --mem-fraction-static 0.85 --cpu-offload-gb 20 --max-total-tokens 256000"
    "--tp 1 --dtype bfloat16 --mem-fraction-static 0.90 --max-total-tokens 131072"
    "--tp 1 --dtype bfloat16 --mem-fraction-static 0.95 --max-total-tokens 131072"
)

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

# Check if model exists in Hugging Face Cache
model_cache_exists() {
    local model_id="$1"
    local cache_folder_name="models--$(echo "$model_id" | sed 's/\//--/g')"
    [ -d "$HF_CACHE_DIR/hub/$cache_folder_name" ]
}

# Get cache size of model or "N/A"
get_cache_size() {
    local model_id="$1"
    local cache_folder_name="models--$(echo "$model_id" | sed 's/\//--/g')"
    local path="$HF_CACHE_DIR/hub/$cache_folder_name"
    if [ -d "$path" ]; then
        du -sh "$path" 2>/dev/null | cut -f1
    else
        echo "N/A"
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
    local model_id="$1"
    cat > "$LITELLM_CONFIG" <<YAML
model_list:
  - model_name: "gpt-4o"
    litellm_params:
      model: "openai/${model_id}"
      api_base: "http://host.docker.internal:${PORT}/v1"
      api_key: "sk-local"
  - model_name: "gemma-4"
    litellm_params:
      model: "openai/${model_id}"
      api_base: "http://host.docker.internal:${PORT}/v1"
      api_key: "sk-local"
  - model_name: "gemma-4-bf16"
    litellm_params:
      model: "openai/google/gemma-4-E4B-it"
      api_base: "http://host.docker.internal:${PORT}/v1"
      api_key: "sk-local"
  - model_name: "gemma-4-fp8"
    litellm_params:
      model: "openai/vrfai/gemma-4-E4B-it-fp8"
      api_base: "http://host.docker.internal:${PORT}/v1"
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
            --msgbox "Failed to launch liteLLM container.\n\nThis is optional. SGLang will still run on port $PORT.\n\nCheck: Docker is running, port $LITELLM_PORT is free" \
            12 70
        return 1
    }
}

# Ask user if they want liteLLM proxy
ask_litellm_option() {
    whiptail --title "Optional: liteLLM Proxy" \
        --yesno "Start liteLLM proxy alongside SGLang?\n\nBenefits:\n• Unified client endpoint (always :$LITELLM_PORT)\n• Semantic caching for faster responses\n• Web UI dashboard at localhost:$LITELLM_PORT/ui\n• Request logging and monitoring\n\nNote: You can always run SGLang alone on port $PORT" \
        18 75 \
        --yes-button "Start liteLLM" --no-button "Skip"
}

# --- UI Functions (whiptail dialogs) -----------------------------------------

# Show model selection menu with file status and sizes
show_model_menu() {
    local container_status litellm_status status_line
    container_status=$(get_container_status)
    litellm_status=$(get_litellm_status)
    status_line="SGLang: $container_status  │  liteLLM: $litellm_status"

    # Build menu items array
    local menu_items=()
    for i in "${!MODEL_NAMES[@]}"; do
        local id=$((i + 1))
        local name="${MODEL_NAMES[$i]}"
        local desc="${MODEL_DESCS[$i]}"
        local model_id="${MODEL_IDS[$i]}"
        local size
        size=$(get_cache_size "$model_id")
        local exists_mark

        if model_cache_exists "$model_id"; then
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
        --title "🚀 SGLang Model Launcher (Grace Blackwell Optimized)" \
        --menu "Status: $status_line\n\nSelect a model to launch:\n[✓] Cache exists  [✗] Missing (Will download)" \
        30 100 7 \
        "${menu_items[@]}" \
        3>&1 1>&2 2>&3) || return 1

    echo "$choice"
}

# Show launch confirmation dialog
confirm_launch() {
    local choice="$1"
    local idx=$((choice - 1))
    local name="${MODEL_NAMES[$idx]}"
    local model_id="${MODEL_IDS[$idx]}"
    local desc="${MODEL_DESCS[$idx]}"
    local size
    size=$(get_cache_size "$model_id")
    local download_msg=""

    # Alert if model will be downloaded
    if ! model_cache_exists "$model_id"; then
        download_msg="\n\n⚠️  WARNING: This model is NOT in cache. SGLang will download it from Hugging Face Hub. This might take a while depending on your network connection."
    fi

    # Show confirmation dialog
    whiptail --title "Confirm Launch" \
        --yesno "Launch the following model using SGLang?\n\nModel:  $name\nID:     $model_id\nSize:   $size\nSpecs:  $desc\nPort:   http://localhost:$PORT${download_msg}\n\nNote: Any running container will be stopped first." \
        18 80
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
        --yesno "Containers launched successfully!\n\nModel:   $name\nSGLang:  http://localhost:$PORT${litellm_msg}\n\nWould you like to tail the container logs?\n(Press Ctrl+C to stop)" \
        16 75 \
        --yes-button "Tail Logs" --no-button "Exit"
}

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

    # Stop SGLang
    if [[ "$status" != "⚫ NOT RUNNING" ]]; then
        docker stop "$CONTAINER_NAME" 2>/dev/null || true
        docker rm "$CONTAINER_NAME" 2>/dev/null || true
    fi
}

# Launch the selected model with Grace Blackwell optimization
launch_model() {
    local choice="$1"
    local idx=$((choice - 1))
    local model_id="${MODEL_IDS[$idx]}"
    local extra_args="${MODEL_EXTRA_ARGS[$idx]}"

    # Grace Blackwell specific optimizations: 128GB LPDDRX, 2x Superchip, ConnectX7
    docker run -d --name "$CONTAINER_NAME" \
        --gpus all \
        --ipc=host \
        --shm-size 120g \
        --memory 124g \
        --ulimit memlock=-1:-1 \
        --ulimit stack=67108864 \
        --cap-add IPC_LOCK \
        --cap-add SYS_ADMIN \
        -p ${PORT}:30000 \
        -v "${HF_CACHE_DIR}:/root/.cache/huggingface" \
        -v "$(pwd)/clippable_linear.py:/sgl-workspace/sglang/python/sglang/srt/layers/clippable_linear.py" \
        -v "$(pwd)/weight_utils.py:/sgl-workspace/sglang/python/sglang/srt/model_loader/weight_utils.py" \
        -e CUDA_VISIBLE_DEVICES=0,1 \
        -e NVIDIA_TF32=1 \
        -e NVIDIA_DISABLE_MPS=0 \
        -e NCCL_LAUNCH_MODE=PARALLEL \
        "$DOCKER_IMAGE" \
        python3 -m sglang.launch_server \
        --model-path "$model_id" \
        --host 0.0.0.0 \
        --port 30000 \
        --trust-remote-code \
        $extra_args \
        2>/dev/null || {
        whiptail --title "❌ Docker Error" \
            --msgbox "Failed to launch docker container.\n\nPlease check:\n- Docker is running\n- No other container uses port $PORT\n- NVIDIA Container Toolkit is properly installed\n- GPU (CUDA) support is enabled on Grace Blackwell\n- Sufficient disk space in HF cache (~30GB for large models)" \
            14 75
        return 1
    }
}

# Wait for SGLang server to become healthy
wait_for_sglang() {
    local max_wait=300 # 5 minutes max wait
    local wait_interval=2
    local elapsed=0
    
    echo -n "⏳ Waiting for SGLang server to initialize (Loading model might take some time)..."
    
    while [ $elapsed -lt $max_wait ]; do
        # Check if container is still running
        if ! docker ps --format '{{.Names}}' 2>/dev/null | grep -q "^${CONTAINER_NAME}$"; then
            echo -e "\n❌ SGLang container stopped unexpectedly!"
            return 1
        fi
        
        # Check health API
        local status_code
        status_code=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:${PORT}/health || echo "000")
        
        if [ "$status_code" -eq 200 ]; then
            echo -e "\n🟢 SGLang server is ready!"
            return 0
        fi
        
        sleep $wait_interval
        elapsed=$((elapsed + wait_interval))
        echo -n "."
    done
    
    echo -e "\n❌ Timeout waiting for SGLang server to start."
    return 1
}

# --- Main Flow ---------------------------------------------------------------

main() {
    check_prerequisites

    # Ensure HF Cache dir exists
    mkdir -p "$HF_CACHE_DIR"

    local choice=""
    local use_litellm=false

    # If argument is provided, skip whiptail menu (non-interactive mode)
    if [ $# -gt 0 ]; then
        choice="$1"
        if ! [[ "$choice" =~ ^[1-6]$ ]]; then
            echo "Error: Invalid model choice '$choice'. Must be 1-6."
            exit 1
        fi
        
        # Pre-flight check passed
        stop_existing_container

        # Generate liteLLM config
        local idx=$((choice - 1))
        local model_id="${MODEL_IDS[$idx]}"
        generate_litellm_config "$model_id"

        # Launch SGLang
        launch_model "$choice" || exit 1

        # Wait for SGLang to be ready
        if ! wait_for_sglang; then
            echo "Showing SGLang server logs:"
            docker logs "$CONTAINER_NAME" | tail -n 50
            exit 1
        fi

        # Launch liteLLM
        launch_litellm || true

        echo "🚀 SGLang ($choice) and liteLLM Proxy started successfully!"
        exit 0
    fi

    # Otherwise, run interactive mode with whiptail
    check_whilltail_available=true
    if ! command -v whiptail &>/dev/null; then
        echo "Error: whiptail not found. Please install whiptail: sudo apt install whiptail"
        exit 1
    fi

    while true; do
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
                local idx=$((choice - 1))
                local model_id="${MODEL_IDS[$idx]}"
                generate_litellm_config "$model_id"
            fi

            # Launch SGLang
            launch_model "$choice" || exit 1

            # Wait for SGLang to be ready
            if ! wait_for_sglang; then
                whiptail --title "❌ SGLang Error" \
                    --msgbox "SGLang failed to start or load the model.\n\nShowing container logs..." \
                    10 60
                echo "Showing SGLang server logs:"
                docker logs "$CONTAINER_NAME" | tail -n 50
                exit 1
            fi

            # Launch liteLLM if requested
            if [[ "$use_litellm" == "true" ]]; then
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
main "$@"
