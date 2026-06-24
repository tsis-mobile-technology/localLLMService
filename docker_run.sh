#!/bin/bash
set -euo pipefail

# ============================================================================
# LLaMA.cpp Interactive Docker Model Launcher
# ============================================================================
# Provides a whiptail TUI for selecting and launching different LLM models
# with docker-compose. Supports model file validation, container status
# checking, and automatic log tailing.
# ============================================================================

# --- Constants ---------------------------------------------------------------
readonly MODELS_DIR="$HOME/Programming/models"
readonly CONTAINER_NAME="llama-server"
readonly DOCKER_IMAGE="ghcr.io/ggml-org/llama.cpp:server-cuda"
readonly PORT=8080

# liteLLM Integration (optional proxy layer for caching/logging/UI)
readonly LITELLM_CONTAINER="litellm-proxy"
readonly LITELLM_PORT=4000
readonly LITELLM_IMAGE="ghcr.io/berriai/litellm:main-latest"
readonly LITELLM_CONFIG="$MODELS_DIR/litellm_config.yaml"
readonly LITELLM_MASTER_KEY="sk-local-master"

# --- Model Definitions (parallel arrays) -----------------------------------
# Optimized for NVIDIA GB10 Grace Blackwell: 128GB LPDDRX, 2x Superchips, ConnectX7
readonly MODEL_NAMES=(
    "GLM-5.2 BF16 (Recommended for GB10)"
    "GLM-5.2 FP8"
    "Gemma 4 E4B Q4"
    "Gemma 4 E4B Q8"
    "Gemma 4 31B"
    "Gemma 4 26B A4B"
    "Qwen 3.6 35B A3B"
)

readonly MODEL_FILES=(
    "GLM-5.2-it-BF16.gguf"
    "GLM-5.2-it-FP8.gguf"
    "google_gemma-4-E4B-it-Q4_K_M.gguf"
    "google_gemma-4-E4B-it-Q8_0.gguf"
    "google_gemma-4-31B-it-Q4_K_M.gguf"
    "google_gemma-4-26B-A4B-it-Q4_K_M.gguf"
    "Qwen_Qwen3.6-35B-A3B-Q4_0.gguf"
)

readonly MODEL_DESCS=(
    "BF16  │ 128K ctx │ Full Precision │ 5.2B Fast"
    "FP8   │ 128K ctx │ Quantized     │ 5.2B Faster"
    "Q4_K_M  │ 131K ctx │ Full GPU    │ 4B"
    "Q8_0  │ 131K ctx │ Full GPU      │ 4B Premium"
    "Q4_K_M │ 256K ctx │ Full GPU      │ 31B High Performance"
    "Q4_K_M │ 256K ctx │ MoE GPU       │ 26B MoE Hybrid"
    "Q4_0  │ 256K ctx │ MoE GPU       │ 35B MoE Advanced"
)

# Grace Blackwell optimization: Full GPU support, large context, high precision
# 128GB LPDDRX allows full model loading for larger models
readonly MODEL_EXTRA_ARGS=(
    "--no-mmap --cache-type-k f16 --cache-type-v f16 --mlock -c 131072 -n 8192"
    "--no-mmap --cache-type-k f16 --cache-type-v f16 --mlock -c 131072 -n 8192"
    "--no-mmap --cache-type-k q4_0 --cache-type-v q4_0 --mlock -c 131072 -n 8192"
    "--no-mmap --cache-type-k q8_0 --cache-type-v q8_0 --mlock -c 131072 -n 8192"
    "--n-gpu-layers 64 -c 256000 --cache-type-k q4_0 --cache-type-v q4_0 -n 16384"
    "--n-gpu-layers 64 -c 256000 --cache-type-k q4_0 --cache-type-v q4_0 -n 16384"
    "--n-gpu-layers 64 -c 256000 --cache-type-k q4_0 --cache-type-v q4_0 -n 16384"
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
    local container_status litellm_status status_line
    container_status=$(get_container_status)
    litellm_status=$(get_litellm_status)
    status_line="llama.cpp: $container_status  │  liteLLM: $litellm_status"

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
        --title "🦙 LLaMA.cpp Model Launcher (Grace Blackwell Optimized)" \
        --menu "Status: $status_line\n\nSelect a model to launch:\n[✓] File exists  [✗] File missing" \
        28 95 7 \
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

# Launch the selected model with Grace Blackwell optimization
launch_model() {
    local choice="$1"
    local idx=$((choice - 1))
    local model_file="${MODEL_FILES[$idx]}"
    local extra_args="${MODEL_EXTRA_ARGS[$idx]}"

    # Grace Blackwell specific optimizations
    # 128GB LPDDRX, Dual Superchip support, ConnectX7 fabric awareness
    docker run -d --name "$CONTAINER_NAME" \
        --gpus all \
        --cap-add IPC_LOCK \
        --cap-add SYS_ADMIN \
        --ulimit memlock=-1:-1 \
        --ulimit stack=67108864 \
        --shm-size 100g \
        --memory 124g \
        -p ${PORT}:8080 \
        -v "${MODELS_DIR}:/models" \
        -e CUDA_VISIBLE_DEVICES=0,1 \
        -e NVIDIA_TF32=1 \
        -e NVIDIA_DISABLE_MPS=0 \
        "$DOCKER_IMAGE" \
        -m "/models/${model_file}" \
        --threads-per-core 4 \
        --threads 32 \
        $extra_args \
        2>/dev/null || {
        whiptail --title "❌ Docker Error" \
            --msgbox "Failed to launch docker container.\n\nPlease check:\n- Docker is running\n- No other container uses port $PORT\n- NVIDIA Container Toolkit is properly installed\n- GPU (CUDA) support is enabled on Grace Blackwell" \
            14 75
        return 1
    }
}

# --- Main Flow ---------------------------------------------------------------

main() {
    check_whiptail
    check_prerequisites

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
