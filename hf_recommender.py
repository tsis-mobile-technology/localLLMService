#!/usr/bin/env python3
"""
hf_recommender.py
- Inspects PC hardware (GPU VRAM, CPU, RAM, Disk).
- Recommends the TOP 10 optimal Unsloth GGUF models from Hugging Face based on local hardware specs.
- Provides interactive download with real-time speed, ETA, progress display and resume support.
"""

import sys
import os
import json
import shutil
import subprocess
import urllib.request
import urllib.error
import time

try:
    from db_manager import DBManager
except ImportError:
    DBManager = None

MODELS_DIR_DEFAULT = os.path.expanduser("~/Projects/models")

# Curated High-Quality Unsloth GGUF Models with verified Hugging Face CDN links
#"args": "-ngl 99 -c 32768 --cache-type-k q4_0 --cache-type-v q4_0"
UNSLOTH_CATALOG = [
    {
        "id": "qwen3.5-9b",
        "repo": "unsloth/Qwen3.5-9B-GGUF",
        "name": "Qwen3.5 9B",
        "filename": "Qwen3.5-9B-Q4_0.gguf",
        "quant": "Q4_0",
        "size_gb": 5.01,
        "min_vram_gb": 7.0,
        "recommended_vram_gb": 10.0,
        "type": "dense",
        "active_params": "9B",
        "category": "고지능 범용",
        "desc": "Alibaba 최신 9B 초고성능 모델 / 32K context 지원 / 범용 최상급",
        "base_priority": 100,
        "args": "-ngl 99 -c 32768 --flash-attn on -ctk q8_0 -ctv q8_0 -b 2048 -ub 512"
    },
    {
        "id": "deepseek-r1-7b",
        "repo": "unsloth/DeepSeek-R1-Distill-Qwen-7B-GGUF",
        "name": "DeepSeek-R1-Distill 7B",
        "filename": "DeepSeek-R1-Distill-Qwen-7B-Q4_K_M.gguf",
        "quant": "Q4_K_M",
        "size_gb": 4.36,
        "min_vram_gb": 6.5,
        "recommended_vram_gb": 8.0,
        "type": "dense",
        "active_params": "7B",
        "category": "심층 추론",
        "desc": "DeepSeek-R1 수학·코딩 추론 증류 모델 / CoT Thinking 탁월",
        "base_priority": 95,
        "args": "-ngl 99 -c 16384 --cache-type-k q4_0 --cache-type-v q4_0"
    },
    {
        "id": "qwen2.5-coder-7b",
        "repo": "unsloth/Qwen2.5-Coder-7B-Instruct-GGUF",
        "name": "Qwen2.5-Coder 7B",
        "filename": "Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf",
        "quant": "Q4_K_M",
        "size_gb": 4.36,
        "min_vram_gb": 6.5,
        "recommended_vram_gb": 8.0,
        "type": "dense",
        "active_params": "7B",
        "category": "코딩 특화",
        "desc": "오픈소스 7B 코딩 최강자 / 코드 자동완성 및 디버깅 최적화",
        "base_priority": 93,
        "args": "-ngl 99 -c 32768 --cache-type-k q4_0 --cache-type-v q4_0"
    },
    {
        "id": "gemma-4-12b",
        "repo": "unsloth/gemma-4-12b-it-GGUF",
        "name": "Gemma 4 12B IT",
        "filename": "gemma-4-12b-it-Q4_0.gguf",
        "quant": "Q4_0",
        "size_gb": 6.28,
        "min_vram_gb": 8.5,
        "recommended_vram_gb": 11.0,
        "type": "dense",
        "active_params": "12B",
        "category": "최신 고성능",
        "desc": "Google 최신 Gemma 4 12B 모델 / 정밀 지시 추론",
        "base_priority": 90,
        "args": "-ngl 99 -c 16384 --cache-type-k q4_0 --cache-type-v q4_0"
    },
    {
        "id": "deepseek-r1-14b",
        "repo": "unsloth/DeepSeek-R1-Distill-Qwen-14B-GGUF",
        "name": "DeepSeek-R1-Distill 14B",
        "filename": "DeepSeek-R1-Distill-Qwen-14B-Q4_K_M.gguf",
        "quant": "Q4_K_M",
        "size_gb": 8.37,
        "min_vram_gb": 10.5,
        "recommended_vram_gb": 12.0,
        "type": "dense",
        "active_params": "14B",
        "category": "14B 심층추론",
        "desc": "14B 고지능 R1 추론 / 12GB VRAM 100% 풀로드 한계치 최적화",
        "base_priority": 88,
        "args": "-ngl 99 -c 8192 --cache-type-k q4_0 --cache-type-v q4_0"
    },
    {
        "id": "qwen3-coder-30b-moe",
        "repo": "unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF",
        "name": "Qwen3-Coder 30B MoE",
        "filename": "Qwen3-Coder-30B-A3B-Instruct-Q4_0.gguf",
        "quant": "Q4_0",
        "size_gb": 16.19,
        "min_vram_gb": 8.0,
        "recommended_vram_gb": 16.0,
        "type": "moe",
        "active_params": "3B",
        "category": "MoE 코딩 최강",
        "desc": "1200만 DL 1위! 활성 3B MoE / GPU+대용량 RAM 하이브리드",
        "base_priority": 85,
        "args": "-ngl 24 --n-cpu-moe 8 -c 16384 --cache-type-k q4_0 --cache-type-v q4_0"
    },
    {
        "id": "gemma-4-e4b",
        "repo": "unsloth/gemma-4-E4B-it-GGUF",
        "name": "Gemma 4 E4B IT",
        "filename": "gemma-4-E4B-it-Q4_0.gguf",
        "quant": "Q4_0",
        "size_gb": 4.50,
        "min_vram_gb": 5.5,
        "recommended_vram_gb": 6.5,
        "type": "dense",
        "active_params": "4B",
        "category": "경량 고효율",
        "desc": "Google Gemma 4 경량 4B 모델 / 낮은 VRAM, 빠른 응답 속도",
        "base_priority": 82,
        "args": "-ngl 99 -c 32768 --cache-type-k q4_0 --cache-type-v q4_0"
    },
    {
        "id": "qwen3.5-4b",
        "repo": "unsloth/Qwen3.5-4B-GGUF",
        "name": "Qwen3.5 4B",
        "filename": "Qwen3.5-4B-Q4_0.gguf",
        "quant": "Q4_0",
        "size_gb": 2.41,
        "min_vram_gb": 4.0,
        "recommended_vram_gb": 5.0,
        "type": "dense",
        "active_params": "4B",
        "category": "초고속 경량",
        "desc": "2.4GB 가벼운 크기로 초당 80+ 토큰 초고속 생성",
        "base_priority": 80,
        "args": "-ngl 99 -c 32768 --cache-type-k q4_0 --cache-type-v q4_0"
    },
    {
        "id": "phi-4-mini",
        "repo": "unsloth/Phi-4-mini-instruct-GGUF",
        "name": "Phi-4 Mini 3.8B",
        "filename": "Phi-4-mini-instruct-Q4_K_M.gguf",
        "quant": "Q4_K_M",
        "size_gb": 2.32,
        "min_vram_gb": 4.0,
        "recommended_vram_gb": 5.0,
        "type": "dense",
        "active_params": "3.8B",
        "category": "MS 고효율",
        "desc": "Microsoft 고효율 소형 모델 / 128K context 지원",
        "base_priority": 78,
        "args": "-ngl 99 -c 16384 --cache-type-k q4_0 --cache-type-v q4_0"
    },
    {
        "id": "llama-3.2-3b",
        "repo": "unsloth/Llama-3.2-3B-Instruct-GGUF",
        "name": "Llama 3.2 3B",
        "filename": "Llama-3.2-3B-Instruct-Q4_0.gguf",
        "quant": "Q4_0",
        "size_gb": 1.79,
        "min_vram_gb": 3.0,
        "recommended_vram_gb": 4.0,
        "type": "dense",
        "active_params": "3B",
        "category": "초경량 일상",
        "desc": "Meta 경량 3B 모델 / 일상 대화 및 가벼운 어시스턴트용",
        "base_priority": 75,
        "args": "-ngl 99 -c 16384 --cache-type-k q4_0 --cache-type-v q4_0"
    },
    {
        "id": "gemma-4-26b-a4b",
        "repo": "unsloth/gemma-4-26B-A4B-it-GGUF",
        "name": "Gemma 4 26B A4B MoE",
        "filename": "gemma-4-26B-A4B-it-UD-Q4_K_M.gguf",
        "quant": "Q4_K_M",
        "size_gb": 15.78,
        "min_vram_gb": 8.0,
        "recommended_vram_gb": 16.0,
        "type": "moe",
        "active_params": "4B",
        "category": "대형 MoE",
        "desc": "Google Gemma 4 MoE / 활성 4B 파라미터 / GPU+RAM 분산 구동",
        "base_priority": 72,
        "args": "-ngl 20 --n-cpu-moe 8 -c 16384 --cache-type-k q4_0 --cache-type-v q4_0"
    },
    {
        "id": "qwen3.6-35b-moe",
        "repo": "unsloth/Qwen3.6-35B-A3B-GGUF",
        "name": "Qwen 3.6 35B MoE",
        "filename": "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf",
        "quant": "Q4_K_M",
        "size_gb": 20.61,
        "min_vram_gb": 10.0,
        "recommended_vram_gb": 24.0,
        "type": "moe",
        "active_params": "3B",
        "category": "고지능 MoE",
        "desc": "35B MoE / 활성 3B / 대용량 RAM을 활용한 하이브리드 구동",
        "base_priority": 70,
        "args": "-ngl 16 --n-cpu-moe 16 -c 16384 --cache-type-k q4_0 --cache-type-v q4_0"
    }
]


def detect_hardware():
    """Detects CPU, RAM, GPU, and disk information."""
    info = {
        "gpu": {
            "count": 0,
            "name": "CPU-Only",
            "total_vram_gb": 0.0,
            "free_vram_gb": 0.0,
            "driver": "N/A"
        },
        "cpu": {
            "model": "Unknown CPU",
            "cores": os.cpu_count() or 1
        },
        "ram": {
            "total_gb": 0.0,
            "avail_gb": 0.0
        },
        "disk": {
            "models_dir": MODELS_DIR_DEFAULT,
            "free_gb": 0.0
        }
    }

    # GPU
    try:
        res = subprocess.run([
            "nvidia-smi", "--query-gpu=name,memory.total,memory.free,driver_version",
            "--format=csv,noheader,nounits"
        ], capture_output=True, text=True, check=True)
        lines = [l.strip() for l in res.stdout.strip().split("\n") if l.strip()]
        if lines:
            parts = [p.strip() for p in lines[0].split(",")]
            info["gpu"]["count"] = len(lines)
            info["gpu"]["name"] = parts[0]
            info["gpu"]["total_vram_gb"] = round(float(parts[1]) / 1024, 1)
            info["gpu"]["free_vram_gb"] = round(float(parts[2]) / 1024, 1)
            info["gpu"]["driver"] = parts[3]
    except Exception:
        pass

    # CPU
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if "model name" in line:
                    info["cpu"]["model"] = line.split(":", 1)[1].strip()
                    break
    except Exception:
        pass

    # RAM
    try:
        with open("/proc/meminfo") as f:
            mem = {}
            for line in f:
                parts = line.split(":")
                if len(parts) == 2:
                    mem[parts[0].strip()] = int(parts[1].split()[0])
            info["ram"]["total_gb"] = round(mem.get("MemTotal", 0) / (1024 * 1024), 1)
            info["ram"]["avail_gb"] = round(mem.get("MemAvailable", 0) / (1024 * 1024), 1)
    except Exception:
        pass

    # Disk
    models_dir = os.environ.get("MODELS_DIR", MODELS_DIR_DEFAULT)
    models_dir = os.path.expanduser(models_dir)
    os.makedirs(models_dir, exist_ok=True)
    info["disk"]["models_dir"] = models_dir
    usage = shutil.disk_usage(models_dir)
    info["disk"]["free_gb"] = round(usage.free / (1024 ** 3), 1)

    return info


def score_and_rank_models(hw, models_dir):
    """
    Ranks models based on hardware compatibility:
    - 100% GPU Offload: Model + KV cache <= VRAM
    - GPU+CPU MoE: Active params fit in VRAM, full weights fit in RAM
    - Returns top 10 models with computed install status
    """
    vram = hw["gpu"]["total_vram_gb"]
    ram_avail = hw["ram"]["avail_gb"]
    is_gpu = hw["gpu"]["count"] > 0

    scored = []
    for m in UNSLOTH_CATALOG:
        m_copy = dict(m)
        full_path = os.path.join(models_dir, m_copy["filename"])
        m_copy["is_downloaded"] = os.path.isfile(full_path)
        if m_copy["is_downloaded"]:
            try:
                m_copy["local_size_gb"] = round(os.path.getsize(full_path) / (1024 ** 3), 2)
            except OSError:
                m_copy["local_size_gb"] = m_copy["size_gb"]
        else:
            m_copy["local_size_gb"] = m_copy["size_gb"]

        base_score = m_copy.get("base_priority", 50)

        # Offload determination and compatibility scoring
        if is_gpu:
            if m_copy["size_gb"] + 2.0 <= vram:
                # 100% GPU offloadable
                m_copy["offload_tag"] = "100% GPU"
                score = base_score + 30
            elif m_copy["type"] == "moe" and (m_copy["size_gb"] < ram_avail):
                # MoE offloading to RAM
                m_copy["offload_tag"] = "GPU+CPU"
                score = base_score + 10
            elif m_copy["size_gb"] <= vram + 4.0:
                # Partial GPU offload
                m_copy["offload_tag"] = "GPU+CPU"
                score = base_score + 5
            else:
                m_copy["offload_tag"] = "GPU+CPU"
                score = base_score - 10
        else:
            # CPU only
            m_copy["offload_tag"] = "CPU-Only"
            score = base_score - (m_copy["size_gb"] * 2)

        # Bonus if already downloaded
        if m_copy["is_downloaded"]:
            score += 25

        m_copy["score"] = score
        scored.append(m_copy)

    # Sort descending by score
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:10]


def format_whiptail_menu_items(models):
    """Formats models into id and item_text for whiptail --menu."""
    items = []
    for idx, m in enumerate(models, 1):
        status_mark = "✓" if m["is_downloaded"] else "⬇"
        status_text = "보유" if m["is_downloaded"] else "다운"
        item_label = f"[{status_mark} {status_text}] {m['name']} ({m['size_gb']:.1f}GB) [{m['offload_tag']}] │ {m['desc']}"
        items.append(str(idx))
        items.append(item_label)
    return items


def print_detailed_environment(hw):
    """Prints beautiful colored environment box in terminal."""
    gpu = hw["gpu"]
    cpu = hw["cpu"]
    ram = hw["ram"]
    disk = hw["disk"]

    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("🖥️  시스템 구동 환경 감지 결과 (Hardware Profile):")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    if gpu["count"] > 0:
        print(f"• GPU 모델  : {gpu['name']} ({gpu['count']}개 장착)")
        print(f"• GPU VRAM  : 전체 {gpu['total_vram_gb']} GB │ 현재 가용 {gpu['free_vram_gb']} GB")
        print(f"• 드라이버  : NVIDIA {gpu['driver']}")
    else:
        print("• GPU 모델  : 없음 (CPU 모드로 구동)")

    print(f"• CPU 모델  : {cpu['model']} ({cpu['cores']} vCPU)")
    print(f"• 시스템 RAM: 전체 {ram['total_gb']} GB │ 현재 가용 {ram['avail_gb']} GB")
    print(f"• 모델저장소: {disk['models_dir']} (디스크 여유 공간: {disk['free_gb']} GB)")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")


def download_file(url, dest_path, expected_size_gb=0, model_id=None):
    """
    Downloads file with real-time speed, ETA, and progress display.
    Supports resuming interrupted downloads.
    Logs start, progress, and completion to PostgreSQL via db_manager.
    """
    db = DBManager() if DBManager else None
    download_id = None
    if db and db.is_available():
        download_id = db.log_download_start(
            model_id=model_id,
            filename=os.path.basename(dest_path),
            local_path=dest_path,
            download_url=url,
            total_bytes=int(expected_size_gb * (1024**3))
        )

    tmp_path = dest_path + ".download"
    initial_bytes = 0

    if os.path.exists(tmp_path):
        initial_bytes = os.path.getsize(tmp_path)

    req = urllib.request.Request(url, headers={
        "User-Agent": "Mozilla/5.0 (compatible; localLLMService/1.0)"
    })
    if initial_bytes > 0:
        req.add_header("Range", f"bytes={initial_bytes}-")

    print(f"\n🌐 Hugging Face CDN 연결 중: {url}")
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            content_range = resp.headers.get("Content-Range")
            total_bytes = None
            if content_range:
                try:
                    total_bytes = int(content_range.split("/")[-1])
                except Exception:
                    pass
            if total_bytes is None:
                cl = resp.headers.get("Content-Length")
                if cl:
                    total_bytes = int(cl) + initial_bytes

            mode = "ab" if initial_bytes > 0 else "wb"
            downloaded = initial_bytes
            start_time = time.time()
            last_print = 0
            chunk_size = 1024 * 512  # 512 KB

            total_display_gb = (total_bytes / (1024**3)) if total_bytes else expected_size_gb
            print(f"📦 다운로드 대상: {os.path.basename(dest_path)} (총 {total_display_gb:.2f} GB)")
            if initial_bytes > 0:
                print(f"🔄 이어받기 시작: {initial_bytes / (1024**3):.2f} GB 부터 이어서 다운로드합니다.")

            with open(tmp_path, mode) as out_f:
                while True:
                    chunk = resp.read(chunk_size)
                    if not chunk:
                        break
                    out_f.write(chunk)
                    downloaded += len(chunk)

                    now = time.time()
                    if now - last_print >= 0.25:
                        elapsed = now - start_time
                        speed = (downloaded - initial_bytes) / elapsed if elapsed > 0 else 0
                        speed_mb = speed / (1024 * 1024)
                        cur_gb = downloaded / (1024 ** 3)

                        if total_bytes and total_bytes > 0:
                            percent = int(downloaded * 100 / total_bytes)
                            remaining_bytes = max(0, total_bytes - downloaded)
                            eta_sec = int(remaining_bytes / speed) if speed > 0 else 0
                            eta_m, eta_s = divmod(eta_sec, 60)
                            bar_len = 30
                            filled = int(bar_len * percent / 100)
                            bar = "=" * filled + (">" if filled < bar_len else "") + " " * (bar_len - filled - (1 if filled < bar_len else 0))
                            sys.stdout.write(f"\r📥 [{bar}] {percent:3d}% | {cur_gb:5.2f}/{total_display_gb:.2f} GB | {speed_mb:5.1f} MB/s | ETA {eta_m:02d}:{eta_s:02d} ")
                        else:
                            sys.stdout.write(f"\r📥 {cur_gb:5.2f} GB 완료 | {speed_mb:5.1f} MB/s")
                        sys.stdout.flush()
                        last_print = now

            duration_sec = int(time.time() - start_time)
            speed_mb = (downloaded - initial_bytes) / (duration_sec * 1024 * 1024) if duration_sec > 0 else 0
            print("\n✨ 다운로드 완료! 파일 검증 및 저장 중...")
            shutil.move(tmp_path, dest_path)
            print(f"✅ 저장 완료: {dest_path}")

            if db and download_id:
                db.log_download_complete(download_id, downloaded, speed_mb, duration_sec)
            return True

    except urllib.error.HTTPError as e:
        if e.code == 416 and initial_bytes > 0:
            shutil.move(tmp_path, dest_path)
            print(f"✅ 파일이 이미 완전히 다운로드되어 있습니다: {dest_path}")
            if db and download_id:
                db.log_download_complete(download_id, initial_bytes, 0, 0)
            return True
        print(f"\n❌ 다운로드 HTTP 에러 ({e.code}): {e.reason}")
        if db and download_id:
            db.log_download_failed(download_id, f"HTTP Error {e.code}: {e.reason}")
        return False
    except Exception as e:
        print(f"\n❌ 다운로드 실패 ({type(e).__name__}): {e}")
        if db and download_id:
            db.log_download_failed(download_id, str(e))
        return False


def main():
    if len(sys.argv) < 2:
        print("Usage: hf_recommender.py [env|env-detail|env-summary|recommend|whiptail-menu|get-model|download]")
        sys.exit(1)

    cmd = sys.argv[1]
    models_dir = os.environ.get("MODELS_DIR", MODELS_DIR_DEFAULT)
    models_dir = os.path.expanduser(models_dir)

    # 1. PC Environment info (JSON)
    if cmd == "env":
        hw = detect_hardware()
        print(json.dumps(hw, ensure_ascii=False))

    # 2. Detailed environment print
    elif cmd == "env-detail":
        hw = detect_hardware()
        print_detailed_environment(hw)

    # 3. Formatted environment summary for whiptail
    elif cmd == "env-summary":
        hw = detect_hardware()
        gpu_txt = f"{hw['gpu']['name']} ({hw['gpu']['total_vram_gb']}GB VRAM / 가용 {hw['gpu']['free_vram_gb']}GB)" if hw['gpu']['count'] > 0 else "CPU Only"
        ram_txt = f"{hw['ram']['total_gb']}GB (가용 {hw['ram']['avail_gb']}GB)"
        cpu_txt = f"{hw['cpu']['model'].split('@')[0].strip()} ({hw['cpu']['cores']}코어)"
        disk_txt = f"{hw['disk']['free_gb']}GB"
        print(f"🖥️ GPU: {gpu_txt} │ RAM: {ram_txt}\n💻 CPU: {cpu_txt} │ 💾 저장소 여유: {disk_txt}")

    # 4. Recommend TOP 10 models (JSON)
    elif cmd == "recommend":
        if DBManager:
            db = DBManager()
            db.sync_catalog(UNSLOTH_CATALOG)
        hw = detect_hardware()
        top10 = score_and_rank_models(hw, models_dir)
        print(json.dumps(top10, ensure_ascii=False, indent=2))

    # 5. Formatted menu items for whiptail
    elif cmd == "whiptail-menu":
        if DBManager:
            db = DBManager()
            db.sync_catalog(UNSLOTH_CATALOG)
        hw = detect_hardware()
        top10 = score_and_rank_models(hw, models_dir)
        items = format_whiptail_menu_items(top10)
        print(json.dumps(items, ensure_ascii=False))

    # 6. Get model by 1-based index
    elif cmd == "get-model":
        if len(sys.argv) < 3:
            sys.exit(1)
        idx = int(sys.argv[2]) - 1
        hw = detect_hardware()
        top10 = score_and_rank_models(hw, models_dir)
        if 0 <= idx < len(top10):
            print(json.dumps(top10[idx], ensure_ascii=False))
        else:
            sys.exit(1)

    # 7. Download model
    elif cmd == "download":
        if len(sys.argv) < 3:
            print("Usage: hf_recommender.py download <1-based index or repo:filename>")
            sys.exit(1)
        target = sys.argv[2]
        hw = detect_hardware()
        top10 = score_and_rank_models(hw, models_dir)
        mid = None

        if target.isdigit():
            idx = int(target) - 1
            if not (0 <= idx < len(top10)):
                print(f"Error: Invalid index {target}")
                sys.exit(1)
            model = top10[idx]
            mid = model["id"]
            repo = model["repo"]
            filename = model["filename"]
            expected_size = model["size_gb"]
        else:
            repo, filename = target.split(":", 1)
            expected_size = 5.0

        dest = os.path.join(models_dir, filename)
        url = f"https://huggingface.co/{repo}/resolve/main/{filename}"

        if os.path.isfile(dest):
            print(f"✅ 모델 파일이 이미 존재합니다: {dest}")
            sys.exit(0)

        success = download_file(url, dest, expected_size, model_id=mid)
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
