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
import struct
import re

try:
    from db_manager import DBManager
except ImportError:
    DBManager = None


def load_dotenv(env_path=None):
    """Loads key-value pairs from .env into os.environ if not already present."""
    if env_path is None:
        env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if not os.path.isfile(env_path):
        return
    try:
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    k, v = line.split("=", 1)
                    k = k.strip()
                    v = v.strip().strip('"').strip("'")
                    if k and k not in os.environ:
                        os.environ[k] = v
    except Exception:
        pass


# Automatically load .env at module import
load_dotenv()


def get_hf_credentials():
    """Extracts Hugging Face credentials and configuration from environment."""
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN") or os.environ.get("HF_API_KEY")
    username = os.environ.get("HF_USERNAME")
    endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
    return {
        "token": token.strip() if token else None,
        "username": username.strip() if username else None,
        "endpoint": endpoint
    }


def get_hf_account_info(token=None):
    """Queries Hugging Face API (whoami-v2) to verify account and token validity."""
    creds = get_hf_credentials()
    tok = token or creds["token"]
    if not tok:
        return {
            "configured": False,
            "authenticated": False,
            "username": None,
            "fullname": None,
            "email": None,
            "plan": None,
            "is_pro": False,
            "message": "HF_TOKEN이 .env에 설정되지 않았습니다 (익명 다운로드 모드)."
        }

    url = f"{creds['endpoint']}/api/whoami-v2"
    req = urllib.request.Request(url, headers={
        "Authorization": f"Bearer {tok}",
        "User-Agent": "localLLMService/1.0 (huggingface-api)"
    })
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return {
                "configured": True,
                "authenticated": True,
                "username": data.get("name", creds.get("username")),
                "fullname": data.get("fullname", ""),
                "email": data.get("email", ""),
                "plan": data.get("plan", "Free"),
                "is_pro": data.get("isPro", False),
                "message": f"계정 인증 성공: {data.get('name')} ({data.get('fullname', '')})"
            }
    except urllib.error.HTTPError as e:
        if e.code == 401:
            return {
                "configured": True,
                "authenticated": False,
                "username": None,
                "fullname": None,
                "email": None,
                "plan": None,
                "is_pro": False,
                "message": "인증 실패: .env의 HF_TOKEN이 유효하지 않습니다."
            }
        return {
            "configured": True,
            "authenticated": False,
            "username": None,
            "fullname": None,
            "email": None,
            "plan": None,
            "is_pro": False,
            "message": f"Hugging Face API 응답 오류 (HTTP {e.code})"
        }
    except Exception as e:
        return {
            "configured": True,
            "authenticated": False,
            "username": None,
            "fullname": None,
            "email": None,
            "plan": None,
            "is_pro": False,
            "message": f"Hugging Face API 연결 실패: {e}"
        }


def check_hf_model_access(url, token=None):
    """
    Performs pre-flight HEAD request via Hugging Face API to verify file existence and permissions.
    Returns (can_access: bool, error_message: str, http_code: int).
    """
    creds = get_hf_credentials()
    tok = token or creds["token"]
    req = urllib.request.Request(url, headers={
        "User-Agent": "localLLMService/1.0 (huggingface-api)"
    }, method="HEAD")
    if tok:
        req.add_header("Authorization", f"Bearer {tok}")

    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return True, "", resp.status
    except urllib.error.HTTPError as e:
        if e.code == 401:
            return False, "Hugging Face 인증 실패: .env 파일의 HF_TOKEN이 유효하지 않거나 권한이 없습니다.", 401
        elif e.code == 403:
            return False, "Hugging Face 접근 제한: 해당 모델은 라이선스 동의(Gated Model)가 필요합니다. HF 웹사이트에서 접근 동의를 받아주세요.", 403
        elif e.code == 404:
            return False, "Hugging Face 저장소에서 모델 파일을 찾을 수 없습니다 (404 Not Found).", 404
        return False, f"Hugging Face API HTTP 에러 ({e.code}): {e.reason}", e.code
    except Exception as e:
        # Network errors are handled during actual download
        return True, "", 0


MODELS_DIR_DEFAULT = os.path.expanduser("~/Projects/models")

# Local Custom Models (Fine-tuned, LoRA Adapter, and Base)
LOCAL_MODELS = [
    {
        "id": "local-gemma-4-e4b-ft",
        "repo": "local/finetuned",
        "name": "[LOCAL / 병합] Gemma 4 E4B IT - Finetuned",
        "filename": "gemma-4-E4B-it-ft-Q4_K_M.gguf",
        "quant": "Q4_K_M",
        "size_gb": 5.0,
        "min_vram_gb": 5.5,
        "recommended_vram_gb": 6.5,
        "type": "dense",
        "active_params": "4B",
        "category": "로컬 파인튜닝",
        "desc": "파인튜닝 가중치 병합 완제품 모델",
        "base_priority": 1000,
        "args": "-ngl 99 -c 16384 --cache-type-k q4_0 --cache-type-v q4_0"
    },
    {
        "id": "local-gemma-4-e4b-lora",
        "repo": "local/lora",
        "name": "[LOCAL / LoRA] Gemma 4 E4B IT + Adapter",
        "filename": "gemma-4-E4B-it-base-Q4_K_M.gguf",
        "lora_file": "gemma-4-E4B-it-lora-f16.gguf",
        "quant": "Q4_K_M + f16",
        "size_gb": 5.07,
        "min_vram_gb": 5.5,
        "recommended_vram_gb": 6.5,
        "type": "dense",
        "active_params": "4B",
        "category": "로컬 파인튜닝",
        "desc": "순정 베이스에 LoRA 어댑터 결합 로드",
        "base_priority": 990,
        "args": "-ngl 99 -c 16384 --cache-type-k q4_0 --cache-type-v q4_0"
    },
    {
        "id": "local-gemma-4-e4b-base",
        "repo": "local/base",
        "name": "[LOCAL / 순정] Gemma 4 E4B IT Base",
        "filename": "gemma-4-E4B-it-base-Q4_K_M.gguf",
        "quant": "Q4_K_M",
        "size_gb": 5.0,
        "min_vram_gb": 5.5,
        "recommended_vram_gb": 6.5,
        "type": "dense",
        "active_params": "4B",
        "category": "로컬 베이스",
        "desc": "순정 베이스 모델 (비교 대조군)",
        "base_priority": 980,
        "args": "-ngl 99 -c 16384 --cache-type-k q4_0 --cache-type-v q4_0"
    }
]

# Curated High-Quality Unsloth GGUF Models with verified Hugging Face CDN links
UNSLOTH_CATALOG = [
    {
        "id": "qwen3.8-27b-q4",
        "repo": "unsloth/Qwen3.8-27B-GGUF",
        "name": "Qwen3.8 27B (UD-Q4_K_M)",
        "filename": "Qwen3.8-27B-UD-Q4_K_M.gguf",
        "quant": "UD-Q4_K_M",
        "size_gb": 15.33,
        "min_vram_gb": 8.0,
        "recommended_vram_gb": 12.0,
        "type": "dense",
        "active_params": "27B",
        "category": "최신 Qwen3.8 플래그십",
        "desc": "HF 665만 DL 1위! 최신 Qwen3.8 27B 정밀 4-bit / RTX 3060 12GB+78GB RAM 하이브리드 최적",
        "base_priority": 105,
        "args": "-ngl 28 -c 16384 --cache-type-k q4_0 --cache-type-v q4_0"
    },
    {
        "id": "qwen3.8-27b-100gpu",
        "repo": "unsloth/Qwen3.8-27B-GGUF",
        "name": "Qwen3.8 27B (UD-Q2_K_XL · 100% GPU)",
        "filename": "Qwen3.8-27B-UD-Q2_K_XL.gguf",
        "quant": "UD-Q2_K_XL",
        "size_gb": 9.15,
        "min_vram_gb": 10.5,
        "recommended_vram_gb": 12.0,
        "type": "dense",
        "active_params": "27B",
        "category": "Qwen3.8 27B (100% GPU)",
        "desc": "Unsloth Dynamic 2.0 양자화 / 27B 대형 모델을 RTX 3060 12GB VRAM에 100% 단독 풀로드",
        "base_priority": 104,
        "args": "-ngl 99 -c 8192 --cache-type-k q4_0 --cache-type-v q4_0"
    },
    {
        "id": "qwen3.8-9b-distill",
        "repo": "empero-ai/Qwen3.8-9B-Distill-GGUF",
        "name": "Qwen3.8 9B Distill",
        "filename": "Qwen3.8-9B-Q4_K_M.gguf",
        "quant": "Q4_K_M",
        "size_gb": 5.38,
        "min_vram_gb": 7.0,
        "recommended_vram_gb": 10.0,
        "type": "dense",
        "active_params": "9B",
        "category": "최신 Qwen3.8 (100% GPU)",
        "desc": "Qwen3.8 추론 지능 9B 증류 모델 / 12GB VRAM 100% GPU 초고속 구동 (32K context)",
        "base_priority": 103,
        "args": "-ngl 99 -c 32768 --flash-attn on --cache-type-k q8_0 --cache-type-v q8_0"
    },
    {
        "id": "qwen3.8-35b-a3b-moe",
        "repo": "empero-ai/Qwen3.8-35B-A3B-Distill-GGUF",
        "name": "Qwen3.8 35B-A3B MoE Distill",
        "filename": "Qwen3.8-35B-A3B-Q4_K_M.gguf",
        "quant": "Q4_K_M",
        "size_gb": 20.22,
        "min_vram_gb": 8.0,
        "recommended_vram_gb": 12.0,
        "type": "moe",
        "active_params": "3B",
        "category": "최신 Qwen3.8 MoE",
        "desc": "총 35B 중 활성 3B MoE / RTX 3060 12GB+78GB RAM 환경에서 3B 속도로 35B 지능 구현",
        "base_priority": 102,
        "args": "-ngl 20 --n-cpu-moe 12 -c 16384 --cache-type-k q4_0 --cache-type-v q4_0"
    },
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
        "category": "고지능 범용 (보유)",
        "desc": "Alibaba 최신 9B 초고성능 모델 / 32K context 지원 / 범용 최상급 검증 완료",
        "base_priority": 100,
        "args": "-ngl 99 -c 32768 --flash-attn on -ctk q8_0 -ctv q8_0 -b 2048 -ub 512"
    },
    {
        "id": "qwen3.6-27b",
        "repo": "unsloth/Qwen3.6-27B-GGUF",
        "name": "Qwen3.6 27B",
        "filename": "Qwen3.6-27B-Q4_K_M.gguf",
        "quant": "Q4_K_M",
        "size_gb": 15.66,
        "min_vram_gb": 8.0,
        "recommended_vram_gb": 12.0,
        "type": "dense",
        "active_params": "27B",
        "category": "27B 고지능 추론",
        "desc": "Unsloth Qwen3.6 27B Dense 모델 / 15.7GB / GPU+78GB RAM 하이브리드 구동",
        "base_priority": 99,
        "args": "-ngl 28 -c 16384 --cache-type-k q4_0 --cache-type-v q4_0"
    },
    {
        "id": "lfm2.5-24b-a2b-moe",
        "repo": "unsloth/LFM2.5-24B-A2B-GGUF",
        "name": "Liquid LFM2.5 24B-A2B MoE",
        "filename": "LFM2.5-24B-A2B-UD-Q4_K_XL.gguf",
        "quant": "UD-Q4_K_XL",
        "size_gb": 13.86,
        "min_vram_gb": 8.0,
        "recommended_vram_gb": 12.0,
        "type": "moe",
        "active_params": "2B",
        "category": "초고속 24B MoE",
        "desc": "2026.10 최신 Liquid AI 24B MoE (활성 2B!) / 13.9GB / GPU+RAM 초고속 응답",
        "base_priority": 99,
        "args": "-ngl 26 --n-cpu-moe 8 -c 16384 --cache-type-k q4_0 --cache-type-v q4_0"
    },
    {
        "id": "nemotron-3.5-lightning-30b-moe",
        "repo": "unsloth/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF",
        "name": "NVIDIA Nemotron 3.5 Lightning 30B-A3B",
        "filename": "NVIDIA-Nemotron-3.5-Lightning-30B-A3B-UD-Q4_K_M.gguf",
        "quant": "UD-Q4_K_M",
        "size_gb": 23.53,
        "min_vram_gb": 8.0,
        "recommended_vram_gb": 12.0,
        "type": "moe",
        "active_params": "3B",
        "category": "NVIDIA 30B MoE",
        "desc": "NVIDIA 최신 Nemotron 3.5 Lightning (활성 3B MoE) / 1M 컨텍스트 지원 / GPU+78GB RAM",
        "base_priority": 98,
        "args": "-ngl 18 --n-cpu-moe 14 -c 16384 --cache-type-k q4_0 --cache-type-v q4_0"
    },
    {
        "id": "gpt-oss-20b",
        "repo": "unsloth/gpt-oss-20b-GGUF",
        "name": "OpenAI gpt-oss 20B",
        "filename": "gpt-oss-20b-Q4_K_M.gguf",
        "quant": "Q4_K_M",
        "size_gb": 10.83,
        "min_vram_gb": 8.0,
        "recommended_vram_gb": 12.0,
        "type": "dense",
        "active_params": "20B",
        "category": "OpenAI 오픈웨이트 20B",
        "desc": "OpenAI 오픈웨이트 20B (10.8GB) / RTX 3060 12GB VRAM 대부분 오프로드 + RAM 보조",
        "base_priority": 98,
        "args": "-ngl 32 -c 16384 --cache-type-k q4_0 --cache-type-v q4_0"
    },
    {
        "id": "qwen2.5-coder-32b",
        "repo": "unsloth/Qwen2.5-Coder-32B-Instruct-GGUF",
        "name": "Qwen2.5-Coder 32B Instruct",
        "filename": "Qwen2.5-Coder-32B-Instruct-Q4_K_M.gguf",
        "quant": "Q4_K_M",
        "size_gb": 18.49,
        "min_vram_gb": 8.0,
        "recommended_vram_gb": 12.0,
        "type": "dense",
        "active_params": "32B",
        "category": "32B 코딩 종결자",
        "desc": "Claude 3.5 Sonnet급 오픈소스 코딩 1위 / 18.5GB / GPU+78GB RAM 하이브리드 구동",
        "base_priority": 98,
        "args": "-ngl 28 -c 16384 --cache-type-k q4_0 --cache-type-v q4_0"
    },
    {
        "id": "deepseek-r1-32b",
        "repo": "unsloth/DeepSeek-R1-Distill-Qwen-32B-GGUF",
        "name": "DeepSeek-R1-Distill 32B",
        "filename": "DeepSeek-R1-Distill-Qwen-32B-Q4_K_M.gguf",
        "quant": "Q4_K_M",
        "size_gb": 18.49,
        "min_vram_gb": 8.0,
        "recommended_vram_gb": 12.0,
        "type": "dense",
        "active_params": "32B",
        "category": "32B 심층추론 끝판왕",
        "desc": "DeepSeek-R1 32B 최고봉 CoT 사고 추론 / 수학·알고리즘 극한 성능 / GPU+RAM",
        "base_priority": 97,
        "args": "-ngl 28 -c 16384 --cache-type-k q4_0 --cache-type-v q4_0"
    },
    {
        "id": "deepseek-r1-14b",
        "repo": "unsloth/DeepSeek-R1-Distill-Qwen-14B-GGUF",
        "name": "DeepSeek-R1-Distill 14B",
        "filename": "DeepSeek-R1-Distill-Qwen-14B-Q4_K_M.gguf",
        "quant": "Q4_K_M",
        "size_gb": 8.37,
        "min_vram_gb": 10.0,
        "recommended_vram_gb": 12.0,
        "type": "dense",
        "active_params": "14B",
        "category": "14B 심층추론 (100% GPU)",
        "desc": "14B 고지능 R1 추론 / 12GB VRAM 100% 풀로드 한계치 최적화",
        "base_priority": 96,
        "args": "-ngl 99 -c 8192 --cache-type-k q4_0 --cache-type-v q4_0"
    },
    {
        "id": "qwen2.5-coder-14b",
        "repo": "unsloth/Qwen2.5-Coder-14B-Instruct-GGUF",
        "name": "Qwen2.5-Coder 14B Instruct",
        "filename": "Qwen2.5-Coder-14B-Instruct-Q4_K_M.gguf",
        "quant": "Q4_K_M",
        "size_gb": 8.37,
        "min_vram_gb": 10.0,
        "recommended_vram_gb": 12.0,
        "type": "dense",
        "active_params": "14B",
        "category": "14B 코딩 최강 (100% GPU)",
        "desc": "14B 코딩 종결자 / 12GB VRAM 100% GPU 단독 풀로드 초고속 구동",
        "base_priority": 95,
        "args": "-ngl 99 -c 16384 --cache-type-k q4_0 --cache-type-v q4_0"
    },
    {
        "id": "phi-4-14b",
        "repo": "unsloth/phi-4-GGUF",
        "name": "Phi-4 14B",
        "filename": "phi-4-Q4_K_M.gguf",
        "quant": "Q4_K_M",
        "size_gb": 8.28,
        "min_vram_gb": 10.0,
        "recommended_vram_gb": 12.0,
        "type": "dense",
        "active_params": "14B",
        "category": "14B 수학·추론 (100% GPU)",
        "desc": "Microsoft 최신 14B 플래그십 / 12GB VRAM 100% GPU 단독 풀 오프로드",
        "base_priority": 94,
        "args": "-ngl 99 -c 16384 --cache-type-k q4_0 --cache-type-v q4_0"
    },
    {
        "id": "mistral-small-24b",
        "repo": "bartowski/Mistral-Small-24B-Instruct-2501-GGUF",
        "name": "Mistral Small 24B Instruct 2501",
        "filename": "Mistral-Small-24B-Instruct-2501-Q4_K_M.gguf",
        "quant": "Q4_K_M",
        "size_gb": 13.35,
        "min_vram_gb": 8.0,
        "recommended_vram_gb": 12.0,
        "type": "dense",
        "active_params": "24B",
        "category": "24B 고성능 범용",
        "desc": "Mistral AI 최신 24B (2501) 모델 / 13.4GB / GPU+78GB RAM 분산 고속 추론",
        "base_priority": 93,
        "args": "-ngl 22 -c 16384 --cache-type-k q4_0 --cache-type-v q4_0"
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
        "category": "7B 심층 추론 (100% GPU)",
        "desc": "DeepSeek-R1 수학·코딩 추론 증류 모델 / CoT Thinking 탁월",
        "base_priority": 92,
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
        "category": "7B 코딩 특화 (100% GPU)",
        "desc": "오픈소스 7B 코딩 최강자 / 코드 자동완성 및 디버깅 최적화",
        "base_priority": 91,
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
        "category": "12B 고성능 (100% GPU)",
        "desc": "Google 최신 Gemma 4 12B 모델 / 정밀 지시 추론",
        "base_priority": 90,
        "args": "-ngl 99 -c 16384 --cache-type-k q4_0 --cache-type-v q4_0"
    },
    {
        "id": "llama-3.3-70b",
        "repo": "unsloth/Llama-3.3-70B-Instruct-GGUF",
        "name": "Llama 3.3 70B Instruct",
        "filename": "Llama-3.3-70B-Instruct-UD-IQ3_XXS.gguf",
        "quant": "IQ3_XXS",
        "size_gb": 25.76,
        "min_vram_gb": 10.0,
        "recommended_vram_gb": 16.0,
        "type": "dense",
        "active_params": "70B",
        "category": "70B 플래그십",
        "desc": "Meta 최신 70B 모델 / 25.8GB / 78GB RAM 환경에서 안정적 CPU+GPU 하이브리드",
        "base_priority": 89,
        "args": "-ngl 18 -c 8192 --cache-type-k q4_0 --cache-type-v q4_0"
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
        "base_priority": 88,
        "args": "-ngl 24 --n-cpu-moe 8 -c 16384 --cache-type-k q4_0 --cache-type-v q4_0"
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
        "base_priority": 85,
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
        "base_priority": 84,
        "args": "-ngl 16 --n-cpu-moe 16 -c 16384 --cache-type-k q4_0 --cache-type-v q4_0"
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
        "category": "경량 고효율 (100% GPU)",
        "desc": "Google Gemma 4 경량 4B 모델 / 낮은 VRAM, 빠른 응답 속도",
        "base_priority": 80,
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
        "category": "초고속 경량 (100% GPU)",
        "desc": "2.4GB 가벼운 크기로 초당 80+ 토큰 초고속 생성",
        "base_priority": 78,
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
        "category": "MS 고효율 (100% GPU)",
        "desc": "Microsoft 고효율 소형 모델 / 128K context 지원",
        "base_priority": 75,
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
        "category": "초경량 일상 (100% GPU)",
        "desc": "Meta 경량 3B 모델 / 일상 대화 및 가벼운 어시스턴트용",
        "base_priority": 70,
        "args": "-ngl 99 -c 16384 --cache-type-k q4_0 --cache-type-v q4_0"
    }
]

# Curated High-Quality Korean Embedding Models (Hugging Face / TEI optimized)
KOREAN_EMBEDDING_CATALOG = [
    {
        "id": "bge-m3",
        "name": "BAAI/bge-m3",
        "repo": "BAAI/bge-m3",
        "params": "560M",
        "dim": 1024,
        "max_seq": 8192,
        "min_vram_gb": 1.5,
        "desc": "한국어/다국어 Dense+Sparse+Multi-Vector 하이브리드 SOTA",
        "is_featured": True,
    },
    {
        "id": "kure-v1",
        "name": "nlpai-lab/KURE-v1",
        "repo": "nlpai-lab/KURE-v1",
        "params": "335M",
        "dim": 1024,
        "max_seq": 4096,
        "min_vram_gb": 1.0,
        "desc": "Ko-MTEB 한국어 벤치마크 1위권 고품질 한국어 특화",
        "is_featured": True,
    },
    {
        "id": "ko-sroberta",
        "name": "jhgan/ko-sroberta-multitask",
        "repo": "jhgan/ko-sroberta-multitask",
        "params": "110M",
        "dim": 768,
        "max_seq": 512,
        "min_vram_gb": 0.4,
        "desc": "VRAM 400MB 미만 초경량, 빠른 추론 속도 및 실시간 검색",
        "is_featured": True,
    },
    {
        "id": "bge-m3-ko",
        "name": "dragonkue/BGE-m3-ko",
        "repo": "dragonkue/BGE-m3-ko",
        "params": "560M",
        "dim": 1024,
        "max_seq": 8192,
        "min_vram_gb": 1.5,
        "desc": "BGE-M3 기반 한국어 정제 데이터셋 특화 파인튜닝",
        "is_featured": False,
    },
    {
        "id": "multilingual-e5-small",
        "name": "intfloat/multilingual-e5-small",
        "repo": "intfloat/multilingual-e5-small",
        "params": "118M",
        "dim": 384,
        "max_seq": 512,
        "min_vram_gb": 0.35,
        "desc": "VRAM 350MB 극저용량 및 안정적인 다국어/한국어 검색",
        "is_featured": False,
    },
]


def get_embedding_recommendations(hw=None):
    """Evaluates Korean embedding models against current available GPU VRAM."""
    if hw is None:
        hw = detect_hardware()
    free_vram = hw.get("gpu", {}).get("free_vram_gb", 0.0)
    has_gpu = hw.get("gpu", {}).get("count", 0) > 0

    results = []
    for m in KOREAN_EMBEDDING_CATALOG:
        entry = dict(m)
        min_vram = m["min_vram_gb"]
        entry["current_free_vram_gb"] = free_vram

        if not has_gpu:
            entry["status"] = "cpu"
            entry["tag"] = "[CPU 모드]"
            entry["can_gpu"] = False
            entry["recommended_mode"] = "cpu"
        elif free_vram >= min_vram + 0.5:
            entry["status"] = "gpu_optimal"
            entry["tag"] = "[100% GPU / 원활]"
            entry["can_gpu"] = True
            entry["recommended_mode"] = "gpu"
        elif free_vram >= min_vram:
            entry["status"] = "gpu_tight"
            entry["tag"] = "[GPU 주의 / 타이트]"
            entry["can_gpu"] = True
            entry["recommended_mode"] = "gpu"
        else:
            entry["status"] = "vram_insufficient"
            entry["tag"] = "[VRAM 부족 / CPU 권장]"
            entry["can_gpu"] = False
            entry["recommended_mode"] = "cpu"
        results.append(entry)
    return results


def format_whiptail_embedding_menu_items(recommendations):
    """Formats recommendations into list suitable for whiptail menu."""
    items = []
    for idx, m in enumerate(recommendations, 1):
        items.append(str(idx))
        items.append(f"{m['name']} ({m['params']}) {m['tag']} - {m['desc']}")
    items.append("0")
    items.append("[선택 안 함] 임베딩 모델 추가 없이 메인 LLM 단독 서빙")
    return items


def print_embedding_summary(hw=None):
    """Prints a console summary of available Korean embedding models and VRAM status."""
    if hw is None:
        hw = detect_hardware()
    recs = get_embedding_recommendations(hw)
    free_vram = hw.get("gpu", {}).get("free_vram_gb", 0.0)
    total_vram = hw.get("gpu", {}).get("total_vram_gb", 0.0)

    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("🔍 한국어 특화 Hugging Face 임베딩(Embedding) 모델 추천 현황:")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    if hw.get("gpu", {}).get("count", 0) > 0:
        print(f"• 현재 가용 VRAM : {free_vram:.1f} GB / 총 {total_vram:.1f} GB ({hw['gpu']['name']})")
    else:
        print("• GPU 상태       : CPU-Only 환경 (CPU 모드로 구동 권장)")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    for idx, m in enumerate(recs, 1):
        print(f"  {idx}. {m['name']} ({m['params']} · {m['dim']}차원) {m['tag']}")
        print(f"     HF Repo: {m['repo']} │ 최소 권장 VRAM: {m['min_vram_gb']}GB")
        print(f"     설명   : {m['desc']}")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")


# GGUF Binary Protocol Definition & Type Sizes (v2/v3 Specification)
GGUF_TYPE_SIZE = {
    0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8
}

GGUF_FILE_TYPE_MAP = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1",
    7: "Q8_0", 8: "Q5_0", 9: "Q5_1",
    10: "Q2_K", 11: "Q3_K_S", 12: "Q3_K_M", 13: "Q3_K_L",
    14: "Q4_K_S", 15: "Q4_K_M", 16: "Q5_K_S", 17: "Q5_K_M",
    18: "Q6_K", 19: "IQ2_XXS", 20: "IQ2_XS", 21: "Q2_K_S",
    22: "IQ3_XS", 23: "IQ3_XXS", 24: "IQ1_S", 25: "IQ4_NL",
    26: "IQ3_S", 27: "IQ3_M", 28: "IQ2_S", 29: "IQ2_M",
    30: "IQ4_XS", 31: "IQ1_M", 32: "BF16"
}


def parse_gguf_metadata(filepath):
    """
    Parses GGUF binary v2/v3 header and metadata KV pairs in pure Python (zero dependency).
    Returns dict containing header info and all string/numeric metadata.
    """
    info = {"valid": False, "metadata": {}, "tensor_count": 0, "kv_count": 0}
    try:
        with open(filepath, "rb") as f:
            magic = f.read(4)
            if magic != b"GGUF":
                return info
            info["valid"] = True
            version = struct.unpack("<I", f.read(4))[0]
            info["version"] = version
            if version >= 2:
                n_tensors, n_kv = struct.unpack("<QQ", f.read(16))
            else:
                n_tensors, n_kv = struct.unpack("<II", f.read(8))
            info["tensor_count"] = n_tensors
            info["kv_count"] = n_kv

            meta = {}
            for _ in range(n_kv):
                klen_bytes = f.read(8)
                if len(klen_bytes) < 8:
                    break
                key_len = struct.unpack("<Q", klen_bytes)[0]
                key = f.read(key_len).decode("utf-8", errors="ignore")
                vtype_bytes = f.read(4)
                if len(vtype_bytes) < 4:
                    break
                val_type = struct.unpack("<I", vtype_bytes)[0]

                if val_type == 8:  # string
                    slen = struct.unpack("<Q", f.read(8))[0]
                    meta[key] = f.read(slen).decode("utf-8", errors="ignore")
                elif val_type in GGUF_TYPE_SIZE:
                    raw = f.read(GGUF_TYPE_SIZE[val_type])
                    if val_type in (4, 5):
                        meta[key] = struct.unpack("<I" if val_type == 4 else "<i", raw)[0]
                    elif val_type == 10:
                        meta[key] = struct.unpack("<Q", raw)[0]
                    elif val_type == 7:
                        meta[key] = struct.unpack("<?", raw)[0]
                elif val_type == 9:  # array
                    arr_hdr = f.read(12)
                    if len(arr_hdr) < 12:
                        break
                    atype, alen = struct.unpack("<IQ", arr_hdr)
                    if atype == 8:
                        arr = []
                        for _ in range(alen):
                            slen = struct.unpack("<Q", f.read(8))[0]
                            arr.append(f.read(slen).decode("utf-8", errors="ignore"))
                        meta[key] = arr
                    elif atype in GGUF_TYPE_SIZE:
                        f.seek(alen * GGUF_TYPE_SIZE[atype], 1)
                    else:
                        break
                else:
                    break
            info["metadata"] = meta
    except Exception as e:
        info["error"] = str(e)
    return info


def discover_local_models(models_dir):
    """
    Scans models_dir for all .gguf files not currently in curated catalog.
    Parses GGUF binary headers + filename heuristics to auto-register custom models.
    """
    if not os.path.exists(models_dir):
        return []

    known_filenames = {m["filename"] for m in LOCAL_MODELS} | {m["filename"] for m in UNSLOTH_CATALOG}
    for m in LOCAL_MODELS:
        if m.get("lora_file"):
            known_filenames.add(m["lora_file"])

    discovered = []
    try:
        entries = sorted(os.listdir(models_dir))
    except Exception:
        return []

    for fn in entries:
        if not fn.endswith(".gguf") or fn.endswith(".download") or fn.startswith("."):
            continue
        if fn in known_filenames:
            continue

        full_path = os.path.join(models_dir, fn)
        if not os.path.isfile(full_path):
            continue

        size_bytes = os.path.getsize(full_path)
        size_gb = round(size_bytes / (1024 ** 3), 2)

        gguf_info = parse_gguf_metadata(full_path)
        meta = gguf_info.get("metadata", {})
        arch = meta.get("general.architecture", "transformer")
        name_meta = meta.get("general.name", "")
        quant_by = meta.get("general.quantized_by", "")
        repo_url = meta.get("general.repo_url", "")

        # Check if LoRA adapter (standalone adapter cannot be run as primary base model)
        is_adapter = False
        if "lora" in fn.lower() or "adapter" in fn.lower() or size_gb < 0.3:
            if name_meta == "Final_Adapter" or "adapter" in name_meta.lower() or "lora" in fn.lower():
                is_adapter = True
        if is_adapter:
            continue

        # Quantization extraction
        ftype = meta.get("general.file_type")
        quant = GGUF_FILE_TYPE_MAP.get(ftype, "")
        if not quant:
            qm = re.search(r"(Q\d+_[KkMmSsLl0-9_]+|IQ\d+_[A-Za-z0-9_]+|Q\d+_0|Q\d+_1|F16|F32|BF16)", fn, re.IGNORECASE)
            quant = qm.group(1).upper() if qm else "Q4_K_M"

        # Active parameter size
        param = meta.get("general.size_label", "")
        if not param:
            pm = re.search(r"(\d+(\.\d+)?[Bb])", fn)
            param = pm.group(1).upper() if pm else "Unknown"

        # Model Name formatting
        clean_name = fn.replace(".gguf", "")
        clean_name = re.sub(r"[\._-]?(Q\d+_[A-Za-z0-9_]+|IQ\d+_[A-Za-z0-9_]+|Q\d+_0|Q\d+_1|F16|F32|BF16)", "", clean_name, flags=re.IGNORECASE)
        clean_name = re.sub(r"[-_]", " ", clean_name).strip()
        if not clean_name:
            clean_name = name_meta if name_meta and name_meta != "Gguf" else fn.replace(".gguf", "")

        # Context length
        ctx = meta.get(f"{arch}.context_length", 32768)
        if not isinstance(ctx, int) or ctx > 131072 or ctx <= 0:
            ctx = 32768

        # Determine if fine-tuned (unsloth, ft, merged)
        is_unsloth = (quant_by == "Unsloth" or "unsloth" in repo_url.lower() or "unsloth" in fn.lower())
        is_ft = is_unsloth or any(k in fn.lower() for k in ["ft", "finetune", "merged", "custom"])

        if is_unsloth:
            tag = "[LOCAL / Unsloth]"
            category = "로컬 파인튜닝 (Unsloth)"
            repo = "local/unsloth"
        elif is_ft:
            tag = "[LOCAL / 파인튜닝]"
            category = "로컬 파인튜닝"
            repo = "local/finetuned"
        else:
            tag = "[LOCAL / 커스텀]"
            category = "로컬 커스텀"
            repo = "local/custom"

        model_name = f"{tag} {clean_name} ({quant})"
        ctx_k = ctx // 1024 if isinstance(ctx, int) else 32
        desc = f"로컬 저장소 자동 감지 모델 ({arch}, {param}, {ctx_k}K context)"
        mid = f"local-auto-{re.sub(r'[^a-zA-Z0-9]', '-', fn.lower()).strip('-')}"

        if size_gb + 2.0 <= 12.0:
            args = f"-ngl 99 -c {min(ctx, 32768)} --cache-type-k q4_0 --cache-type-v q4_0"
        else:
            args = f"-ngl 24 -c {min(ctx, 16384)} --cache-type-k q4_0 --cache-type-v q4_0"

        discovered.append({
            "id": mid,
            "repo": repo,
            "name": model_name,
            "filename": fn,
            "quant": quant,
            "size_gb": size_gb,
            "min_vram_gb": round(size_gb + 1.0, 1),
            "recommended_vram_gb": round(size_gb + 2.0, 1),
            "type": "dense",
            "active_params": param,
            "category": category,
            "desc": desc,
            "base_priority": 960 if is_ft else 940,
            "args": args,
            "is_downloaded": True
        })

    return discovered


def get_all_local_models(models_dir, hw=None):
    """
    Returns a unified list of all GGUF models physically located in models_dir,
    including curated local models, auto-discovered custom models, and downloaded catalog models.
    """
    if hw is None:
        hw = detect_hardware()
    all_models = score_and_rank_models(hw, models_dir)
    return [m for m in all_models if m.get("is_downloaded")]


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
    - Returns local models and top HF models with computed install status
    """
    vram = hw["gpu"]["total_vram_gb"]
    ram_avail = hw["ram"]["avail_gb"]
    is_gpu = hw["gpu"]["count"] > 0

    # Auto-discover unlisted local GGUF models directly from models_dir
    discovered_local = discover_local_models(models_dir)
    all_catalog = LOCAL_MODELS + discovered_local + UNSLOTH_CATALOG
    scored = []
    for m in all_catalog:
        m_copy = dict(m)
        full_path = os.path.join(models_dir, m_copy["filename"])
        
        # Check LoRA adapter existence if specified
        is_lora = bool(m_copy.get("lora_file"))
        if is_lora:
            lora_path = os.path.join(models_dir, m_copy["lora_file"])
            m_copy["is_downloaded"] = os.path.isfile(full_path) and os.path.isfile(lora_path)
        else:
            m_copy["is_downloaded"] = os.path.isfile(full_path)

        # Skip local models if files are not present in models_dir
        if m_copy.get("repo", "").startswith("local/") and not m_copy["is_downloaded"]:
            continue

        if m_copy["is_downloaded"]:
            try:
                sz = os.path.getsize(full_path)
                if is_lora:
                    sz += os.path.getsize(lora_path)
                m_copy["local_size_gb"] = round(sz / (1024 ** 3), 2)
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
                m_copy["offload_tag"] = "GPU+CPU (MoE)"
                score = base_score + 18
            elif m_copy["size_gb"] <= vram + 4.0:
                # Partial GPU offload
                m_copy["offload_tag"] = "GPU+CPU"
                score = base_score + 15
            elif m_copy["size_gb"] < ram_avail:
                # High-capacity system RAM offload (e.g. 78GB RAM)
                m_copy["offload_tag"] = "GPU+대용량RAM"
                score = base_score + 12
            else:
                m_copy["offload_tag"] = "RAM초과"
                score = base_score - 20
        else:
            # CPU only
            m_copy["offload_tag"] = "CPU-Only"
            score = base_score - (m_copy["size_gb"] * 2)

        # Bonus if already downloaded
        if m_copy["is_downloaded"]:
            score += 35

        m_copy["score"] = score
        scored.append(m_copy)

    # Sort descending by score
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:35]


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


def determine_laya_tier(model_info, hw=None):
    """
    Determines the execution tier for Laya based on hardware specs and the selected model.
    Tiers:
      - 'gpu' (Tier 1): VRAM headroom >= 2.0 GB (Laya ~1.5GB + 0.5GB buffer). Runs on GPU (~30ms).
      - 'cpu' (Tier 2): VRAM headroom < 2.0 GB. Offloads to host 78GB RAM on CPU (~50ms), 0GB VRAM used.
      - 'bypass' (Tier 3): Explicitly disabled or no capacity.
    """
    if os.environ.get("LAYA_ENABLED", "true").lower() in ("false", "0", "no", "off"):
        return {
            "tier": "bypass",
            "device": "none",
            "reason": "사용자 또는 환경설정에 의해 Laya가 비활성화됨 (--no-laya)",
            "headroom_gb": 0.0
        }

    if hw is None:
        hw = detect_hardware()

    total_vram = hw["gpu"]["total_vram_gb"]
    if total_vram <= 0:
        return {
            "tier": "cpu",
            "device": "cpu",
            "reason": "GPU 미장착 시스템: CPU + 시스템 RAM 모드로 실행",
            "headroom_gb": 0.0
        }

    # Realistic model VRAM estimation:
    # Model weight size + 1.2GB for realistic quantized KV Cache (8K context with --cache-type-k q4_0)
    size_gb = float(model_info.get("size_gb") or model_info.get("min_vram_gb", 5.0))
    if size_gb >= total_vram - 1.0:
        estimated_llm_vram = total_vram
    else:
        estimated_llm_vram = round(size_gb + 1.2, 1)

    # Calculate realistic available VRAM headroom after loading the LLM
    headroom = round(total_vram - estimated_llm_vram, 2)

    # Laya needs ~1.5GB VRAM. 2.0GB headroom provides Laya + 0.5GB extra margin (Total KV margin: 1.7GB)
    if headroom >= 2.0:
        return {
            "tier": "gpu",
            "device": "cuda",
            "reason": f"VRAM 여유 {headroom:.1f}GB 확보 (Tier 1: 100% GPU 초고속 모드, 약 30ms)",
            "headroom_gb": headroom
        }
    else:
        return {
            "tier": "cpu",
            "device": "cpu",
            "reason": f"VRAM 잔여({headroom:.1f}GB) 부족으로 GPU OOM 방지 (Tier 2: 호스트 78GB RAM + CPU 모드 자동 전환, 0GB VRAM 소모)",
            "headroom_gb": headroom
        }


def print_detailed_environment(hw):
    """Prints beautiful colored environment box in terminal."""
    gpu = hw["gpu"]
    cpu = hw["cpu"]
    ram = hw["ram"]
    disk = hw["disk"]
    hf_info = get_hf_account_info()

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

    if hf_info["authenticated"]:
        pro_tag = " [PRO]" if hf_info.get("is_pro") else ""
        print(f"• HF 계정   : @{hf_info['username']}{pro_tag} (🟢 API 연동 완료 / 고속 CDN 가속 활성)")
    elif hf_info["configured"]:
        print(f"• HF 계정   : ⚠️ {hf_info['message']}")
    else:
        print(f"• HF 계정   : 미설정 (익명 모드 │ .env에 HF_TOKEN 설정 시 전용 CDN 가속 및 게이티드 모델 지원)")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")


def download_file(url, dest_path, expected_size_gb=0, model_id=None):
    """
    Downloads file with real-time speed, ETA, and progress display.
    Supports resuming interrupted downloads.
    Primary engine: high-speed curl with automatic retry, timeout recovery, and HTTP/2.
    Fallback engine: robust chunked urllib with automatic retry loop (up to 10 attempts).
    Uses Hugging Face API authentication (HF_TOKEN) if configured in .env for maximum bandwidth and access.
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

    creds = get_hf_credentials()
    hf_token = creds["token"]

    tmp_path = dest_path + ".download"
    initial_bytes = os.path.getsize(tmp_path) if os.path.exists(tmp_path) else 0

    print(f"\n🌐 Hugging Face CDN 연결 중: {url}")
    if hf_token:
        account_info = get_hf_account_info(hf_token)
        account_label = f"@{account_info['username']}" if account_info["authenticated"] else "HF_TOKEN 활성"
        print(f"🔑 Hugging Face API 인증 적용 ({account_label}) - 전용 CDN 가속 및 Rate-Limit 해제 활성화")
        # Pre-flight access check
        can_access, access_err, code = check_hf_model_access(url, token=hf_token)
        if not can_access:
            print(f"\n❌ {access_err}")
            if db and download_id:
                db.log_download_failed(download_id, access_err)
            return False
    else:
        print("ℹ️ 익명 다운로드 모드 (Hugging Face 토큰 미설정 - .env의 HF_TOKEN으로 고속 인증 활성화 가능)")

    if initial_bytes > 0:
        print(f"🔄 이어받기 시작: {initial_bytes / (1024**3):.2f} GB 부터 이어서 다운로드합니다.")

    start_time = time.time()

    # 1. Primary Engine: High-Speed Curl (HTTP/2, auto-reconnect, 30~50+ MB/s)
    if shutil.which("curl"):
        print(f"⚡ 고속 가속 다운로더(curl) 가동 (최대 10회 자동 재연결/재시도 보장)...")
        cmd = [
            "curl",
            "-f", "-L",
            "-C", "-",
            "--retry", "10",
            "--retry-delay", "3",
            "--retry-max-time", "0",
            "--retry-all-errors",
            "--connect-timeout", "30",
            "--speed-time", "60",
            "--speed-limit", "1024",
            "-H", "User-Agent: localLLMService/1.0 (huggingface-api)",
            "--progress-bar",
            "-o", tmp_path,
            url
        ]
        if hf_token:
            cmd.insert(12, "-H")
            cmd.insert(13, f"Authorization: Bearer {hf_token}")

        try:
            res = subprocess.run(cmd)
            cur_size = os.path.getsize(tmp_path) if os.path.exists(tmp_path) else 0
            # Success code 0, or 33 (range not satisfiable if file already complete)
            if res.returncode == 0 or (res.returncode == 33 and cur_size >= int(expected_size_gb * (1024**3) * 0.95)):
                duration_sec = max(1, int(time.time() - start_time))
                downloaded_now = max(0, cur_size - initial_bytes)
                speed_mb = (downloaded_now / (duration_sec * 1024 * 1024)) if downloaded_now > 0 else 0
                print("\n✨ 다운로드 완료! 파일 검증 및 저장 중...")
                shutil.move(tmp_path, dest_path)
                print(f"✅ 저장 완료: {dest_path}")
                if db and download_id:
                    db.log_download_complete(download_id, cur_size, speed_mb, duration_sec)
                return True
            else:
                print(f"\n⚠️ curl 종료 코드 ({res.returncode}), 표준 스트리밍 다운로더로 전환하여 재시도합니다...")
        except Exception as e:
            print(f"\n⚠️ curl 실행 중 오류 ({e}), 표준 스트리밍 다운로더로 전환합니다...")

    # 2. Resilient Python urllib loop with automatic retry and resume
    max_retries = 10
    timeout_sec = 60
    chunk_size = 1024 * 1024  # 1 MB chunk

    for attempt in range(1, max_retries + 1):
        cur_initial = os.path.getsize(tmp_path) if os.path.exists(tmp_path) else 0
        req = urllib.request.Request(url, headers={
            "User-Agent": "localLLMService/1.0 (huggingface-api)"
        })
        if hf_token:
            req.add_header("Authorization", f"Bearer {hf_token}")
        if cur_initial > 0:
            req.add_header("Range", f"bytes={cur_initial}-")

        try:
            with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
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
                        total_bytes = int(cl) + cur_initial

                mode = "ab" if cur_initial > 0 else "wb"
                downloaded = cur_initial
                last_print = 0
                total_display_gb = (total_bytes / (1024**3)) if total_bytes else expected_size_gb

                print(f"📦 다운로드 대상: {os.path.basename(dest_path)} (총 {total_display_gb:.2f} GB)")

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

                duration_sec = max(1, int(time.time() - start_time))
                speed_mb = (downloaded - initial_bytes) / (duration_sec * 1024 * 1024) if duration_sec > 0 else 0
                print("\n✨ 다운로드 완료! 파일 검증 및 저장 중...")
                shutil.move(tmp_path, dest_path)
                print(f"✅ 저장 완료: {dest_path}")

                if db and download_id:
                    db.log_download_complete(download_id, downloaded, speed_mb, duration_sec)
                return True

        except urllib.error.HTTPError as e:
            if e.code == 416 and cur_initial > 0:
                shutil.move(tmp_path, dest_path)
                print(f"✅ 파일이 이미 완전히 다운로드되어 있습니다: {dest_path}")
                if db and download_id:
                    db.log_download_complete(download_id, cur_initial, 0, 0)
                return True
            print(f"\n❌ 다운로드 HTTP 에러 ({e.code}): {e.reason}")
            if attempt == max_retries:
                if db and download_id:
                    db.log_download_failed(download_id, f"HTTP Error {e.code}: {e.reason}")
                return False
        except Exception as e:
            print(f"\n⚠️ 일시적 네트워크 지연/끊김 ({type(e).__name__}: {e}) - {attempt}/{max_retries} 재시도 준비 중...")
            if attempt == max_retries:
                print(f"\n❌ 최대 재시도 횟수 초과로 다운로드 실패: {e}")
                if db and download_id:
                    db.log_download_failed(download_id, str(e))
                return False
            time.sleep(3)

    return False


def main():
    if len(sys.argv) < 2:
        print("Usage: hf_recommender.py [env|env-detail|env-summary|recommend|whiptail-menu|get-model|download|hf-status|hf-check|scan-models|scan-summary]")
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
        hf_info = get_hf_account_info()
        gpu_txt = f"{hw['gpu']['name']} ({hw['gpu']['total_vram_gb']}GB VRAM / 가용 {hw['gpu']['free_vram_gb']}GB)" if hw['gpu']['count'] > 0 else "CPU Only"
        ram_txt = f"{hw['ram']['total_gb']}GB (가용 {hw['ram']['avail_gb']}GB)"
        cpu_txt = f"{hw['cpu']['model'].split('@')[0].strip()} ({hw['cpu']['cores']}코어)"
        disk_txt = f"{hw['disk']['free_gb']}GB"
        hf_tag = f"🤗 HF: @{hf_info['username']}" if hf_info["authenticated"] else "🤗 HF: 익명"
        print(f"🖥️ GPU: {gpu_txt} │ RAM: {ram_txt}\n💻 CPU: {cpu_txt} │ 💾 저장소 여유: {disk_txt} │ {hf_tag}")

    # 4. Recommend models (JSON)
    elif cmd == "recommend":
        if DBManager:
            db = DBManager()
            db.sync_catalog(LOCAL_MODELS + discover_local_models(models_dir) + UNSLOTH_CATALOG)
        hw = detect_hardware()
        top10 = score_and_rank_models(hw, models_dir)
        print(json.dumps(top10, ensure_ascii=False, indent=2))

    # 5. Formatted menu items for whiptail
    elif cmd == "whiptail-menu":
        if DBManager:
            db = DBManager()
            db.sync_catalog(LOCAL_MODELS + discover_local_models(models_dir) + UNSLOTH_CATALOG)
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
            if model.get("repo", "").startswith("local/"):
                if model.get("is_downloaded", False):
                    print(f"✅ 로컬 모델 파일이 이미 준비되어 있습니다: {model['filename']}")
                    sys.exit(0)
                else:
                    print(f"❌ 로컬 모델 파일({model['filename']})을 찾을 수 없습니다. {models_dir} 경로에 배치해주세요.")
                    sys.exit(1)
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

    # 8. Hugging Face account status (JSON)
    elif cmd == "hf-status":
        info = get_hf_account_info()
        print(json.dumps(info, ensure_ascii=False, indent=2))

    # 9. Hugging Face account check (Console text)
    elif cmd == "hf-check":
        info = get_hf_account_info()
        print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print("🤗 Hugging Face API 연동 상태 점검:")
        print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        env_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
        print(f"• .env 파일 위치 : {env_file}")
        creds = get_hf_credentials()
        tok = creds["token"]
        if tok:
            masked = tok[:6] + "*" * (len(tok) - 10) + tok[-4:] if len(tok) > 10 else "******"
            print(f"• 토큰 설정 여부 : 설정됨 ({masked})")
        else:
            print(f"• 토큰 설정 여부 : ⚠️ 미설정 (익명 모드)")

        if info["authenticated"]:
            pro_str = " (PRO 유료 플랜)" if info.get("is_pro") else " (무료 플랜)"
            print(f"• 계정 인증 상태 : 🟢 인증 성공 (@{info['username']})")
            if info.get("fullname"):
                print(f"• 사용자 이름   : {info['fullname']}")
            if info.get("email"):
                print(f"• 등록된 이메일 : {info['email']}")
            print(f"• 구독 플랜     : {info['plan']}{pro_str}")
            print(f"• 다운로드 혜택 : 전용 고속 CDN 가속 및 Gated 모델 접근 지원")
        elif info["configured"]:
            print(f"• 계정 인증 상태 : ❌ {info['message']}")
            print(f"• 조치 안내     : .env 파일의 HF_TOKEN 값을 올바른 토큰으로 수정해주세요.")
            print(f"• 토큰 발급 링크 : https://huggingface.co/settings/tokens")
        else:
            print(f"• 계정 인증 상태 : ℹ️ 익명 다운로드 모드")
            print(f"• 설정 방법     : .env 파일에 아래 설정을 입력하시면 고속 인증이 활성화됩니다:")
            print(f"                   HF_TOKEN=hf_your_access_token_here")
            print(f"                   HF_USERNAME=your_username")
            print(f"• 토큰 발급 링크 : https://huggingface.co/settings/tokens (Read 권한 권장)")
        print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

    # 10. Scan local models directory (JSON)
    elif cmd == "scan-models":
        hw = detect_hardware()
        discovered = discover_local_models(models_dir)
        all_local = get_all_local_models(models_dir, hw)
        if DBManager:
            db = DBManager()
            db.sync_catalog(LOCAL_MODELS + discovered + UNSLOTH_CATALOG)
        print(json.dumps({
            "models_dir": models_dir,
            "total_local_files": len(all_local),
            "newly_discovered_count": len(discovered),
            "local_models": all_local
        }, ensure_ascii=False, indent=2))

    # 11. Scan local models directory summary (Console text)
    elif cmd == "scan-summary":
        hw = detect_hardware()
        discovered = discover_local_models(models_dir)
        all_local = get_all_local_models(models_dir, hw)
        if DBManager:
            db = DBManager()
            db.sync_catalog(LOCAL_MODELS + discovered + UNSLOTH_CATALOG)
        print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print("🔄 로컬 모델 저장소 스캔 결과 (Reload Summary):")
        print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print(f"• 저장소 경로 : {models_dir}")
        print(f"• 총 로컬 모델 : {len(all_local)}개 (자동 감지된 커스텀/Unsloth: {len(discovered)}개)")
        print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print("【현재 보유 중인 로컬 GGUF 모델 목록】")
        for idx, m in enumerate(all_local, 1):
            tag = m.get("offload_tag", "100% GPU")
            cat = m.get("category", "")
            print(f"  {idx:2d}. {m['name']} ({m['size_gb']}GB) [{tag}]")
            print(f"      파일명: {m['filename']} │ 분류: {cat}")
        print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

    # 12. Determine Laya Smart Tier
    elif cmd == "laya-tier":
        if len(sys.argv) < 3:
            print("Usage: hf_recommender.py laya-tier <1-based index or model-id>")
            sys.exit(1)
        target = sys.argv[2]
        hw = detect_hardware()
        top10 = score_and_rank_models(hw, models_dir)
        target_model = None
        if target.isdigit():
            idx = int(target) - 1
            if 0 <= idx < len(top10):
                target_model = top10[idx]
        else:
            for m in top10:
                if m.get("id") == target:
                    target_model = m
                    break
        if not target_model:
            target_model = {"recommended_vram_gb": 6.5, "name": "Default Model"}
        tier_info = determine_laya_tier(target_model, hw)
        print(json.dumps(tier_info, ensure_ascii=False))
        sys.exit(0)

    # 13. Embedding recommendations (JSON)
    elif cmd == "embedding-list":
        hw = detect_hardware()
        recs = get_embedding_recommendations(hw)
        print(json.dumps({
            "free_vram_gb": hw["gpu"]["free_vram_gb"],
            "total_vram_gb": hw["gpu"]["total_vram_gb"],
            "models": recs
        }, ensure_ascii=False, indent=2))
        sys.exit(0)

    # 14. Embedding whiptail menu items (JSON list)
    elif cmd == "embedding-menu":
        hw = detect_hardware()
        recs = get_embedding_recommendations(hw)
        menu_items = format_whiptail_embedding_menu_items(recs)
        print(json.dumps(menu_items, ensure_ascii=False))
        sys.exit(0)

    # 15. Get embedding model info by index or repo/id
    elif cmd == "embedding-get":
        if len(sys.argv) < 3:
            print("Usage: hf_recommender.py embedding-get <1-based index or repo/id>")
            sys.exit(1)
        target = sys.argv[2]
        hw = detect_hardware()
        recs = get_embedding_recommendations(hw)
        selected = None
        if target.isdigit():
            idx = int(target) - 1
            if 0 <= idx < len(recs):
                selected = recs[idx]
        else:
            for r in recs:
                if r["id"] == target or r["repo"] == target or r["name"] == target:
                    selected = r
                    break
        if selected:
            print(json.dumps(selected, ensure_ascii=False))
            sys.exit(0)
        else:
            print(f"Error: Unknown embedding model '{target}'", file=sys.stderr)
            sys.exit(1)

    # 16. Console summary of Korean embedding recommendations
    elif cmd == "embedding-summary":
        hw = detect_hardware()
        print_embedding_summary(hw)
        sys.exit(0)


if __name__ == "__main__":
    main()
