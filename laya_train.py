#!/usr/bin/env python3
"""
laya_train.py
- Fine-tunes Laya System 1 Decision Engine on custom labelled data (CSV or JSONL).
- Leverages local-laya Docker container with PyTorch CUDA / CPU toolchain.
- Intelligent device selection: automatically switches to CPU if GPU VRAM is constrained by LLM server.
"""

import os
import sys
import argparse
import subprocess
import shutil
import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATA = BASE_DIR / "data" / "laya_sample_train.csv"
DEFAULT_OUT = BASE_DIR / "custom_laya_model"
DEFAULT_BASE_MODEL = "convaiinnovations/laya-multilingual"
LAYA_IMAGE = "local-laya:latest"


def check_gpu_free_vram_mb() -> int:
    """Returns free GPU VRAM in MiB using nvidia-smi, or 0 if unavailable."""
    try:
        res = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True
        )
        lines = [line.strip() for line in res.stdout.strip().splitlines() if line.strip()]
        if lines:
            return int(lines[0])
    except Exception:
        pass
    return 0


def validate_dataset(data_path: Path) -> tuple[bool, int, list[str], str]:
    """Inspects CSV / JSONL dataset format and returns basic stats."""
    if not data_path.exists():
        return False, 0, [], f"파일을 찾을 수 없습니다: {data_path}"

    if data_path.suffix.lower() == ".csv":
        import csv
        try:
            with open(data_path, "r", encoding="utf-8-sig") as f:
                reader = csv.DictReader(f)
                if not reader.fieldnames or "text" not in reader.fieldnames or "label" not in reader.fieldnames:
                    return False, 0, [], f"CSV 파일에 'text' 및 'label' 헤더 컬럼이 필요합니다 (현재: {reader.fieldnames})"
                rows = list(reader)
                labels = sorted(list(set(r.get("label", "").strip() for r in rows if r.get("label", "").strip())))
                if len(labels) < 2:
                    return False, len(rows), labels, f"최소 2개 이상의 고유 라벨이 필요합니다 (현재 발견된 라벨: {labels})"
                return True, len(rows), labels, "OK"
        except Exception as e:
            return False, 0, [], f"CSV 파일 파싱 실패: {e}"
    elif data_path.suffix.lower() == ".jsonl":
        try:
            count = 0
            labels = set()
            with open(data_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    count += 1
                    item = json.loads(line)
                    expected = item.get("expected", {})
                    for v in expected.values():
                        labels.add(str(v))
            return True, count, sorted(list(labels)), "OK"
        except Exception as e:
            return False, 0, [], f"JSONL 파일 파싱 실패: {e}"
    else:
        return False, 0, [], f"지원하지 않는 파일 형식입니다 (.csv 또는 .jsonl 지원): {data_path}"


def main():
    parser = argparse.ArgumentParser(
        description="Laya System 1 초경량 결정 엔진 로컬 파인튜닝 도구",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--data", type=str, default=str(DEFAULT_DATA), help="학습용 데이터셋 경로 (.csv 또는 .jsonl)")
    parser.add_argument("--eval", type=str, default=None, help="검증용 평가 데이터셋 경로 (.csv 또는 .jsonl)")
    parser.add_argument("--out", type=str, default=str(DEFAULT_OUT), help="학습 완료 체크포인트 저장 디렉토리")
    parser.add_argument("--base", type=str, default=DEFAULT_BASE_MODEL, help="기반 Laya 모델 ID 또는 로컬 경로")
    parser.add_argument("--epochs", type=int, default=4, help="학습 에포크 횟수")
    parser.add_argument("--micro-batch", type=int, default=8, help="마이크로 배치 크기")
    parser.add_argument("--grad-accum", type=int, default=8, help="그래디언트 누적 스텝")
    parser.add_argument("--encoder-lr", type=float, default=2.5e-5, help="인코더 학습률")
    parser.add_argument("--head-lr", type=float, default=1e-4, help="분류 헤드 학습률")
    parser.add_argument("--device", type=str, choices=["auto", "cuda", "cpu"], default="auto", help="연산 장치 (auto/cuda/cpu)")
    parser.add_argument("--freeze-encoder", action="store_true", help="인코더 고정 후 헤드만 초고속 학습")
    parser.add_argument("--dry-run", action="store_true", help="데이터 검증 및 옵티마이저 스케줄 계산만 수행")
    parser.add_argument("--local", action="store_true", help="Docker 대신 호스트 python/laya-train 직접 실행")

    args = parser.parse_args()

    data_path = Path(args.data).resolve()
    out_path = Path(args.out).resolve()
    eval_path = Path(args.eval).resolve() if args.eval else None

    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("🧠 Laya System 1 Decision Engine 파인튜닝 (Fine-Tuning)")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"• 학습 데이터셋 : {data_path}")
    print(f"• 출력 디렉토리 : {out_path}")
    print(f"• 기반 베이스   : {args.base}")
    print(f"• 에포크 수     : {args.epochs}")
    print(f"• 마이크로 배치 : {args.micro_batch} (누적 스텝: {args.grad_accum})")

    # 1. Dataset Pre-flight Validation
    ok, num_rows, labels, msg = validate_dataset(data_path)
    if not ok:
        print(f"\n❌ 데이터셋 검증 오류: {msg}")
        sys.exit(1)

    print(f"• 데이터 유효성 : 정상 ({num_rows}개 샘플, 고유 라벨 {len(labels)}개: {labels})")

    # 2. Device Selection (Auto-fallback to CPU if VRAM is low)
    selected_device = args.device
    if selected_device == "auto":
        free_vram = check_gpu_free_vram_mb()
        if free_vram >= 2500:
            selected_device = "cuda"
            print(f"⚡ 연산 가속 장치 : GPU (CUDA, 여유 VRAM: {free_vram} MiB)")
        else:
            selected_device = "cpu"
            print(f"⚡ 연산 가속 장치 : CPU (여유 VRAM {free_vram} MiB 부족으로 LLM OOM 방지를 위해 CPU 선택)")
    else:
        print(f"⚡ 연산 가속 장치 : {selected_device.upper()} (사용자 지정)")

    # 3. Dry-Run Flag Notice
    if args.dry_run:
        print("🔍 모드          : 드라이 런 (Dry Run - 가중치 변경 없음)")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")

    out_path.mkdir(parents=True, exist_ok=True)

    # 4. Execution Mode: Docker vs Local
    if args.local or not shutil.which("docker"):
        # Local execution using host python
        cmd = [
            "laya-train",
            "--data", str(data_path),
            "--out", str(out_path),
            "--base", args.base,
            "--epochs", str(args.epochs),
            "--micro-batch", str(args.micro_batch),
            "--grad-accum", str(args.grad_accum),
            "--encoder-lr", str(args.encoder_lr),
            "--head-lr", str(args.head_lr),
            "--device", selected_device
        ]
        if eval_path:
            cmd.extend(["--eval", str(eval_path)])
        if args.freeze-encoder:
            cmd.append("--freeze-encoder")
        if args.dry_run:
            cmd.append("--dry-run")

        print(f"🚀 로컬 laya-train 실행: {' '.join(cmd)}\n")
        res = subprocess.run(cmd)
        if res.returncode != 0:
            print(f"\n❌ 학습 중 오류 발생 (종료 코드 {res.returncode})", file=sys.stderr)
            sys.exit(res.returncode)
    else:
        # Docker Execution using local-laya:latest
        hf_cache = Path.home() / ".cache" / "huggingface"
        hf_cache.mkdir(parents=True, exist_ok=True)

        docker_cmd = ["docker", "run", "--rm", "-i"]
        if selected_device == "cuda":
            docker_cmd.extend(["--gpus", "all"])

        docker_cmd.extend([
            "--entrypoint", "laya-train",
            "-v", f"{data_path.parent}:/workspace/data:ro",
            "-v", f"{out_path}:/workspace/out",
            "-v", f"{hf_cache}:/root/.cache/huggingface",
            "-e", "HF_HOME=/root/.cache/huggingface",
            "-e", f"LAYA_DEVICE={selected_device}",
            "-e", "PYTHONUNBUFFERED=1",
            LAYA_IMAGE,
            "--data", f"/workspace/data/{data_path.name}",
            "--out", "/workspace/out",
            "--base", args.base,
            "--epochs", str(args.epochs),
            "--micro-batch", str(args.micro_batch),
            "--grad-accum", str(args.grad_accum),
            "--encoder-lr", str(args.encoder_lr),
            "--head-lr", str(args.head_lr),
            "--device", selected_device
        ])

        if eval_path:
            docker_cmd.extend([
                "-v", f"{eval_path.parent}:/workspace/eval:ro",
                "--eval", f"/workspace/eval/{eval_path.name}"
            ])
        if args.freeze_encoder:
            docker_cmd.append("--freeze-encoder")
        if args.dry_run:
            docker_cmd.append("--dry-run")

        print(f"🐳 Docker 컨테이너({LAYA_IMAGE})를 통한 파인튜닝 시작...")
        res = subprocess.run(docker_cmd)
        if res.returncode != 0:
            print(f"\n❌ Docker 학습 중 오류 발생 (종료 코드 {res.returncode})", file=sys.stderr)
            sys.exit(res.returncode)

    # 5. Result Verification
    if args.dry_run:
        print("\n✅ 드라이 런 검증이 완료되었습니다.")
        return

    safetensors_file = out_path / "model.safetensors"
    config_file = out_path / "rl_agent_config.json"
    if safetensors_file.exists() and config_file.exists():
        size_mb = safetensors_file.stat().st_size / (1024 * 1024)
        print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print("🎉 Laya 커스텀 모델 파인튜닝 완료!")
        print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print(f"• 저장 경로     : {out_path}")
        print(f"• 모델 크기     : {size_mb:.1f} MB (model.safetensors)")
        print(f"• 라벨 카테고리 : {labels}")
        print("\n[다음 배포 단계 안내]:")
        print("• Laya 엔진 컨테이너를 재시작하면 새로 학습된 모델이 자동 탑재됩니다.")
        print("  명령어: docker restart laya-engine")
        print("  또는:   ./docker_run.sh 를 재실행하세요.")
        print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    else:
        print(f"\n⚠️ 완료되었으나 체크포인트 파일이 온전하지 않습니다: {out_path}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
