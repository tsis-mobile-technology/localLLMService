# 🦙 Local LLM Service Platform - LLaMA.cpp + liteLLM + PostgreSQL

로컬 하드웨어(GPU VRAM, CPU, RAM, Disk)를 자동으로 감지하여 **Hugging Face GGUF 최적 모델(TOP 20)**을 추천하고, **PostgreSQL DB**를 통해 모델 카탈로그, 다운로드 이력, 서버 수명주기 로그, LiteLLM 다중 사용자 및 사용량 감사 로그를 영구 관리하는 로컬 LLM 통합 플랫폼입니다.

---

## 🌟 주요 기능

### 🖥️ 하드웨어 자동 감지 & 최신 모델 추천
- **실시간 하드웨어 프로파일링**: GPU 모델, VRAM 크기(가용량), CPU 코어수, 시스템 RAM, 디스크 여유 공간 자동 감지
- **하드웨어 최적화 GGUF TOP 20 추천**: VRAM 100% 오프로드 가능 여부 및 GPU+78GB RAM 하이브리드 적합도를 자동 계산하여 최적 모델 20종 선별
- **자동 다운로드 & 즉시 실행**: TUI 메뉴 또는 단축 번호 선택 시 Hugging Face에서 고속 스트리밍 다운로드 후 Docker 컨테이너 자동 구동

### 🧬 로컬 파인튜닝 & 커스텀 GGUF 자동 감지 (Auto-Discovery & Reload) (NEW!)
- **GGUF 바이너리 헤더 자동 파싱 (Zero-Dependency)**: `~/Projects/models/` 경로에 사용자가 직접 파인튜닝(Unsloth 등)하거나 다운로드한 `.gguf` 파일을 가져다 놓으면, 파이썬 내장 바이너리 파서가 `general.architecture`, `general.quantized_by`, `file_type`(양자화), `context_length`를 자동 추출하여 즉시 목록화
- **Unsloth 파인튜닝 모델 자동 식별**: Unsloth로 파인튜닝된 GGUF 파일(`Qwen2.5-1.5B-Instruct` 등)을 메타데이터 기반으로 자동 감지하여 `[LOCAL / Unsloth]` 태그 부여
- **원클릭 실시간 새로고침 (Reload)**: TUI 메뉴에서 `R` 키를 누르거나 `./docker_run.sh --reload` 실행 시 디렉토리를 즉각 재스캔하여 신규 파일을 메뉴에 즉시 반영
- **Gemma 4 E4B IT 파인튜닝 3단 비교 체계**:
  - `1. [LOCAL / 병합]`: 파인튜닝 가중치가 병합된 완제품 GGUF 단독 구동 (`gemma-4-E4B-it-ft-Q4_K_M.gguf`)
  - `2. [LOCAL / LoRA]`: 순정 베이스 모델에 LoRA 어댑터를 동시 결합 구동 (`gemma-4-E4B-it-base-Q4_K_M.gguf` + `gemma-4-E4B-it-lora-f16.gguf` via `--lora`)
  - `3. [LOCAL / 순정]`: 파인튜닝 전 순정 베이스 모델 구동 (`gemma-4-E4B-it-base-Q4_K_M.gguf`)
- **원클릭 A/B 비교 테스트**: 메뉴에서 번호 선택만으로 순정 vs LoRA 결합 vs 병합 완제품의 답변 품질 비교 가능

### 🤗 Hugging Face API 인증 & 초고속 안정 다운로드 (NEW!)
- **공식 API 및 고속 CDN 스트리밍**: Hugging Face API 토큰(`HF_TOKEN`)을 통한 전용 CDN 가속 및 익명 요청 제한(Rate-Limit / Connection Reset) 완전 해제
- **Gated Model(라이선스 동의 모델) 지원**: Meta Llama 3.3, Google Gemma 등 HF 라이선스 동의 모델도 인증 헤더를 통해 끊김 없이 다운로드
- **Pre-flight 권한 선제 검증**: 다운로드 개시 전 토큰 유효성(401), Gated 권한(403), 파일 존재 여부(404)를 사전 검증하여 안전성 확보
- **간편한 `.env` 설정 & 진단 CLI**: `.env`에 `HF_TOKEN` 입력 시 TUI 메뉴(`K`) 또는 `./docker_run.sh --hf-check`로 계정 등급 및 연동 상태 즉시 확인

### 🐘 PostgreSQL 기반 통합 데이터 관리 (Docker 동일 네트워크)
- **Hugging Face 모델 카탈로그 (`hf_model_catalog`)**: 추천 모델 메타데이터(양자화, 크기, 권장 VRAM, 파라미터 등) DB 자동 동기화
- **모델 다운로드 이력 (`model_downloads`)**: 다운로드 상태(`DOWNLOADING`/`COMPLETED`/`FAILED`), 다운로드 속도(MB/s), 소요 시간, 파일 크기 영구 보관
- **서버 런타임 수명주기 (`server_runtime_logs`)**: 컨테이너 시작/정상종료/에러 상태, GPU 오프로드 프로파일, 가동 시간(분) 추적 (LoRA 결합 상세 표기)
- **LiteLLM 사용자 및 사용량 감사 (`LiteLLM_UserTable`, `LiteLLM_SpendLogs`)**: 사용자별 API Key 발급 및 토큰 사용량(Prompt/Completion), 지연시간(Latency) DB 실시간 적재

### 🧠 Laya 초고속 결정 엔진 (System 1) & 3-Tier 자동 전환 (NEW!)
- **초경량 이중 의사결정 모델 (Fast Interceptor)**: LiteLLM `Pre-call Hook`을 통해 메인 LLM 호출 직전(~30ms) 보안 검문, 카테고리 분류, 긴급도 평가 수행
- **선제적 보안 가드레일 (Zero-Token Guardrail)**: 프롬프트 탈옥, 지침 무시 공격 감지 시 메인 LLM 토큰 낭비 없이 `HTTP 400`으로 즉각 차단
- **동적 프롬프트 & 파라미터 최적화**: 코딩(Temp 0.1), 고객 지원(Temp 0.3), 일반 대화(Temp 0.7)에 맞춰 시스템 프롬프트 및 온도를 실시간 자동 조정
- **하드웨어 인지형 스마트 3-Tier 자동 전환**:
  - **Tier 1 (GPU/CUDA)**: VRAM 여유 $\ge$ 2.5GB 확보 시 GPU 완전 가속 (~30ms)
  - **Tier 2 (CPU + 78GB RAM)**: VRAM 여유 부족 시 호스트 CPU + RAM으로 자동 오프로드 (0GB VRAM 소모로 메인 LLM OOM 원천 방지)
  - **Tier 3 (Bypass / Fail-Open)**: Laya 장애 발생 또는 `--no-laya` 지정 시 메인 LLM으로 안전 무중단 바이패스
- **한국어/다국어 & 영어 자동 라우팅**: 한글/비라틴 문자는 `laya-multilingual` (322M), 영어는 `laya` (305M)로 자동 라우팅 로드

### 🔍 한국어 특화 Hugging Face 임베딩(Embedding) 서빙 & 실시간 VRAM 적합도 감지 (NEW!)
- **한국어 SOTA 임베딩 5종 큐레이션**:
  - `BAAI/bge-m3` (560M / 1024차원 / ~1.5GB VRAM): Dense, Sparse, Multi-vector 하이브리드 지원, RAG 및 다국어/한국어 검색 표준 SOTA
  - `nlpai-lab/KURE-v1` (335M / 1024차원 / ~1.0GB VRAM): 고려대 연구실 개발 Ko-MTEB 한국어 벤치마크 1위권
  - `jhgan/ko-sroberta-multitask` (110M / 768차원 / ~400MB VRAM): VRAM 400MB 미만 초경량, 빠른 추론 속도
  - `dragonkue/BGE-m3-ko` (560M / 1024차원 / ~1.5GB VRAM): BGE-M3 한국어 집중 파인튜닝
  - `intfloat/multilingual-e5-small` (118M / 384차원 / ~350MB VRAM): 350MB 극저용량 및 안정적인 다국어/한국어 검색
- **Hugging Face TEI (Text Embeddings Inference) 초고속 컨테이너 연동**:
  - 포트 `8088`에서 OpenAI 호환 `/v1/embeddings` 엔드포인트 제공
  - 메인 LLM 구동 후 **실시간 잔여 VRAM(Free VRAM)**을 정밀 측정하여 `[100% GPU / 원활]`, `[GPU 주의 / 타이트]`, `[VRAM 부족 / CPU 권장]` 자동 태깅 및 대화형 선택
- **LiteLLM 프록시 자동 통합**:
  - `http://localhost:4000/v1/embeddings` 단일 게이트웨이로 메인 LLM 대화와 임베딩 벡터 생성을 동시 라우팅 (`model: text-embedding-3-small` 또는 원본 모델명)

### 💎 LiteLLM 프록시 & 다중 사용자 관리
- **OpenAI 호환 API**: `:4000/v1` 엔드포인트 제공 (`/v1/chat/completions`, `/v1/models` 등)
- **다중 사용자 및 API Key 발급**: 사용자별 독립된 API Key 및 월간 예산(`max_budget`) 한도 설정 지원
- **Web UI 대시보드**: `http://localhost:4000/ui`에서 실시간 모니터링
- **시맨틱 캐싱**: 동일/유사 질의 캐시를 통한 초고속 응답

---

## 🚀 빠른 시작 (Quick Start)

### 1️⃣ 환경 설정 및 초기 실행
```bash
# 저장소 클론 및 이동
git clone https://github.com/tsis-mobile-technology/localLLMService.git
cd localLLMService

# 환경 변수 템플릿 복사
cp .env.example .env

# Hugging Face 토큰 설정 (https://huggingface.co/settings/tokens 에서 Read 권한 발급)
# .env 파일에 아래 항목을 입력합니다:
#   HF_TOKEN=hf_your_token_here
#   HF_USERNAME=your_username

# Hugging Face API 인증 상태 점검
./docker_run.sh --hf-check

# 실행 권한 부여
chmod +x docker_run.sh docker_stop.sh docker_down.sh
```

### 2️⃣ 대화형 TUI 메뉴로 실행
```bash
./docker_run.sh
```
- 메뉴 최상단에 로컬 모델(1~3번) 및 하드웨어 최적화 추천 모델들이 표시됩니다.
- 서브 메뉴:
  - `R`: 모델 저장소 폴더 재스캔 (Reload) (NEW!)
  - `K`: Hugging Face API 토큰 및 계정 연동 상태 확인
  - `U`: LiteLLM 사용자 계정 및 API Key 목록 조회
  - `S`: LiteLLM 토큰 소비 및 요청 통계 조회
  - `H`: 모델 다운로드 및 서버 런타임 이력 조회
  - `B`: PostgreSQL DB 수동 백업 및 복원 관리

### 3️⃣ 단축 번호로 즉시 실행
```bash
./docker_run.sh 1    # [LOCAL / 병합] Gemma 4 E4B IT Finetuned 완제품 구동
./docker_run.sh 2    # [LOCAL / LoRA] Gemma 4 E4B IT Base + LoRA 결합 구동
./docker_run.sh 3    # [LOCAL / 순정] Gemma 4 E4B IT Base 순정 모델 구동
./docker_run.sh 4    # [LOCAL / Unsloth] Qwen2.5 1.5B Instruct Q4_K_M 자동 감지 모델 구동
./docker_run.sh 5    # [LOCAL / Unsloth] Qwen2.5 1.5B Instruct Q8_0 자동 감지 모델 구동
./docker_run.sh 6    # Qwen3.5 9B 등 추천 모델 구동
```

### 4️⃣ CLI 실시간 통계, Laya 상태, 백업 및 복구
```bash
# Laya 초고속 결정 엔진 모니터링 & 파인튜닝 (NEW!)
./docker_run.sh --laya-status # Laya 초고속 결정 엔진 상태 점검 (GPU/CPU 티어 및 헬스체크)
./docker_run.sh --laya-stats  # Laya 누적 판별 통계 (수행 건수, 차단율, 평균 지연시간, 평균 신뢰도, 카테고리 분포)
./docker_run.sh --laya-logs   # Laya 최근 판별 이력 및 라우팅 결과 실시간 조회 (기본 20건)
./docker_run.sh --laya-export data/laya_train.csv # DB에 적재된 판별 로그를 파인튜닝용 데이터셋으로 내보내기
./docker_run.sh --train-laya  # Laya 로컬 파인튜닝 실행 (샘플 또는 추출 데이터셋 기반 초경량 학습)
./docker_run.sh --no-laya 1   # Laya 결정 엔진을 구동하지 않고 LLM 단독 실행 (Tier 3)

# 한국어 임베딩(TEI) 모델 관리 & 테스트 (NEW!)
./docker_run.sh --embedding-list   # 한국어 임베딩 모델 목록 및 현재 VRAM 적합도 확인
./docker_run.sh --embedding-status # TEI 임베딩 컨테이너 가동 상태 점검
./test_curl.sh --embedding "한국어 문서 검색 및 임베딩 벡터 생성 테스트" # 임베딩 벡터 생성 테스트

# 플랫폼 통합 관리
./docker_run.sh --reload      # 모델 저장소(~/Projects/models) 즉시 재스캔 및 감지 모델 요약
./docker_run.sh --hf-check    # Hugging Face API 연동 상태 및 계정 확인
./docker_run.sh --stats       # LiteLLM 토큰 소비량 및 평균 레이턴시 통계
./docker_run.sh --users       # 등록된 사용자 및 활성 API Key 목록
./docker_run.sh --history     # 모델 다운로드 및 서버 구동/종료 이력
./docker_run.sh --backup      # PostgreSQL DB 즉시 백업 (backups/ 저장)
./docker_run.sh --restore     # 최신 백업본으로부터 DB 데이터 복원
./docker_run.sh --env         # 감지된 PC 하드웨어 환경 상세
./docker_run.sh --list        # 보유 모델 및 추천 모델 전체 목록 출력

# API 테스트 쿼리 전송 (3가지 모드 지원)
./test_curl.sh "파이썬 소수 판별 함수 작성해줘"           # 1. 기본: LiteLLM(:4000) + Laya 결정 엔진 가동
./test_curl.sh --no-laya "파이썬 소수 판별 함수 작성해줘"  # 2. 프록시 경유하되 Laya만 우회 (skip_laya=True)
./test_curl.sh --direct "파이썬 소수 판별 함수 작성해줘"   # 3. LLaMA.cpp 백엔드(:8080) 직접 호출 (순수 엔진 테스트)
```

### 5️⃣ 서비스 정상 종료 및 자동 백업/복원
```bash
# 1) LLaMA.cpp 및 LiteLLM 종료 (DB는 상시 유지되며, 최신 데이터 자동 백업됨)
./docker_stop.sh

# 2) DB까지 완전히 종료하고 싶을 때 (종료 전 DB 자동 백업 -> 컨테이너 종료)
./docker_down.sh
# 또는: ./docker_stop.sh --all

# 3) 다시 시작할 때 (DB 볼륨이 삭제되었거나 초기화되어도 백업된 데이터 자동 복구)
./docker_run.sh
# 또는: docker compose up -d
```


---
- **Docker** (with NVIDIA GPU support)
- **Docker Compose** (선택사항)
- **whiptail** (TUI 메뉴용)
- **bash** (v4.0+)

### 권장
- NVIDIA GPU (llama.cpp 가속)
- **Grace Blackwell 워크스테이션**: 128GB LPDDRX, 2x Superchip, ConnectX7 (최적화됨)
- 최소 16GB RAM (모델 로드용)
- 최소 50GB 디스크 (모델 파일)

### Grace Blackwell 최적화 기능
✅ 128GB LPDDRX 메모리 풀 활용  
✅ 듀얼 Superchip 분산 처리 (TP=2 지원)  
✅ ConnectX7 고대역폭 네트워킹 자동 인식  
✅ TF32 정밀도 최적화  
✅ 대용량 컨텍스트 지원 (256K tokens)  
✅ 배치 처리 최적화 (n=16384)

### 확인 방법
```bash
# Docker 확인
docker --version
docker ps

# whiptail 확인
which whiptail

# bash 확인
bash --version
```

---

## 🚀 설치 및 사용

### 1️⃣ 저장소 클론
```bash
git clone https://github.com/tsis-mobile-technology/localLLMService.git
cd localLLMService
chmod +x docker_run.sh docker_stop.sh
```

### 2️⃣ 모델 파일 준비

#### LLaMA.cpp 모델 (HuggingFace Hub에서 자동 로드)
```bash
# 모델은 스크립트 실행 시 자동으로 HuggingFace에서 다운로드됨
# 또는 로컬에 사전 다운로드:

~/Programming/models/
├── # GLM-5.2 (HuggingFace Hub에서 자동로드)
│   # unsloth/GLM-5.2-GGUF:UD-IQ1_S (1-bit, 223GB)
│   # unsloth/GLM-5.2-GGUF:UD-IQ2_M (2-bit, 245GB)
│   # unsloth/GLM-5.2-GGUF:UD-IQ3_XXS (3-bit, 110GB)
│
├── # Gemma 4 (HuggingFace Hub에서 자동로드)
│   # unsloth/gemma-4-31b-it-GGUF:UD-Q4_K_XL (31B, 20GB)
│   # google_gemma-4-E4B-it-Q4_K_M.gguf (4B)
│   # google_gemma-4-E4B-it-Q8_0.gguf (8B)
│   # google_gemma-4-26B-A4B-it-Q4_K_M.gguf (26B MoE)
│
└── # Qwen (로컬 또는 HuggingFace)
    # Qwen_Qwen3.6-35B-A3B-Q4_0.gguf (35B MoE)
```

**LLAMA_CACHE 환경변수**:
- 첫 실행: HuggingFace에서 모델 다운로드 → `~/Programming/models/` 저장
- 2회차+: 캐시된 모델 사용 (인터넷 불필요)

#### SGLang 모델 (자동 다운로드)
- `zai-org/GLM-5.2` (추천)
- `zai-org/GLM-5.2-Multi-Vision`
- `google/gemma-4-E4B-it` 등

> 💡 **팁**: 
> - **GLM-5.2 GGUF 모델**: https://huggingface.co/unsloth/GLM-5.2-GGUF (권장)
>   - 다양한 양자화 옵션 (Q8_0, BF16, IQ3_XXS 등)
>   - 분할 파일 자동 로드 지원
> - **LLaMA.cpp**: [ollama.ai](https://ollama.ai) 또는 [huggingface.co](https://huggingface.co)에서 `.gguf` 형식 다운로드
> - **SGLang**: Hugging Face Hub에서 자동 다운로드 (첫 실행 시 ~30GB 소요 가능)
> - **GLM-5.2 원본**: https://huggingface.co/zai-org/GLM-5.2

### 3️⃣ 스크립트 실행

#### **llama.cpp 기동**
```bash
./docker_run.sh
```

**메뉴 흐름**:
```
1. 모델 선택 메뉴 표시
   ┌──────────────────────────────────────────┐
   │ 🦙 LLaMA.cpp Model Launcher              │
   │ Status: llama.cpp: ⚫ NOT | liteLLM: ⚫  │
   │ Hardware: 🟢 HIGH (256GB) | GPU: 2x128GB│
   │                                          │
   │ [✓] GLM-5.2 IQ1_S (256GB Cluster)      │
   │ [✓] GLM-5.2 IQ2_M (Dual-Node)          │
   │ [✓] GLM-5.2 IQ3_XXS (128GB Single)     │
   │ [✓] Gemma 4 31B (Multi-Instance) 🆕    │
   │ [✓] Gemma 4 E4B (7.5G)                 │
   └──────────────────────────────────────────┘

2. 모델 선택

3. 모드 선택 (Gemma 4 31B의 경우)
   Model: Gemma 4 31B (Multi-Instance)
   Mode:  [Yes] Multi-Instance (4x + LiteLLM) ← 권장
          [No]  Single-Instance

4. 인스턴스 설정 (Multi-Instance 선택 시)
   ✅ 2 instances per workstation
   Ports: 8081, 8082 (WS1) + 8083, 8084 (WS2)

5. liteLLM 프록시 옵션
   "liteLLM으로 4개 인스턴스 통합?"
   [Start liteLLM] [Skip]

6. 컨테이너 기동
   - Instance A → :8081
   - Instance B → :8082
   - Instance C → :8083 (WS2)
   - Instance D → :8084 (WS2)
   - liteLLM → :4000 (라운드로빈 분배)

7. 로그 표시 (선택)
   [Tail Logs] [Exit]
```

#### **컨테이너 정지**
```bash
./docker_stop.sh
# ✓ All containers stopped and removed
```

---

## 🔌 API 사용

### llama.cpp 직접 사용 (포트 8080)

**Chat Completion**:
```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "local",
    "messages": [
      {"role": "user", "content": "Hello, who are you?"}
    ],
    "max_tokens": 100
  }'
```

**모델 목록**:
```bash
curl http://localhost:8080/v1/models
```

### liteLLM 프록시 사용 (포트 4000)

**Chat Completion** (동일한 OpenAI API):
```bash
curl http://localhost:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "local",
    "messages": [
      {"role": "user", "content": "Hello, who are you?"}
    ],
    "max_tokens": 100
  }'
```

**Health Check**:
```bash
curl http://localhost:4000/health
```

**Web UI 대시보드**:
```
http://localhost:4000/ui
Master Key: sk-local-master
```
- 📊 요청 로그 조회
- 💾 캐시 통계 및 관리
- 🔑 API 키 관리
- ⚙️ 모델 라우팅 설정

**API 엔드포인트**:
```bash
curl http://localhost:4000/v1/chat/completions \
  -H "Authorization: Bearer sk-local-master" \
  -H "Content-Type: application/json" \
  -d '{"model":"local","messages":[{"role":"user","content":"test"}]}'
```

**필수 조건**: PostgreSQL 데이터베이스 필요
- 현재 설정: `postgresql://user:password@host.docker.internal:5433/litellm`
- 자동으로 테이블 생성 및 마이그레이션 수행

---

## 🔒 HTTPS 프록시 가이드 (사전 준비 사항)

본 서비스의 4000번 포트(liteLLM) 또는 8080번 포트(llama.cpp) 앞단에 HTTPS 리버스 프록시를 배치하여 외부에서 안전한 암호화 통신으로 접근할 수 있습니다. 아래 3가지 솔루션의 특징과 구비 사항을 확인하여 나중에 필요에 맞게 구축해 보시기 바랍니다.

> [!NOTE]
> **인증서 비용**: 3가지 방식 모두 정식 SSL/HTTPS 인증서를 개별적으로 구매할 필요가 전혀 없으며, **무료**로 자동 발급 및 갱신해 줍니다.
> 
> **도메인 준비**: 외부 HTTPS 연결을 위해서는 인터넷 도메인 등록 대행업체(가비아, Porkbun, Cloudflare 등)를 통해 저렴한 도메인(예: `yourdomain.com`, 연간 약 2,000원 ~ 15,000원 선)을 하나 구매해 두셔야 합니다.

### 1. Cloudflare Tunnels (가장 강력 권장)
사내망이나 공유기 하단처럼 외부로 방화벽 포트(80, 443)를 개방하기 힘들 때 가장 안전하고 확실한 방식입니다.
* **준비물**: 도메인 구매 후, 해당 도메인의 네임서버를 **Cloudflare**로 지정 및 Zero Trust 서비스 가입.
* **방화벽**: 포트 개방 및 포트포워딩 **불필요**.
* **동작**: 서버의 `cloudflared` 데몬이 Cloudflare Edge와 아웃바운드로 터널을 형성하므로 방화벽 개방 없이 Cloudflare 망을 통해 무료 HTTPS 접속이 가능합니다.

### 2. Caddy
공인 IP가 존재하여 방화벽 포트를 직접 열 수 있고, 가장 간단하고 가벼운 리버스 프록시를 설정 파일(Caddyfile) 하나로 깔끔하게 구동하고 싶을 때 좋습니다.
* **준비물**: 도메인의 A 레코드를 서버의 **공인 IP**로 지정.
* **방화벽**: 외부에서 서버의 **80(HTTP), 443(HTTPS)** 포트로 접속할 수 있도록 포트 개방 필요.
* **동작**: Caddy가 실행되면서 Let's Encrypt를 통해 자동으로 SSL 인증서를 발급받고 기간 만료 전에 자동 갱신합니다.

### 3. Nginx Proxy Manager (NPM)
텍스트 설정 파일 조작 없이 웹 브라우저 관리 화면(GUI)을 보면서 마우스 클릭으로 간편하게 SSL 및 프록시 룰을 설정하고 싶을 때 적합합니다.
* **준비물**: 도메인의 A 레코드를 서버의 **공인 IP**로 지정.
* **방화벽**: 서버의 **80(HTTP), 443(HTTPS)** 포트와 **81(NPM 관리자 UI)** 포트 개방 필요.
* **동작**: 컨테이너 실행 후 웹 UI에서 도메인과 `host.docker.internal:4000`을 매핑하고, Let's Encrypt SSL 자동 발급 옵션을 클릭해서 켭니다.

---

## 📦 Docker 이미지

| 서비스 | 이미지 | 포트 | 용도 |
|--------|--------|------|------|
| **llama.cpp** | `ghcr.io/ggml-org/llama.cpp:server-cuda` | 8080 | System 2: 메인 고차원 추론 및 텍스트 생성 |
| **laya** | `local-laya:latest` (`python:3.11-slim`) | 8001 | System 1: 초고속 보안 검문, 분류, 긴급도 평가 |
| **liteLLM** | `ghcr.io/berriai/litellm:main-latest` | 4000 | OpenAI 호환 프록시, Laya 인터셉터, DB 로깅 |
| **postgres** | `postgres:16-alpine` | 5432 | 사용자, 토큰 감사 로그, 모델 카탈로그 영구 저장 |

---

## 🏗️ vLLM / LLaMA.cpp + LiteLLM + Laya 통합 아키텍처

```text
[Client / App] ──(OpenAI 호환 요청)──▶ [LiteLLM Proxy (:4000)]
                                              │
          ┌───────────────────────────────────┴───────────────────────────────────┐
          ▼ (1. Pre-call Hook: ~30-50ms)                                          ▼ (2. 동적 시스템 프롬프트 & 온도로 메인 추론)
[Laya Decision Engine (:8001)]                                          [LLaMA.cpp / vLLM Server (:8080)]
 - 한국어/다국어 자동 감지 (mmBERT 322M)                                 - PagedAttention / GPU 최적화 생성 모델
 - 가드레일 판별 (위험 시 HTTP 400 차단 응답 반환)                        - Laya가 주입한 최적 시스템 프롬프트 및
 - 의도 분류 (Coding / CS / General) 및 긴급도 산출                        온도(0.1~0.7)로 고품질 텍스트 생성 수행
```

### 🧠 스마트 3-Tier 하드웨어 적응형 자동 전환 체계
로컬 PC 하드웨어(GPU VRAM, CPU, 호스트 RAM) 상태와 사용자가 선택한 모델 크기를 실시간 계산하여 자동으로 최적 구동 모드를 결정합니다:

| 티어 (Tier) | 구동 환경 | VRAM 소모 | 조건 및 특성 |
|---|---|---|---|
| **Tier 1: GPU** | NVIDIA CUDA (RTX 3060) | ~1.5 GB | VRAM 여유 $\ge$ 2.0GB (소/중/준대형 모델: Gemma 4B, Qwen 1.5B/9B, DeepSeek 14B). 초고속 응답 (~30ms) |
| **Tier 2: CPU+RAM** | 12 vCPU + 78GB RAM | **0 GB (VRAM 보존)** | VRAM 여유 $<$ 2.0GB (초대형 모델: Qwen 27B, DeepSeek 32B 등). **메인 LLM의 GPU OOM 원천 방지** |
| **Tier 3: Bypass** | 비활성화 (Bypass) | 0 GB | `--no-laya` 옵션 또는 장애 발생 시 자동 Fail-Open 전환으로 메인 LLM 서비스 무중단 지속 |

### 🛠️ Laya 실시간 의사결정 고도화 및 파인튜닝 파이프라인 (Phase 1 ~ Phase 3)

본 프로젝트는 오픈소스 Laya 엔진의 성능을 극대화하기 위해 3단계 고도화 아키텍처를 탑재하였습니다:

1. **질문 스키마 및 가드레일 외부화 (`laya_questions.yaml`)**:
   - 코드 수정 없이 YAML 파일 수정만으로 Laya 질문 항목, 카테고리 정의, 분류별 LLM 파라미터(`temperature`, `system_prompt`), 보안 차단 임계값을 동적으로 핫 리로드합니다.

2. **PostgreSQL 감사 로깅 & 통계 추적 (`laya_decision_logs`)**:
   - 모든 Laya 판별 결과(보안 안전 여부, 카테고리, 신뢰도, 긴급도 점수, 판별 소요 시간, 바이패스 여부)를 백그라운드 비동기(Async)로 DB에 영구 기록합니다.
   - CLI 명령어 `./docker_run.sh --laya-stats` 및 `--laya-logs`로 실시간 분석이 가능합니다.

3. **오분류 데이터셋 추출 & 로컬 파인튜닝 (`laya_train.py`)**:
   - **데이터셋 추출**: `python3 db_manager.py laya-export dataset.csv` 명령어로 DB에 누적된 판별 로그를 CSV/JSONL 형태로 즉시 추출합니다.
   - **로컬 파인튜닝**: `python3 laya_train.py --data dataset.csv` (또는 `./docker_run.sh --train-laya`) 실행 시 Docker 컨테이너의 PyTorch 툴체인을 통해 가볍고 빠르게 파인튜닝을 수행합니다. (VRAM 여유가 부족할 경우 LLM 충돌을 방지하기 위해 CPU 연산으로 자동 안전 전환됩니다).
   - **커스텀 모델 자동 탑재**: 파인튜닝 완료 시 `custom_laya_model/`에 모델이 저장되며, Laya 엔진 시작 시 이를 감지하여 최우선 기본 모델로 즉시 서빙합니다.

---

## 🏗️ 아키텍처 (Grace Blackwell 최적화)

### 단일 인스턴스 아키텍처 (GLM-5.2 등)
```
┌──────────────────────────────────────────────────────────────┐
│                    Your Applications                          │
│         (Claude, Python SDK, Web App, etc.)                 │
└──────────────────────┬───────────────────────────────────────┘
                       │
        ┌──────────────┴───────────────┐
        ▼                              ▼
┌──────────────────┐        ┌──────────────────┐
│  liteLLM Proxy   │        │ SGLang Server    │
│    :4000         │        │    :30000        │
│ Caching, Logging │        │ Fast Inference   │
└────────┬─────────┘        └────────┬─────────┘
         │                           │
         └──────────────┬────────────┘
                        ▼
            ┌─────────────────────────┐
            │   LLaMA.cpp Server      │
            │      :8080              │
            │ 🚀 GPU Inference        │
            └────────────┬────────────┘
                         │
            ┌────────────┴────────────┐
            ▼                         ▼
    ┌─────────────────┐      ┌─────────────────┐
    │  GPU 0          │      │  GPU 1          │
    │ (Superchip)     │◄────►│ (Superchip)     │
    │ ConnectX7       │      │ ConnectX7       │
    └─────────────────┘      └─────────────────┘
            │
            ▼
    ┌─────────────────┐
    │   LLM Models    │
    │  744B Models    │
    │ 256GB LPDDRX    │
    └─────────────────┘
```

### Multi-Instance 아키텍처 (Gemma 4 31B - Data Parallelism) 🆕
```
클라이언트
    │
    ▼
┌──────────────────────────────────────────────────────────┐
│         LiteLLM Proxy (라운드로빈 + 부하 분배)            │
│              :4000                                        │
│         usage-based-routing-v2                           │
└──────────────────────────────────────────────────────────┘
    │
    ├──────────────┬──────────────┬──────────────┬─────────────┐
    ▼              ▼              ▼              ▼             ▼
┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐
│Instance │  │Instance │  │Instance │  │Instance │
│   A     │  │   B     │  │   C     │  │   D     │
│ :8081   │  │ :8082   │  │ :8083   │  │ :8084   │
└────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘
     │            │            │            │
     └────────────┴────────────┬┴────────────┘
                                │
     ┌──────────────────────────┼──────────────────────────┐
     │              ConnectX7 (400Gbps)                    │
     │          (WS1 ↔ WS2 고속 연결)                      │
     │                                                      │
  ┌──▼──────────────┐              ┌─────────────────▼───┐
  │ 워크스테이션 1   │              │  워크스테이션 2     │
  │   128GB         │              │     128GB          │
  │ Instance A (20GB) │            │ Instance C (20GB)   │
  │ Instance B (20GB) │            │ Instance D (20GB)   │
  │ Spare: 88GB     │              │ Spare: 88GB        │
  └─────────────────┘              └────────────────────┘

장점:
✅ 4명 동시 사용자 처리
✅ 각 인스턴스 독립 실행 (장애 격리)
✅ 각 워크스테이션 당 88GB KV 캐시 가용
✅ ConnectX7로 고속 네트워킹
```

### 네트워킹
- **호스트 → llama.cpp**: `http://localhost:8080`
- **호스트 → liteLLM**: `http://localhost:4000`
- **liteLLM → llama.cpp**: `http://host.docker.internal:8080` (Docker 컨테이너 간)

---

## 🎯 다중 하드웨어 환경 자동 최적화

### 🔄 자동 감지 및 최적화 (NEW!)

스크립트 실행 시 **GPU 메모리를 자동으로 감지**하고, 하드웨어에 맞게 파라미터를 자동 조정합니다:

```
┌─────────────────────────────────────────────────────────────┐
│ 하드웨어 감지                                               │
│ nvidia-smi로 GPU 메모리 확인                                │
│ free/sysctl로 시스템 메모리 확인                            │
└──────────────────┬──────────────────────────────────────────┘
                   ▼
    ┌──────────────────────────────────┐
    │ 프로파일 결정                     │
    ├──────────────────────────────────┤
    │ < 12GB   → LOW   (저사양)        │
    │ 12-24GB  → MEDIUM (중사양)       │
    │ > 24GB   → HIGH  (고사양)        │
    └──────────────────┬───────────────┘
                       ▼
    ┌──────────────────────────────────┐
    │ 파라미터 자동 조정                │
    │ • GPU 레이어 개수                │
    │ • 메모리 할당                   │
    │ • 배치 크기                     │
    │ • 컨텍스트 길이                 │
    └──────────────────────────────────┘
```

### 📊 하드웨어별 최적화 설정

| 항목 | LOW (<12GB) | MEDIUM (12-24GB) | HIGH (>24GB) |
|------|------------|-----------------|-------------|
| **권장 모델** | Gemma 4 E4B | Gemma 4 31B | Qwen 3.6 35B |
| **GPU 레이어** | 0-8 | 32 | 64 |
| **메모리** | 8GB shm | 16GB shm | 120GB shm |
| **배치 크기** | 512-1K | 4K-8K | 8K-16K |
| **컨텍스트** | 8K-32K | 65K-128K | 128K-256K |
| **캐시 정밀도** | Q4 | F16 | F16/BF16 |
| **CPU 오프로드** | 활성화 | 중간 | 최소 |

### Grace Blackwell 워크스테이션 (HIGH 프로파일)

#### 단일 128GB 워크스테이션
```
✅ NVIDIA GB10 Grace Blackwell Superchip (단일)
✅ 128GB LPDDRX Memory (고대역폭)
✅ 4TB SSD
✅ NVIDIA ConnectX7

최적화 설정:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• 권장 모델: GLM-5.2 IQ3_XXS (110GB, 단일 로드)
• 메모리: 100GB shm-size, 124GB 메모리 제한
• GPU: 64개 레이어
• 컨텍스트: 65K 토큰
```

#### 256GB 클러스터 (2 × 128GB 워크스테이션 + ConnectX7)
```
✅ 2x NVIDIA GB10 Grace Blackwell Superchip
✅ 256GB LPDDRX Memory 총합 (고대역폭 통합)
✅ NVIDIA ConnectX7 (400Gbps 인터커넥트)

최적화 설정 (GLM-5.2):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• 권장 모델: GLM-5.2 IQ1_S (223GB, 1-bit 극압축)
• 메모리/워크스테이션: 120GB shm-size, 124GB 제한
• 여유: 33GB (OS, 백그라운드, KV 캐시)
• 컨텍스트: 32-65K 토큰
• 배치: n=8192
• 특징: Tensor Parallelism 불필요 (단일 모델)

최적화 설정 (Gemma 4 31B - Multi-Instance):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• 인스턴스: 4개 (2개/워크스테이션)
• 메모리/인스턴스: 32GB shm-size, 40GB 제한
• 모델/인스턴스: 20GB 차용, 25GB KV 캐시
• 여유/워크스테이션: ~88GB (추가 캐싱, 컨텍스트)
• 컨텍스트: 45K 토큰
• 특징: Continuous Batching (--cb), 4개 병렬 요청
• 라우팅: LiteLLM usage-based-routing (부하 분산)
```

---

## 🔧 설정 파일

### litellm_config.yaml (자동 생성)

#### 단일 인스턴스 모드 (GLM-5.2 등)
```yaml
model_list:
  - model_name: "local"
    litellm_params:
      model: "openai/local"
      api_base: "http://host.docker.internal:8080"
      api_key: "sk-local"

litellm_settings:
  drop_params: true              # llama.cpp 미지원 파라미터 무시
  cache: true                    # 시맨틱 캐싱 활성화
  cache_params:
    type: "local"                # 인메모리 캐시 (Redis 불필요)
    supported_call_types: ["completion", "text_completion"]

general_settings:
  master_key: "sk-local-master"
```

#### 다중 인스턴스 모드 (Gemma 4 31B - Data Parallelism) 🆕
```yaml
model_list:
  # 워크스테이션 1
  - model_name: "gemma-4-31b"
    litellm_params:
      model: "openai/gemma-4-31b"
      api_base: "http://localhost:8081"
  
  - model_name: "gemma-4-31b"
    litellm_params:
      model: "openai/gemma-4-31b"
      api_base: "http://localhost:8082"
  
  # 워크스테이션 2 (네트워크 기반)
  - model_name: "gemma-4-31b"
    litellm_params:
      model: "openai/gemma-4-31b"
      api_base: "http://<WS2-IP>:8083"
  
  - model_name: "gemma-4-31b"
    litellm_params:
      model: "openai/gemma-4-31b"
      api_base: "http://<WS2-IP>:8084"

router_settings:
  routing_strategy: usage-based-routing-v2  # 부하 기반 분배 (라운드로빈보다 효율적)
  
litellm_settings:
  drop_params: true
  cache: true
  cache_params:
    type: "local"
    supported_call_types: ["completion", "text_completion"]

general_settings:
  master_key: "sk-local-master"
```

**기능**:
- ✅ 시맨틱 캐싱: 유사한 요청 즉시 응답
- ✅ 요청 로깅: 모든 API 호출 기록
- ✅ 모델 라우팅: 자동으로 llama.cpp로 전달
- ✅ (Multi-Instance) 부하 분산: 4개 인스턴스 중 가장 한가한 곳으로 요청 분배

---

## 🚨 문제 해결

### Q1: "Docker daemon is not running"
```bash
# Docker 시작
sudo systemctl start docker
# 또는 Docker Desktop 앱 실행
```

### Q2: "NVIDIA GPU not detected"
```bash
# NVIDIA GPU 지원 확인
nvidia-smi

# Docker에서 GPU 사용 가능 확인
docker run --rm --gpus all ubuntu nvidia-smi
```

### Q3: "Port 8080 is already in use"
```bash
# 기존 프로세스 확인
lsof -i :8080

# 프로세스 종료
kill -9 <PID>

# 또는 다른 포트 사용
# docker_run.sh에서 PORT=9090 으로 변경
```

### Q4: "Model file not found"
- 모델 파일이 `~/Programming/models/` 디렉토리에 있는지 확인
- 파일명이 정확한지 확인 (대소문자 구분)
- 충분한 디스크 공간 확인 (최소 50GB)

### Q5: "liteLLM container fails to start"
- Docker 이미지 재다운로드:
  ```bash
  docker pull ghcr.io/berriai/litellm:main-latest
  ```
- liteLLM은 선택사항이므로 "No"를 선택하고 llama.cpp만 사용 가능

### Q5-1: "liteLLM Web UI 접근"
- **필수**: PostgreSQL 데이터베이스 필요
- **기본 설정**: 기존 PostgreSQL 활용 (docker_run.sh에서 자동 설정)
  ```bash
  DATABASE_URL=postgresql://user:password@host.docker.internal:5433/litellm
  ```
- **접근 방법**:
  1. `./docker_run.sh` 실행
  2. 모델 선택
  3. liteLLM Y 선택
  4. 브라우저: `http://localhost:4000/ui`
  5. Master Key 입력: `sk-local-master`

- **PostgreSQL이 없는 경우**:
  - API 기능은 완전히 작동
  - Web UI 대신 API로 모든 기능 사용 가능:
    ```bash
    curl http://localhost:4000/v1/chat/completions \
      -H "Authorization: Bearer sk-local-master" \
      -H "Content-Type: application/json" \
      -d '{"model":"local","messages":[{"role":"user","content":"Hello"}]}'
    ```

### Q6: "whiptail command not found"
```bash
# whiptail 설치
sudo apt-get update
sudo apt-get install whiptail

# 또는 (다른 배포판)
sudo yum install newt
```

---

## 📊 성능 팁

### 모델 선택 가이드

| 사용 환경 | 권장 모델 | 메모리 | 이유 |
|---------|---------|--------|------|
| **256GB 클러스터 (최고 품질)** | GLM-5.2 IQ1_S | 223GB | 극압축 (1-bit), 33GB 여유, 깊이있는 추론 |
| **팀 협업 (4-10명)** | Gemma 4 31B Multi-Instance | 20GB/인스턴스 | 4개 인스턴스 + LiteLLM, 병렬 처리 |
| **장문서 코딩 분석** | DeepSeek-Coder-V2 | 60-65GB | 164K 컨텍스트 (Gemma의 3배), MoE 효율성, 3-4개 인스턴스 |
| **단일 128GB 워크스테이션** | GLM-5.2 IQ3_XXS | 110GB | 3-bit 압축, 충분한 여유, 균형잡힌 성능 |
| **빠른 응답** | GLM-5.2 IQ2_M | 245GB | 2-bit 압축, 높은 품질, 부하 분산 필요 |
| **멀티모달** | GLM-5.2-Multi-Vision | 변동 | 이미지/텍스트 처리 |
| **최고 추론 품질** | Qwen 3.6 35B A3B | 19GB | 복잡한 논리, 코딩 |
| **경량 (로컬 테스트)** | Gemma 4 E4B | 7-8GB | 4B, 초경량, 빠른 테스트 |
| **캐싱 성능** | liteLLM 포함 | 1GB 미만 | 시맨틱 캐싱, 부하 분산 |

### 최적화 팁
1. **자동 감지**: 스크립트가 실행 시 하드웨어를 자동으로 감지하고 최적화합니다
   - 낮은 사양 환경에서도 안정적으로 실행 가능
   - 고사양 환경에서는 최대 성능 활용
2. **GPU 메모리**: 자동으로 사용 가능한 GPU 메모리에 맞춰 설정
3. **캐싱**: liteLLM 활성화로 반복 질문에 즉시 응답
4. **컨텍스트**: 각 모델의 최적 컨텍스트 길이 사용 (자동 조정)
5. **캐시**: 정기적으로 liteLLM 캐시 초기화 (`localhost:4000/ui`에서)

---

## 📝 예제

### Python SDK 사용
```python
from openai import OpenAI

# liteLLM 프록시 사용 (권장)
client = OpenAI(
    api_key="sk-1234",
    base_url="http://localhost:4000/v1"
)

# llama.cpp 직접 사용
# client = OpenAI(
#     api_key="sk-1234",
#     base_url="http://localhost:8080/v1"
# )

response = client.chat.completions.create(
    model="local",
    messages=[
        {"role": "user", "content": "안녕하세요"}
    ]
)

print(response.choices[0].message.content)
```

### JavaScript/Node.js
```javascript
const OpenAI = require('openai');

const openai = new OpenAI({
    apiKey: 'sk-1234',
    baseURL: 'http://localhost:4000/v1',
});

const message = await openai.chat.completions.create({
    model: 'local',
    messages: [{ role: 'user', content: '안녕하세요' }],
});

console.log(message.choices[0].message.content);
```

### cURL
```bash
curl http://localhost:4000/v1/models \
  -H "Authorization: Bearer sk-local-master"
```
```bash
curl http://localhost:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-local-master" \
  -d '{
    "model": "gpt-4o",
    "messages": [{"role": "user", "content": "안녕하세요"}],
    "temperature": 0.7,
    "max_tokens": 256
  }'
```

---

## 📋 파일 구조
```
localLLMService/
├── docker_run.sh              # LLaMA.cpp 메인 스크립트 (Grace Blackwell 최적화)
├── docker_run_sglang.sh       # SGLang 고성능 추론 스크립트
├── docker_stop.sh             # 컨테이너 정지 스크립트
├── docker_stop_sglang.sh      # SGLang 컨테이너 정지 스크립트
├── clippable_linear.py        # SGLang 커스텀 레이어
├── weight_utils.py            # 가중치 로더 최적화
├── litellm_config_sglang.yaml # SGLang liteLLM 설정
├── .gitignore                 # Git 제외 규칙 (*.gguf, venv/ 제외)
└── README.md                  # 이 파일

~/Programming/models/
├── GLM-5.2-it-BF16.gguf           # 권장 모델
├── GLM-5.2-it-FP8.gguf
├── google_gemma-4-E4B-it-Q4_K_M.gguf
├── google_gemma-4-E4B-it-Q8_0.gguf
├── google_gemma-4-31B-it-Q4_K_M.gguf
├── google_gemma-4-26B-A4B-it-Q4_K_M.gguf
└── Qwen_Qwen3.6-35B-A3B-Q4_0.gguf

~/.cache/huggingface/hub/
└── models--zai-org--GLM-5.2/     # SGLang 자동 다운로드
```

---

## 🔐 보안

### API 키
- 로컬 사용: `sk-local` (개발용)
- 프로덕션: 환경 변수로 `LITELLM_MASTER_KEY` 설정

### 방화벽
- **로컬 전용**: 포트 8080, 4000을 localhost에만 바인딩
- **원격 접근**: `0.0.0.0`으로 바인딩 시 방화벽 규칙 추가 필요

### 모델 업데이트
- `.gitignore`가 `*.gguf` 파일을 자동 제외
- 모델 파일은 GitHub에 업로드되지 않음 (용량)

---

## 📞 지원

### 문제 해결
1. `docker ps` - 컨테이너 상태 확인
2. `docker logs llama-server` - llama.cpp 로그
3. `docker logs litellm-proxy` - liteLLM 로그
4. `./docker_stop.sh` - 모든 컨테이너 정지

### 로그 확인
```bash
# 실시간 로그
docker logs -f llama-server

# 최근 100줄
docker logs --tail 100 llama-server
```

---

## 📄 라이선스

이 프로젝트는 다음 오픈소스를 활용합니다:
- **llama.cpp**: MIT License
- **liteLLM**: ISC License
- **whiptail**: GPL License

---

## 🤝 기여

문제 보고 및 개선 제안은 GitHub Issues에서 해주세요.

---

**Last Updated**: 2026-09-27  
**Version**: 2.2 (Local Fine-tuned & LoRA Adapter Support)  
**Author**: Claude Code & Antigravity  
**Repository**: https://github.com/tsis-mobile-technology/localLLMService

### 변경사항 (v2.2) - 2026-09-27
- 🧬 **Gemma 4 E4B IT 로컬 파인튜닝 & LoRA 어댑터 결합 지원**
  - `[LOCAL / 병합]` Finetuned 모델 (`gemma-4-E4B-it-ft-Q4_K_M.gguf`) 완제품 단독 구동
  - `[LOCAL / LoRA]` 순정 베이스 + LoRA 어댑터 (`gemma-4-E4B-it-lora-f16.gguf`) 동시 결합 로드 (`--lora` 옵션 자동 바인딩)
  - `[LOCAL / 순정]` 순정 베이스 모델 (`gemma-4-E4B-it-base-Q4_K_M.gguf`) 비교 대조군 지원
- 🔍 **로컬 모델 자동 감지 & TUI 최우선순위(1~3번) 배치**
  - `~/Projects/models/` 내 파일 존재 여부 자동 확인 (LoRA의 경우 베이스+어댑터 동시 검증)
  - Whiptail 대화형 다이얼로그 및 구동 완료 안내 창에 LoRA 어댑터 결합 상세 정보 노출
  - CLI 단축 번호(`1`, `2`, `3`) 즉시 실행 지원 (다운로드 스킵 처리)
  - PostgreSQL 런타임 로그(`server_runtime_logs`)에 LoRA 결합 여부 상세 기록 (`(+LoRA: ...)`)

### 이전 변경사항 (v2.1)
- 🆕 **DeepSeek-Coder-V2 (164K Context Long-Document Coding)**
  - Q4_K_S 양자화 (60-65GB, 7개 파일)
  - 164K 컨텍스트 (Gemma 4 31B의 3배)
  - MoE 구조 (236B, 21B 활성) - 효율적 추론
  - 3-4개 인스턴스 + LiteLLM 라운드로빈
  - 전체 파일/프로젝트 분석, 긴 코드 리뷰 최적화

- 🆕 **Gemma 4 31B Multi-Instance Mode (Data Parallelism)**
  - 4개 독립 인스턴스 (2개/워크스테이션)
  - LiteLLM 라운드로빈 + usage-based 부하 분산
  - 팀 협업 (5-10명) 최적화
  - Continuous Batching (--cb) 지원
  - 45K 컨텍스트 윈도우 (8K-32K 프롬프트 대응)

- 🔄 **GLM-5.2 양자화 옵션 재정의**
  - IQ1_S (1-bit, 223GB) - 256GB 클러스터 권장
  - IQ2_M (2-bit, 245GB) - 고품질 압축
  - IQ3_XXS (3-bit, 110GB) - 단일 128GB 최적
  - ~~Q8_0, BF16~~ (구 버전, 메모리 비효율)

- 📊 **HuggingFace Hub 직접 로드**
  - 로컬 파일 + HuggingFace 자동 감지
  - LLAMA_CACHE 환경변수 자동 설정
  - 첫 실행: 자동 다운로드, 2회차+: 캐시 활용

- 🎯 **NVIDIA GB10 Grace Blackwell 최적화 (유지)**
  - 128GB LPDDRX 메모리 풀 활용
  - ConnectX7 고대역폭 인터커넥트 (400Gbps)
  - 256GB 클러스터 지원 (2 × 워크스테이션)
  - 256K 토큰 컨텍스트 지원

- 🚀 **SGLang 고성능 추론 엔진 지원 (유지)**
- 📋 **자동 하드웨어 감지** (LOW/MEDIUM/HIGH 프로파일)
- ✨ **지원 모델 수**: 8개 → **9개** (DeepSeek-Coder-V2 추가)
