# 🦙 Local LLM Service - LLaMA.cpp + liteLLM + SGLang

**NVIDIA GB10 Grace Blackwell 워크스테이션(2EA)** 최적화 버전입니다. 대화형 쉘 스크립트를 통한 로컬 LLM 서비스 관리 도구로, **whiptail TUI** 메뉴로 모델을 선택하고, **liteLLM 프록시**를 선택적으로 추가하여 시맨틱 캐싱, 로깅, Web UI 대시보드 등의 기능을 활용할 수 있습니다.

## 🌟 주요 기능

### 🎯 대화형 모델 선택
- **whiptail TUI 메뉴**로 직관적인 모델 선택
- 모델 파일 자동 감지 (파일 존재 여부, 크기 표시)
- 현재 컨테이너 상태 실시간 표시

### 🚀 다양한 LLM 모델 지원 (Grace Blackwell 최적화)
| 모델 | 크기 | 컨텍스트 | 특징 |
|------|------|---------|------|
| **GLM-5.2** (권장) | 5.2B | 128K | 빠른 응답, BF16/FP8 지원 |
| **Gemma 4 E4B** | 4B/8B | 131K | 초경량, 풀 GPU 실행 |
| **Gemma 4 31B** | 31B | 256K | 고성능, 64 GPU 레이어 |
| **Gemma 4 26B A4B** | 26B | 256K | MoE 혼합, GPU 최적화 |
| **Qwen 3.6 35B A3B** | 35B | 256K | MoE 고급, Sparse 처리 |
| **Gemma 2 9B** | 9B | 8K | Dense 경량 |
| **Qwen 2.5 7B** | 7B | 131K | Dense 컴팩트 |

**LLaMA.cpp와 SGLang 두 가지 추론 엔진 지원**

### 💎 선택적 liteLLM 프록시 레이어
```
클라이언트 → liteLLM (:4000) → llama.cpp (:8080)
```

**liteLLM의 이점**:
- ✅ **클라이언트 설정 고정**: 모델 변경해도 항상 `:4000`으로 연결
- ✅ **시맨틱 캐싱**: 유사한 질문에 즉시 응답
- ✅ **Web UI 대시보드**: `http://localhost:4000/ui`에서 요청 모니터링
- ✅ **요청 로깅**: 모든 요청/응답 자동 기록

### 🛡️ 완벽한 에러 처리
- Docker 오류 시 명확한 에러 메시지
- liteLLM 실패 시 llama.cpp는 계속 실행
- 모델 파일 미존재 시 에러 + 메뉴 재진입
- 사용자 취소 시 정상 종료

---

## 📋 요구사항

### 필수
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

#### LLaMA.cpp 모델 (`.gguf` 형식)
```bash
~/Programming/models/
├── GLM-5.2-it-BF16.gguf          # 권장 (빠름)
├── GLM-5.2-it-FP8.gguf           # 더 빠름
├── google_gemma-4-E4B-it-Q4_K_M.gguf
├── google_gemma-4-E4B-it-Q8_0.gguf
├── google_gemma-4-31B-it-Q4_K_M.gguf
├── google_gemma-4-26B-A4B-it-Q4_K_M.gguf
└── Qwen_Qwen3.6-35B-A3B-Q4_0.gguf
```

#### SGLang 모델 (자동 다운로드)
- `zai-org/GLM-5.2` (추천)
- `zai-org/GLM-5.2-Multi-Vision`
- `google/gemma-4-E4B-it` 등

> 💡 **팁**: 
> - LLaMA.cpp: [ollama.ai](https://ollama.ai) 또는 [huggingface.co](https://huggingface.co)에서 `.gguf` 형식 다운로드
> - SGLang: Hugging Face Hub에서 자동 다운로드 (첫 실행 시 ~30GB 소요 가능)
> - GLM-5.2: https://huggingface.co/zai-org/GLM-5.2

### 3️⃣ 스크립트 실행

#### **llama.cpp 기동**
```bash
./docker_run.sh
```

**메뉴 흐름**:
```
1. 모델 선택 메뉴 표시
   ┌─────────────────────────────────┐
   │ 🦙 LLaMA.cpp Model Launcher    │
   │ Status: llama-server: ⚫ NOT    │
   │                                 │
   │ [✓] Gemma 4 E4B     (7.5G)    │
   │ [✓] Gemma 4 31B       (19G)   │
   │ [✓] Gemma 4 26B A4B   (16G)   │
   │ [✓] Qwen 3.6 35B A3B  (19G)   │
   └─────────────────────────────────┘

2. 모델 선택

3. 실행 확인
   Model: Gemma 4 E4B
   Port:  http://localhost:8080
   [Yes] [No]

4. liteLLM 프록시 옵션
   "liteLLM 함께 시작?"
   [Start liteLLM] [Skip]

5. 컨테이너 기동
   - llama.cpp → :8080
   - liteLLM → :4000 (선택 시)

6. 로그 표시 (선택)
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
| **llama.cpp** | `ghcr.io/ggml-org/llama.cpp:server-cuda` | 8080 | LLM 추론 |
| **liteLLM** | `ghcr.io/berriai/litellm:main-latest` | 4000 | 프록시 & 캐싱 |

---

## 🏗️ 아키텍처 (Grace Blackwell 최적화)

### 아키텍처 다이어그램
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
    │  5-35B Models   │
    │ 128GB LPDDRX    │
    └─────────────────┘
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

```
✅ 2x NVIDIA GB10 Grace Blackwell Superchip
✅ 128GB LPDDRX Memory (고대역폭)
✅ 4TB SSD
✅ NVIDIA ConnectX7 (400Gbps 인터커넥트)

최적화 설정:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• 메모리: 120GB shm-size, 124GB 메모리 제한
• GPU: 64개 레이어 (전체 GPU에 로드)
• 컨텍스트: 256K 토큰
• 배치: n=16384
• TP: 2 (분산 처리)
```

---

## 🔧 설정 파일

### litellm_config.yaml (자동 생성)
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

**기능**:
- ✅ 시맨틱 캐싱: 유사한 요청 즉시 응답
- ✅ 요청 로깅: 모든 API 호출 기록
- ✅ 모델 라우팅: 자동으로 llama.cpp로 전달

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

| 사용 환경 | 권장 모델 | 이유 |
|---------|---------|------|
| **Grace Blackwell (128GB)** | GLM-5.2 / Gemma 4 31B | 최적화됨, 빠른 응답 |
| **빠른 응답** | GLM-5.2 BF16 | 5.2B, 낮은 지연시간 |
| **멀티모달** | GLM-5.2-Multi-Vision | 이미지/텍스트 처리 |
| **최고 품질** | Qwen 3.6 35B A3B | 복잡한 추론 |
| **경량 (CPU 제한)** | Gemma 4 E4B | 4B, 초경량 |
| **멀티클라이언트** | liteLLM 포함 | 캐싱으로 속도 향상 |

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

**Last Updated**: 2026-06-24  
**Version**: 2.0 (Grace Blackwell Optimization)  
**Author**: Claude Code  
**Repository**: https://github.com/tsis-mobile-technology/localLLMService

### 변경사항 (v2.0)
- ✨ **GLM-5.2 모델 추가** (권장)
- 🎯 **NVIDIA GB10 Grace Blackwell 최적화**
  - 128GB LPDDRX 메모리 풀 활용
  - 듀얼 Superchip 분산 처리 (TP=2)
  - ConnectX7 고대역폭 인터커넥트
  - 256K 토큰 컨텍스트 지원
  - TF32 정밀도 최적화
- 🚀 **SGLang 고성능 추론 엔진 지원**
- 📊 **확장된 모델 라이브러리** (7가지 모델)
