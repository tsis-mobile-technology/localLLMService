# 🦙 Local LLM Service - LLaMA.cpp + liteLLM

대화형 쉘 스크립트를 통한 로컬 LLM 서비스 관리 도구입니다. **whiptail TUI** 메뉴로 모델을 선택하고, **liteLLM 프록시**를 선택적으로 추가하여 시맨틱 캐싱, 로깅, Web UI 대시보드 등의 기능을 활용할 수 있습니다.

## 🌟 주요 기능

### 🎯 대화형 모델 선택
- **whiptail TUI 메뉴**로 직관적인 모델 선택
- 모델 파일 자동 감지 (파일 존재 여부, 크기 표시)
- 현재 컨테이너 상태 실시간 표시

### 🚀 4가지 LLM 모델 지원
| 모델 | 파일 | 크기 | 컨텍스트 | 특징 |
|------|------|------|---------|------|
| **Gemma 4 E4B** | google_gemma-4-E4B-it-Q8_0.gguf | 7.5G | 128K | Full GPU |
| **Gemma 4 31B** | google_gemma-4-31B-it-Q4_K_M.gguf | 19G | 18K | 22 GPU Layers |
| **Gemma 4 26B A4B** | google_gemma-4-26B-A4B-it-Q4_K_M.gguf | 16G | 70K | MoE CPU Offload |
| **Qwen 3.6 35B A3B** | Qwen_Qwen3.6-35B-A3B-Q4_0.gguf | 19G | 70K | MoE CPU Offload |

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
- 최소 16GB RAM (모델 로드용)
- 최소 50GB 디스크 (모델 파일)

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
모델 파일을 저장소 디렉토리에 배치합니다:
```bash
# 다음 4개 파일 중 필요한 것들을 다운로드
~/Programming/models/
├── google_gemma-4-E4B-it-Q8_0.gguf
├── google_gemma-4-31B-it-Q4_K_M.gguf
├── google_gemma-4-26B-A4B-it-Q4_K_M.gguf
└── Qwen_Qwen3.6-35B-A3B-Q4_0.gguf
```

> 💡 **팁**: HuggingFace 또는 Ollama에서 `.gguf` 형식 모델 다운로드

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

## 📦 Docker 이미지

| 서비스 | 이미지 | 포트 | 용도 |
|--------|--------|------|------|
| **llama.cpp** | `ghcr.io/ggml-org/llama.cpp:server-cuda` | 8080 | LLM 추론 |
| **liteLLM** | `ghcr.io/berriai/litellm:main-latest` | 4000 | 프록시 & 캐싱 |

---

## 🏗️ 아키텍처

### 아키텍처 다이어그램
```
┌──────────────────────────────────────────────────────────┐
│                    Your Applications                      │
│         (Claude, Python SDK, Web App, etc.)             │
└──────────────────────┬───────────────────────────────────┘
                       │
                       ▼ (Optional with caching & logging)
            ┌─────────────────────┐
            │   liteLLM Proxy     │
            │   :4000             │
            │ ✨ Caching, UI      │
            └────────────┬────────┘
                         │
                         ▼ (OpenAI API compatible)
            ┌─────────────────────┐
            │  llama.cpp Server   │
            │  :8080              │
            │ 🚀 GPU Inference    │
            └────────────┬────────┘
                         │
                         ▼
            ┌─────────────────────┐
            │   LLM Models        │
            │ 📦 4 models (7.5-19G) │
            └─────────────────────┘
```

### 네트워킹
- **호스트 → llama.cpp**: `http://localhost:8080`
- **호스트 → liteLLM**: `http://localhost:4000`
- **liteLLM → llama.cpp**: `http://host.docker.internal:8080` (Docker 컨테이너 간)

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
| **제한된 VRAM** (< 12GB) | Gemma 4 E4B (7.5G) | 가장 작음, 빠름 |
| **중간 VRAM** (12-24GB) | Gemma 4 31B (19G) | 성능 ↑ |
| **고성능 서버** (> 24GB) | Qwen 3.6 35B A3B (19G) | 최고 품질 |
| **멀티클라이언트** | liteLLM 포함 | 캐싱으로 속도 향상 |

### 최적화 팁
1. **GPU 메모리**: `--gpus all` 사용으로 모든 GPU 활용
2. **캐싱**: liteLLM 활성화로 반복 질문에 즉시 응답
3. **컨텍스트**: 각 모델의 최적 컨텍스트 길이 사용
4. **캐시**: 정기적으로 liteLLM 캐시 초기화 (`localhost:4000/ui`에서)

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
curl http://localhost:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-1234" \
  -d '{
    "model": "local",
    "messages": [{"role": "user", "content": "안녕하세요"}],
    "temperature": 0.7,
    "max_tokens": 256
  }'
```

---

## 📋 파일 구조
```
localLLMService/
├── docker_run.sh          # 메인 스크립트 (모델 선택, liteLLM 옵션)
├── docker_stop.sh         # 컨테이너 정지 스크립트
├── .gitignore             # Git 제외 규칙 (*.gguf, venv/ 제외)
└── README.md              # 이 파일

~/Programming/models/
├── google_gemma-4-E4B-it-Q8_0.gguf
├── google_gemma-4-31B-it-Q4_K_M.gguf
├── google_gemma-4-26B-A4B-it-Q4_K_M.gguf
└── Qwen_Qwen3.6-35B-A3B-Q4_0.gguf
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

**Last Updated**: 2026-05-17  
**Author**: Claude Code  
**Repository**: https://github.com/tsis-mobile-technology/localLLMService
