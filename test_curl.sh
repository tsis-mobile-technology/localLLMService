#!/bin/bash
set -euo pipefail

# Default configuration
BASE_URL="http://localhost:4000/v1"
API_KEY="sk-local-master"
MODEL="gpt-4o"
QUERY=""
MODE="laya"
SKIP_LAYA=false

# 1. Parse command line arguments
while [ $# -gt 0 ]; do
  case "$1" in
    --no-laya|-n)
      MODE="no-laya"
      SKIP_LAYA=true
      shift
      ;;
    --direct|-d)
      MODE="direct"
      BASE_URL="http://localhost:8080/v1"
      API_KEY=""
      shift
      ;;
    --embedding|-e)
      MODE="embedding"
      MODEL="text-embedding-3-small"
      BASE_URL="http://localhost:4000/v1"
      shift
      ;;
    --help|-h)
      echo "사용법: $0 [옵션] [\"질문\"]"
      echo ""
      echo "옵션:"
      echo "  --no-laya,   -n    LiteLLM(:4000)을 경유하되 Laya 인터셉터만 건너뜀 (skip_laya=True)"
      echo "  --direct,    -d    LiteLLM과 Laya를 모두 건너뛰고 LLaMA.cpp 백엔드(:8080) 직접 호출"
      echo "  --embedding, -e    LiteLLM(:4000/v1/embeddings) 임베딩 벡터 생성 테스트"
      echo "  --help,      -h    도움말 출력"
      echo ""
      echo "예시:"
      echo "  $0 \"파이썬 소수 판별 함수 작성해줘\"           # 기본 (LiteLLM + Laya 결정 엔진)"
      echo "  $0 --no-laya \"파이썬 소수 판별 함수 작성해줘\"  # Laya 우회 모드 (LiteLLM 단독)"
      echo "  $0 --direct \"파이썬 소수 판별 함수 작성해줘\"   # LLaMA.cpp 백엔드 직접 호출"
      echo "  $0 --embedding \"한국어 문서 검색 및 임베딩\"     # 임베딩 모델 벡터 생성 테스트"
      exit 0
      ;;
    *)
      if [ -z "$QUERY" ]; then
        QUERY="$1"
      else
        QUERY="$QUERY $1"
      fi
      shift
      ;;
  esac
done

if [ -z "$QUERY" ]; then
  QUERY="안녕, 오늘 날짜를 유추해서 시를 작성해.."
fi

# Print banner based on mode
echo "=================================================="
if [ "$MODE" = "direct" ]; then
  echo "🚀 Testing API Endpoint (Direct LLaMA.cpp - Bypass LiteLLM & Laya)"
elif [ "$MODE" = "no-laya" ]; then
  echo "🚀 Testing API Endpoint (LiteLLM Proxy - Laya Bypassed)"
elif [ "$MODE" = "embedding" ]; then
  echo "🚀 Testing API Endpoint (LiteLLM Embeddings - TEI Server)"
else
  echo "🚀 Testing API Endpoint (with Laya Decision Interceptor)"
fi
echo " - URL:   $BASE_URL"
echo " - Mode:  $MODE"
echo " - Model: $MODEL"
echo " - Query: $QUERY"
echo "=================================================="

# 2. JSON payload build
START_TIME=$(date +%s.%N)

if [ "$MODE" = "embedding" ]; then
  PAYLOAD=$(python3 -c "
import json, sys
model = sys.argv[1]
query = sys.argv[2]
print(json.dumps({'model': model, 'input': query}))
" "$MODEL" "$QUERY")
  ENDPOINT="$BASE_URL/embeddings"
else
  PAYLOAD=$(python3 -c "
import json, sys
model = sys.argv[1]
query = sys.argv[2]
skip_laya = sys.argv[3].lower() == 'true'

data = {
    'model': model,
    'messages': [{'role': 'user', 'content': query}]
}
if skip_laya:
    data['metadata'] = {'skip_laya': True}

print(json.dumps(data))
" "$MODEL" "$QUERY" "$SKIP_LAYA")
  ENDPOINT="$BASE_URL/chat/completions"
fi

# Build headers
AUTH_HEADER=()
if [ -n "$API_KEY" ]; then
  AUTH_HEADER=(-H "Authorization: Bearer $API_KEY")
fi

RESPONSE_DATA=$(curl -s -w "\n%{http_code}" --location "$ENDPOINT" \
  -H "Content-Type: application/json" \
  ${AUTH_HEADER[@]+"${AUTH_HEADER[@]}"} \
  --data "$PAYLOAD")

END_TIME=$(date +%s.%N)
DURATION=$(echo "$END_TIME - $START_TIME" | bc 2>/dev/null || echo "N/A")

# Response parsing
HTTP_STATUS=$(echo "$RESPONSE_DATA" | tail -n 1)
HTTP_BODY=$(echo "$RESPONSE_DATA" | sed '$d')

# 3. Output visualization
echo ""
if [ "$HTTP_STATUS" -eq 200 ]; then
    echo "🟢 [SUCCESS] API Response Received (Status: $HTTP_STATUS)"
    echo "⏱️  Response Time: ${DURATION}s"
    echo "--------------------------------------------------"

    if [ "$MODE" = "embedding" ]; then
        echo "🔍 [Embedding Result]:"
        python3 -c "
import json, sys
try:
    d = json.loads(sys.argv[1])
    vec = d['data'][0]['embedding']
    print(f' - 벡터 차원 수 (Dimension): {len(vec)}')
    print(f' - 벡터 샘플 (First 5): {vec[:5]}...')
    total_tok = d.get('usage', {}).get('total_tokens', 'N/A')
    print(f' - 토큰 수 (Tokens): {total_tok}')
except Exception as e:
    print('Failed to parse embedding response:', e)
" "$HTTP_BODY"
    else
        echo "🤖 [Assistant Answer]:"
        echo "$HTTP_BODY" | jq -r '
          .choices[0].message | 
          if (.content != null and .content != "") then 
            .content 
          elif (.reasoning_content != null and .reasoning_content != "") then 
            .reasoning_content 
          else 
            "No response text" 
          end'

        echo "--------------------------------------------------"
        echo "📊 [Token Usage]:"
        echo "$HTTP_BODY" | jq -r '" - Prompt Tokens:     " + (.usage.prompt_tokens|tostring) + "\n - Completion Tokens: " + (.usage.completion_tokens|tostring) + "\n - Total Tokens:      " + (.usage.total_tokens|tostring)'
    fi
else
    echo "🔴 [FAIL] API Request Failed (Status: $HTTP_STATUS)"
    echo "--------------------------------------------------"
    echo "⚠️  [Error Message or Raw Body]:"
    if echo "$HTTP_BODY" | jq . >/dev/null 2>&1; then
        echo "$HTTP_BODY" | jq .
    else
        echo "$HTTP_BODY"
    fi
fi
echo "=================================================="
