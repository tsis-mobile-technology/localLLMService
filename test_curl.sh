#!/bin/bash

# Configuration
#BASE_URL="http://172.25.110.204:4000/v1"
#API_KEY="sk-drPtCcb3695mAXc_5_SRlg"
#MODEL="gemma-4-31b"
BASE_URL="http://172.25.110.204:8000/v1"
API_KEY="sk-"
MODEL="google/gemma-4-31B-it"
#BASE_URL="http://localhost:4000/v1"
#API_KEY="sk-drPtCcb3695mAXc_5_SRlg"
#API_KEY="sk-local-master"
#MODEL="gpt-4o"
# "gemma-4-31b"

echo "=================================================="
echo "🚀 Testing API Endpoint"
echo " - URL:   $BASE_URL"
echo " - Model: $MODEL"
echo "=================================================="

# 1. API 호출 및 소요 시간 측정 (HTTP 상태 코드와 응답 바디 분리 수집)
START_TIME=$(date +%s.%N)

RESPONSE_DATA=$(curl -s -w "\n%{http_code}" --location "$BASE_URL/chat/completions" \
  --header "Content-Type: application/json" \
  --header "Authorization: Bearer $API_KEY" \
  --data "{
    \"model\": \"$MODEL\",
    \"messages\": [
      {
        \"role\": \"user\",
        \"content\": \"안녕, 오늘 날짜를 유추해서 시를 작성해..\"
      }
    ]
  }")

END_TIME=$(date +%s.%N)
DURATION=$(echo "$END_TIME - $START_TIME" | bc 2>/dev/null || echo "N/A")

# 응답 바디와 HTTP 상태 코드 분리
HTTP_STATUS=$(echo "$RESPONSE_DATA" | tail -n 1)
HTTP_BODY=$(echo "$RESPONSE_DATA" | sed '$d')

# 2. 결과 판별 및 시각화 출력
echo ""
if [ "$HTTP_STATUS" -eq 200 ]; then
    echo "🟢 [SUCCESS] API Response Received (Status: $HTTP_STATUS)"
    echo "⏱️  Response Time: ${DURATION}s"
    echo "--------------------------------------------------"
    
    # jq를 이용한 주요 데이터 파싱 및 가독성 높은 출력
    echo "🤖 [Assistant Answer]:"
    echo "$HTTP_BODY" | jq -r '.choices[0].message.content'
    
    echo "--------------------------------------------------"
    echo "📊 [Token Usage]:"
    echo "$HTTP_BODY" | jq -r '" - Prompt Tokens:     " + (.usage.prompt_tokens|tostring) + "\n - Completion Tokens: " + (.usage.completion_tokens|tostring) + "\n - Total Tokens:      " + (.usage.total_tokens|tostring)'
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
