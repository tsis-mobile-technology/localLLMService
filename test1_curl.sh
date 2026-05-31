#!/bin/bash

# Configuration
SGLANG_URL="http://localhost:30000/v1"
LITELLM_URL="http://localhost:4000/v1"
LITELLM_KEY="sk-local-master"
SGLANG_KEY="sk-local"

SGLANG_MODEL="google/gemma-4-E4B-it"
LITELLM_MODEL="gemma-4"

echo "=================================================="
echo "🚀 Starting Sequence API Tests"
echo "=================================================="

# --------------------------------------------------
# Test 1: SGLang Server directly (Port 30000)
# --------------------------------------------------
echo ""
echo "👉 [STEP 1] Testing SGLang Direct (Port 30000)"
echo " - URL:   $SGLANG_URL"
echo " - Model: $SGLANG_MODEL"
echo "--------------------------------------------------"

START_TIME=$(date +%s.%N)
RESPONSE_DATA=$(curl -s -w "\n%{http_code}" --location "$SGLANG_URL/chat/completions" \
  --header "Content-Type: application/json" \
  --header "Authorization: Bearer $SGLANG_KEY" \
  --data "{
    \"model\": \"$SGLANG_MODEL\",
    \"messages\": [
      {
        \"role\": \"user\",
        \"content\": \"안녕하세요! 간단하게 한 단어로 답해주세요: (예: 성공)\"
      }
    ],
    \"max_tokens\": 10
  }")
END_TIME=$(date +%s.%N)
DURATION=$(echo "$END_TIME - $START_TIME" | bc 2>/dev/null || echo "N/A")

HTTP_STATUS=$(echo "$RESPONSE_DATA" | tail -n 1)
HTTP_BODY=$(echo "$RESPONSE_DATA" | sed '$d')

if [ "$HTTP_STATUS" -eq 200 ]; then
    echo "🟢 [SUCCESS] SGLang Direct Response Received (Status: $HTTP_STATUS)"
    echo "⏱️  Response Time: ${DURATION}s"
    echo "🤖 [Answer]: $(echo "$HTTP_BODY" | jq -r '.choices[0].message.content')"
else
    echo "🔴 [FAIL] SGLang Direct Request Failed (Status: $HTTP_STATUS)"
    echo "⚠️  [Error]:"
    echo "$HTTP_BODY"
fi

# --------------------------------------------------
# Test 2: liteLLM Proxy (Port 4000)
# --------------------------------------------------
echo ""
echo "👉 [STEP 2] Testing liteLLM Proxy (Port 4000)"
echo " - URL:   $LITELLM_URL"
echo " - Model: $LITELLM_MODEL"
echo "--------------------------------------------------"

START_TIME=$(date +%s.%N)
RESPONSE_DATA=$(curl -s -w "\n%{http_code}" --location "$LITELLM_URL/chat/completions" \
  --header "Content-Type: application/json" \
  --header "Authorization: Bearer $LITELLM_KEY" \
  --data "{
    \"model\": \"$LITELLM_MODEL\",
    \"messages\": [
      {
        \"role\": \"user\",
        \"content\": \"안녕하세요! 간단하게 한 단어로 답해주세요: (예: 완료)\"
      }
    ],
    \"max_tokens\": 10
  }")
END_TIME=$(date +%s.%N)
DURATION=$(echo "$END_TIME - $START_TIME" | bc 2>/dev/null || echo "N/A")

HTTP_STATUS=$(echo "$RESPONSE_DATA" | tail -n 1)
HTTP_BODY=$(echo "$RESPONSE_DATA" | sed '$d')

if [ "$HTTP_STATUS" -eq 200 ]; then
    echo "🟢 [SUCCESS] liteLLM Proxy Response Received (Status: $HTTP_STATUS)"
    echo "⏱️  Response Time: ${DURATION}s"
    echo "🤖 [Answer]: $(echo "$HTTP_BODY" | jq -r '.choices[0].message.content')"
else
    echo "🔴 [FAIL] liteLLM Proxy Request Failed (Status: $HTTP_STATUS)"
    echo "⚠️  [Error]:"
    echo "$HTTP_BODY"
fi
echo ""
echo "=================================================="
