#!/bin/sh
# Stop and remove both liteLLM proxy and sglang containers
docker stop litellm-proxy 2>/dev/null || true
docker rm   litellm-proxy 2>/dev/null || true
docker stop sglang-server  2>/dev/null || true
docker rm   sglang-server  2>/dev/null || true
echo "✓ All sglang and litellm containers stopped and removed"
