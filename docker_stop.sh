#!/bin/sh
# Stop and remove both liteLLM proxy and llama.cpp containers
docker stop litellm-proxy 2>/dev/null || true
docker rm   litellm-proxy 2>/dev/null || true
docker stop llama-server  2>/dev/null || true
docker rm   llama-server  2>/dev/null || true
echo "✓ All containers stopped and removed"
