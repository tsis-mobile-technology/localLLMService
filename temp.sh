#!/bin/bash
# Find config.json files in the circulus gemma-4 model cache
CONFIG_FILES=$(find /home/proidea/.cache/huggingface/hub/models--circulus--gemma-4-E4B-it-ov-awq -name "config.json" 2>/dev/null)

if [ -z "$CONFIG_FILES" ]; then
  echo "❌ Config files not found in /home/proidea/.cache/huggingface/hub/models--circulus--gemma-4-E4B-it-ov-awq"
  exit 1
fi

for file in $CONFIG_FILES; do
  echo "Patching: $file"
  # Grant write permissions (using sudo)
  sudo chmod 666 "$file"
  
  # Apply the quantization configuration patch using sed
  sudo sed -i 's/"vision_soft_tokens_per_image": 280/"quantization_config": {"quant_method": "awq", "bits": 4, "group_size": 128, "zero_point": true, "version": "gemm"}, "vision_soft_tokens_per_image": 280/g' "$file"
  
  # Restore to read-only permissions
  sudo chmod 444 "$file"
done
echo "✅ AWQ configuration patch applied successfully!"
