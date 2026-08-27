#!/bin/bash

DATASETS=(
  "mqar_N8_V8192_L24"
  "mqar_N16_V8192_L48"
  "mqar_N32_V8192_L96"
  "mqar_N64_V8192_L192"
  "mqar_N8_V8192_L24_noise0.25"
  "mqar_N8_V8192_L24_noise0.5"
  "mqar_N8_V8192_L24_noise0.75"
  "mqar_N8_V8192_L24_noise0.9"
)

for DATASET in "${DATASETS[@]}"; do
    echo "Downloading $DATASET"
    python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='mkairov/${DATASET}', repo_type='dataset', local_dir='./data/${DATASET}')"
done
