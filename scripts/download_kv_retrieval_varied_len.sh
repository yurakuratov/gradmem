#!/bin/bash

DATASETS=("N4-vary-K2V2-V62_1M" "N8-vary-K2V2-V62_1M" "N16-vary-K2V2-V62_1M" "N32-vary-K2V2-V62_1M" "N64-vary-K2V2-V62_1M")

for DATASET in "${DATASETS[@]}"; do
    echo "Downloading $DATASET"
    python -c "import datasets; datasets.load_dataset('mkairov/${DATASET}').save_to_disk(f'./data/${DATASET}')"
done