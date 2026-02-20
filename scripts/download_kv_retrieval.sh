#!/bin/bash

DATASETS=("N4-K2V2-V62_1M" "N8-K2V2-V62_1M" "N16-K2V2-V62_1M" "N32-K2V2-V62_1M" "N64-K2V2-V62_1M"
"N96-K2V2-V62_1M" "N128-K2V2-V62_1M" 
"N8-K2V2-V62_noise_0.25_1M" "N8-K2V2-V62_noise_0.5_1M" "N8-K2V2-V62_noise_0.75_1M"
"N16-K2V2-V62_noise_0.25_1M" "N16-K2V2-V62_noise_0.5_1M" "N16-K2V2-V62_noise_0.75_1M"
)

for DATASET in "${DATASETS[@]}"; do
    echo "Downloading $DATASET"
    python -c "import datasets; datasets.load_dataset('yurakuratov/${DATASET}').save_to_disk(f'./data/${DATASET}')"
done