#!/bin/bash

DATASETS=("babilong_qa1_0k" "babilong_qa2_0k" "babilong_qa3_0k" "babilong_qa4_0k" "babilong_qa5_0k")

for DATASET in "${DATASETS[@]}"; do
    echo "Downloading $DATASET"
    python -c "import datasets; datasets.load_dataset('yurakuratov/${DATASET}').save_to_disk(f'./data/${DATASET}')"
done