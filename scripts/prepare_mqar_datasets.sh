#!/bin/bash
set -euo pipefail

OUTPUT_DIR=${OUTPUT_DIR:-./data}
VOCAB_SIZE=${VOCAB_SIZE:-8192}
TRAIN_NUM_EXAMPLES=${TRAIN_NUM_EXAMPLES:-1000000}
VALID_NUM_EXAMPLES=${VALID_NUM_EXAMPLES:-5000}
DATA_SEED=${DATA_SEED:-123}
NOISE_LEVELS=${NOISE_LEVELS:-0.0}
read -r -a NOISE_LEVEL_ARRAY <<< "$NOISE_LEVELS"

CMD=(
  python prepare_mqar_datasets.py
  --output_dir "$OUTPUT_DIR"
  --pair_counts 8 16 32 64
  --noise_levels "${NOISE_LEVEL_ARRAY[@]}"
  --vocab_size "$VOCAB_SIZE"
  --train_num_examples "$TRAIN_NUM_EXAMPLES"
  --valid_num_examples "$VALID_NUM_EXAMPLES"
  --data_seed "$DATA_SEED"
)
if [ "${OVERWRITE:-false}" = true ]; then
  CMD+=( --overwrite )
fi

"${CMD[@]}"
