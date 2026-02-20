#!/bin/bash

# ./scripts/prepare_pg19_chunks.sh

# pre-split PG19 books into non-overlapping chunks by ~word count
MIN_WORDS=8000
SAMPLE_N_VALID_CHUNKS=5000
OUTPUT_PATH="./data/pg19_chunks_w${MIN_WORDS}"
SPLITS="train,validation,test"

EXTRA_ARGS=()
if [ -n "$SAMPLE_N_VALID_CHUNKS" ]; then
  EXTRA_ARGS+=(--sample_n_valid_chunks "$SAMPLE_N_VALID_CHUNKS")
fi

python prepare_pg19_chunks.py \
  --dataset_name "deepmind/pg19" \
  --text_field "text" \
  --splits "$SPLITS" \
  --output_path "$OUTPUT_PATH" \
  --min_words $MIN_WORDS \
  "${EXTRA_ARGS[@]}" \
  --seed 144 \
  --overwrite
