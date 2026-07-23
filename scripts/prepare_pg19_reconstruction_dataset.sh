#!/bin/bash

# ./scripts/prepare_pg19_reconstruction_dataset.sh
#
# Builds the PG19 associative-reconstruction dataset from the chunks produced
# by prepare_pg19_chunks.sh. Requires ./data/pg19_chunks_w8000 (or pass
# CHUNKS_PATH) to already exist.

set -euo pipefail

CHUNKS_PATH="${CHUNKS_PATH:-./data/pg19_chunks_w8000}"
OUTPUT_PATH="${OUTPUT_PATH:-./data/pg19_reconstruction}"

# Token-length budgets (estimated with GPT-2 for length control only).
CONTEXT_MIN_TOKENS="${CONTEXT_MIN_TOKENS:-512}"
CONTEXT_MAX_TOKENS="${CONTEXT_MAX_TOKENS:-1024}"
SPAN_MIN_TOKENS="${SPAN_MIN_TOKENS:-32}"
SPAN_MAX_TOKENS="${SPAN_MAX_TOKENS:-128}"
QUERY_FRACTION="${QUERY_FRACTION:-0.5}"

N_TRAIN="${N_TRAIN:-100000}"
N_VALID="${N_VALID:-2000}"
SEED="${SEED:-142}"

python prepare_pg19_reconstruction_dataset.py \
  --chunks_path "$CHUNKS_PATH" \
  --output_path "$OUTPUT_PATH" \
  --context_min_tokens "$CONTEXT_MIN_TOKENS" \
  --context_max_tokens "$CONTEXT_MAX_TOKENS" \
  --span_min_tokens "$SPAN_MIN_TOKENS" \
  --span_max_tokens "$SPAN_MAX_TOKENS" \
  --query_fraction "$QUERY_FRACTION" \
  --n_train "$N_TRAIN" \
  --n_valid "$N_VALID" \
  --seed "$SEED" \
  --pretrained_model gpt2 \
  --overwrite
