#!/usr/bin/env bash

# Zoology configuration for the shared multihop curriculum implementation.
# The entry-point script defines SEGMENT_REGIME and the three curriculum arrays.

if [ "${BASH_SOURCE[0]}" = "$0" ]; then
  echo "Source this helper from a Zoology curriculum entry point." >&2
  exit 2
fi

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ZOOLOGY_REPO_ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd)
DATASET_FORMAT=zoology
DATASET_VOCAB_SUFFIX=4096
MODEL_VOCAB_SIZE=4096
DEFAULT_HF_DATASET=irodkin/zoology_multihop
DEFAULT_OUTPUT_ROOT="$ZOOLOGY_REPO_ROOT/runs/energy_gradmem_zoology_mh_curriculum"
DEFAULT_WANDB_PROJECT=zoology_multihop
RUN_NAME_PREFIX=energy_gradmem_zoology_mh
INCLUDE_HOP_IN_WANDB_NAME=false
CURRICULUM_GATE_SCRIPT="$ZOOLOGY_REPO_ROOT/zoology_curriculum_gate.py"

source "$SCRIPT_DIR/../ar_multihop/run_energy_gradmem_curriculum_common.sh"
