#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=1
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

SEGMENT_REGIME=all
CURRICULUM_N=(8 8 16 32 64 128)
CE_WEIGHTS=(1 0 0 0 0 0)
INNER_LRS=(5.0 5.0 5.0 5.0 5.0 5.0)
START_STAGE=1
RUN_NUMBERS=(4)
MAX_STEPS=200000
source "$SCRIPT_DIR/run_energy_gradmem_curriculum_common.sh"
