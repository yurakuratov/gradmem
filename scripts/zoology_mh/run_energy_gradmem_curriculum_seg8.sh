#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

SEGMENT_REGIME=8
CURRICULUM_N=(8 16 16 32 64 128)
CE_WEIGHTS=(1 1 0 0 0 0)
INNER_LRS=(5.0 0.4 0.1 0.1 0.03 0.03)
START_STAGE=1
RUN_NUMBERS=(2)

source "$SCRIPT_DIR/run_energy_gradmem_curriculum_common.sh"
