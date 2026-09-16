#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

SEGMENT_REGIME=32
CURRICULUM_N=(8 16 32 64 64 128)
CE_WEIGHTS=(1 1 1 1 0 0)
INNER_LRS=(1 1 1 0.1 0.1 0.1)

source "$SCRIPT_DIR/run_energy_gradmem_curriculum_common.sh"
