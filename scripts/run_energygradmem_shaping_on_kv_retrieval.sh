#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Calibrated landscape-shaping recipe. Every value remains environment-
# overridable so stronger or weaker ablations can use the same launcher.
export ENERGY_RANK_WEIGHT=${ENERGY_RANK_WEIGHT:-0.01}
export ENERGY_TRAJ_WEIGHT=${ENERGY_TRAJ_WEIGHT:-0.0}
export ENERGY_ANCHOR_WEIGHT=${ENERGY_ANCHOR_WEIGHT:-0.001}
export LIPSCHITZ_WEIGHT=${LIPSCHITZ_WEIGHT:-0.0}
export LIPSCHITZ_CONSTRAINT=${LIPSCHITZ_CONSTRAINT:-1.0}
export ENERGY_MEMORY_SEARCH_MIN_RELATIVE_TARGET_GAIN=${ENERGY_MEMORY_SEARCH_MIN_RELATIVE_TARGET_GAIN:-0.0}
export MEMORY_NOISE_SIGMA=${MEMORY_NOISE_SIGMA:-0.0}
export ENERGY_MARGIN=${ENERGY_MARGIN:-0.1}
export ENERGY_TRAJ_MARGIN=${ENERGY_TRAJ_MARGIN:-0.0}
export ENERGY_RANK_TEMPERATURE=${ENERGY_RANK_TEMPERATURE:-1.0}
export ENERGY_MIX_ALPHA=${ENERGY_MIX_ALPHA:-0.75}
export WRITE_OBJECTIVE="energy"

# shellcheck source=run_energygradmem_on_kv_retrieval.sh
source "$SCRIPT_DIR/run_energygradmem_on_kv_retrieval.sh"
