#!/bin/bash

set -euo pipefail
shopt -s nullglob

# Two-stage curriculum for hidden-state-only EnergyGradMem:
# stage 1: inner objective = energy + CE
# stage 2: inner objective = energy only, initialized from stage 1 checkpoint

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
BASE_SCRIPT=${BASE_SCRIPT:-$SCRIPT_DIR/run_energy_gradmem_on_kv_retrieval.sh}

MODEL=${MODEL:-energy_gradmem}
BASE_MODEL=${BASE_MODEL:-llama}
L=${L:-4}
H=${H:-4}
D=${D:-128}
N_PAIRS=${N_PAIRS:-8}
N_PAIRS_IN_SEGMENT=${N_PAIRS_IN_SEGMENT:-$N_PAIRS}
N_SEGMENTS_IN_CONTEXT=${N_SEGMENTS_IN_CONTEXT:-1}
K_SIZE=${K_SIZE:-2}
V_SIZE=${V_SIZE:-2}
VOCAB_SIZE=${VOCAB_SIZE:-62}
HF_SUBSET=${HF_SUBSET:-N${N_PAIRS}-K${K_SIZE}V${V_SIZE}-V${VOCAB_SIZE}}
N_MEM_TOKENS=${N_MEM_TOKENS:-8}
K=${K:-2}
INNER_LR=${INNER_LR:-1.0}
GRAD_MODE=${GRAD_MODE:-second}
TBS=${TBS:-64}
LR=${LR:-1e-04}
N_VALUES=${N_VALUES:-1}
CURRICULUM_WEIGHTS=${CURRICULUM_WEIGHTS:-"1.0 0.0"}

RUN_NAME=${RUN_NAME:-energy_gradmem_curriculum_${BASE_MODEL}_L${L}H${H}D${D}_${HF_SUBSET}_mem${N_MEM_TOKENS}_K${K}_ilr${INNER_LR}_grad_${GRAD_MODE}_bs_${TBS}_lr_${LR}}
EXP_ROOT=${EXP_ROOT:-./runs/energy_gradmem_kv_curriculum/${HF_SUBSET}/${RUN_NAME}}

for N in $N_VALUES; do
  INIT_CKPT=${INIT_CHECKPOINT:-}
  STAGE=0
  for CE_WEIGHT in $CURRICULUM_WEIGHTS; do
    STAGE=$((STAGE + 1))
    STAGE_EXP_PATH=${EXP_ROOT}/run_${N}/stage_${STAGE}_ce_${CE_WEIGHT}
    STAGE_WANDB_NAME=${WANDB_NAME:-${MODEL}_curriculum_ce${CE_WEIGHT}_N${N_PAIRS_IN_SEGMENT}x${N_SEGMENTS_IN_CONTEXT}}

    EXP_PATH="$STAGE_EXP_PATH" \
    N_VALUES="$N" \
    RUN_NAME="$RUN_NAME" \
    WANDB_NAME="$STAGE_WANDB_NAME" \
    INIT_CHECKPOINT="$INIT_CKPT" \
    INNER_OBJECTIVE=lstm \
    ENERGY_FUTURE_MODE=none \
    ENERGY_INNER_CE_WEIGHT="$CE_WEIGHT" \
    ENERGY_CE_GUIDANCE=false \
    ENERGY_CE_GUIDANCE_ALPHA=0.0 \
    ENERGY_PRETRAIN_STEPS=0 \
    ENERGY_FREEZED_STEPS=0 \
    "$BASE_SCRIPT"

    BEST_STEP=-1
    NEXT_INIT_CKPT=""
    for CKPT_FILE in "$STAGE_EXP_PATH"/checkpoint-*/model.safetensors; do
      CKPT_DIR=$(basename "$(dirname "$CKPT_FILE")")
      CKPT_STEP=${CKPT_DIR#checkpoint-}
      if [[ "$CKPT_STEP" =~ ^[0-9]+$ ]] && [ "$CKPT_STEP" -gt "$BEST_STEP" ]; then
        BEST_STEP=$CKPT_STEP
        NEXT_INIT_CKPT=$CKPT_FILE
      fi
    done
    if [ -z "$NEXT_INIT_CKPT" ]; then
      echo "No checkpoint found after stage $STAGE in $STAGE_EXP_PATH" >&2
      exit 1
    fi
    INIT_CKPT=$NEXT_INIT_CKPT
  done
done
