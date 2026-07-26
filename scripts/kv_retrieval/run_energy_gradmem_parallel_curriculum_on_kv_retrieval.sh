#!/bin/bash

set -euo pipefail
shopt -s nullglob

# Curriculum with semi-parallel segment writes.
# stage 1: inner objective = energy + CE
# later stages: inner objective = energy only, initialized from previous stage checkpoint

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
BASE_SCRIPT=$SCRIPT_DIR/run_energy_gradmem_on_kv_retrieval.sh

MODEL=energy_gradmem
BASE_MODEL=llama
L=4
H=4
D=128
N_PAIRSS_IN_SEGMENT=(8 8 8 8)
N_SEGMENTSS_IN_CONTEXT=(1 1 2 4)
K_SIZE=2
V_SIZE=2
VOCAB_SIZE=62
N_MEM_TOKENS=8
K=2
INNER_LRS=(1.0 1.0 1.0 1.0)
GRAD_MODE=second
TBS=64
LR=1e-04
N_VALUES=1
CE_WEIGHTS=(1.0 0.0 0.0 0.0)
START_ITERATION=1
INIT_CHECKPOINT=""
STOP_EXACT_MATCH_VALUE=0.99
INNER_OBJECTIVE=neural
ENERGY_MODEL_TYPE=lstm

if [ ${#CE_WEIGHTS[@]} -eq 0 ]; then
  echo "CE_WEIGHTS must be non-empty" >&2
  exit 1
fi
if [ ${#N_PAIRSS_IN_SEGMENT[@]} -ne ${#CE_WEIGHTS[@]} ]; then
  echo "N_PAIRSS_IN_SEGMENT must have one value per curriculum stage" >&2
  exit 1
fi
if [ ${#N_SEGMENTSS_IN_CONTEXT[@]} -ne ${#CE_WEIGHTS[@]} ]; then
  echo "N_SEGMENTSS_IN_CONTEXT must have one value per curriculum stage" >&2
  exit 1
fi
if [ ${#INNER_LRS[@]} -ne ${#CE_WEIGHTS[@]} ]; then
  echo "INNER_LRS must have one value per curriculum stage" >&2
  exit 1
fi

if ! [[ "$START_ITERATION" =~ ^[0-9]+$ ]] || [ "$START_ITERATION" -lt 1 ]; then
  echo "START_ITERATION must be a positive 1-based integer, got: $START_ITERATION" >&2
  exit 1
fi

latest_checkpoint() {
  local stage_path=$1
  local best_step=-1
  local next_init_ckpt=""
  local ckpt_file ckpt_dir ckpt_step

  for ckpt_file in "$stage_path"/checkpoint-*/model.safetensors; do
    ckpt_dir=$(basename "$(dirname "$ckpt_file")")
    ckpt_step=${ckpt_dir#checkpoint-}
    if [[ "$ckpt_step" =~ ^[0-9]+$ ]] && [ "$ckpt_step" -gt "$best_step" ]; then
      best_step=$ckpt_step
      next_init_ckpt=$ckpt_file
    fi
  done

  if [ -z "$next_init_ckpt" ]; then
    return 1
  fi
  printf '%s\n' "$next_init_ckpt"
}

for N in $N_VALUES; do
  SEED=$((N + 42))
  INIT_CKPT=$INIT_CHECKPOINT
  STAGE=0
  for STAGE_INDEX in "${!CE_WEIGHTS[@]}"; do
    STAGE=$((STAGE + 1))
    CE_WEIGHT=${CE_WEIGHTS[$STAGE_INDEX]}
    INNER_LR=${INNER_LRS[$STAGE_INDEX]}
    N_PAIRS_IN_SEGMENT=${N_PAIRSS_IN_SEGMENT[$STAGE_INDEX]}
    N_SEGMENTS_IN_CONTEXT=${N_SEGMENTSS_IN_CONTEXT[$STAGE_INDEX]}
    N_PAIRS=$((N_PAIRS_IN_SEGMENT * N_SEGMENTS_IN_CONTEXT))
    HF_SUBSET=N${N_PAIRS}-K${K_SIZE}V${V_SIZE}-V${VOCAB_SIZE}
    RUN_NAME=energy_gradmem_parallel_curriculum_${BASE_MODEL}_L${L}H${H}D${D}_${HF_SUBSET}_mem${N_MEM_TOKENS}_K${K}_ilr${INNER_LR}_grad_${GRAD_MODE}_bs_${TBS}_lr_${LR}
    EXP_ROOT=./runs/energy_gradmem_kv_parallel_curriculum/${HF_SUBSET}/${RUN_NAME}
    STAGE_EXP_PATH=${EXP_ROOT}/run_${N}/stage_${STAGE}_ce_${CE_WEIGHT}
    STAGE_WANDB_NAME=${MODEL}_parallel_curriculum_ce${CE_WEIGHT}_N${N_PAIRS_IN_SEGMENT}x${N_SEGMENTS_IN_CONTEXT}

    if [ "$STAGE" -lt "$START_ITERATION" ]; then
      echo "Skipping stage $STAGE; reading checkpoint from $STAGE_EXP_PATH"
    else
      EXP_PATH="$STAGE_EXP_PATH" \
      N_VALUES="$N" \
      SEED="$SEED" \
      RUN_NAME="$RUN_NAME" \
      WANDB_NAME="$STAGE_WANDB_NAME" \
      INIT_CHECKPOINT="$INIT_CKPT" \
      N_PAIRS="$N_PAIRS" \
      N_PAIRS_IN_SEGMENT="$N_PAIRS_IN_SEGMENT" \
      N_SEGMENTS_IN_CONTEXT="$N_SEGMENTS_IN_CONTEXT" \
      INNER_LR="$INNER_LR" \
      SEGMENT_WRITE_MODE=parallel \
      INNER_OBJECTIVE="$INNER_OBJECTIVE" \
      ENERGY_MODEL_TYPE="$ENERGY_MODEL_TYPE" \
      ENERGY_FUTURE_MODE=none \
      ENERGY_INNER_CE_WEIGHT="$CE_WEIGHT" \
      ENERGY_CE_GUIDANCE=false \
      ENERGY_CE_GUIDANCE_ALPHA=0.0 \
      ENERGY_PRETRAIN_STEPS=0 \
      ENERGY_FREEZED_STEPS=0 \
      STOP_EXACT_MATCH_VALUE="$STOP_EXACT_MATCH_VALUE" \
      "$BASE_SCRIPT"
    fi

    if ! INIT_CKPT=$(latest_checkpoint "$STAGE_EXP_PATH"); then
      echo "No checkpoint found for stage $STAGE in $STAGE_EXP_PATH" >&2
      exit 1
    fi
  done
done
