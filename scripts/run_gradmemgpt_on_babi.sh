#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/collect_env_state.sh"

# Define arguments for the script
NP=${NP:-1}  # Default to 1 process if not set
LR=1e-04
TBS=64
PER_DEVICE_BATCH_SIZE=8
GRAD_ACC_STEPS=$(($TBS/($PER_DEVICE_BATCH_SIZE*$NP)))

MODEL_NAME=gpt2
PRETRAINED_MODEL=gpt2
MAX_CONTEXT_LENGTH=1016

# GradMemGPT specific parameters
N_MEM_TOKENS=8
N_CTRL_TOKENS=0
K=1
INNER_LR=0.04
INNER_CLIP_VALUE=None
INNER_CLIP_NORM=None
USE_ADAM=false
GRAD_MODE="second"
USE_MEM_PROJ=true
MEM_PROJ_MODE="proj"
USE_WRITE_HEAD=true
ATTN_IMPL="eager"
MIXED_PRECISION='bf16'

# INIT_CHECKPOINT=./runs/babilong_qa3_0k/gradmem_gpt2_mem8_K1_ilr0.4_mem_proj_whead_grad_second_bs_64_lr_1e-04/run_2_bf16/checkpoint-127500/model.safetensors
# RUN_NAME_SUFFIX=init_qa3_K1_ilr0.4_run_2_bf16

RUN_NAME=gradmem_${MODEL_NAME}_mem${N_MEM_TOKENS}
if [ "$N_CTRL_TOKENS" -gt 0 ]; then
  RUN_NAME=${RUN_NAME}_c${N_CTRL_TOKENS}
fi
RUN_NAME=${RUN_NAME}_K${K}_ilr${INNER_LR}
if [ "$INNER_CLIP_VALUE" != "None" ]; then
  RUN_NAME=${RUN_NAME}_icv${INNER_CLIP_VALUE}
fi
if [ "$INNER_CLIP_NORM" != "None" ]; then
  RUN_NAME=${RUN_NAME}_icn${INNER_CLIP_NORM}
fi
if [ "$USE_MEM_PROJ" = true ]; then
  RUN_NAME=${RUN_NAME}_mem_proj
  if [ "$MEM_PROJ_MODE" == "per_sample" ]; then
    RUN_NAME=${RUN_NAME}_ps
  fi
fi
if [ "$USE_WRITE_HEAD" = true ]; then
  RUN_NAME=${RUN_NAME}_whead
fi
RUN_NAME=${RUN_NAME}_grad_${GRAD_MODE}
if [ "$USE_ADAM" = true ]; then
  RUN_NAME=${RUN_NAME}_with_adam
fi

RUN_NAME=${RUN_NAME}_bs_${TBS}_lr_${LR}

if [ -n "${RUN_NAME_SUFFIX:-}" ]; then
  RUN_NAME=${RUN_NAME}_${RUN_NAME_SUFFIX}
fi

for task_name in "qa2"; do
  DATA_NAME="babilong_${task_name}_0k"
  DATA_PATH="./data/${DATA_NAME}"

  N_VALUES=(1 2 3)
  for N in "${N_VALUES[@]}"; do
    # Path to save experiment results
    RND=$(date +%Y%m%d%H%M%S)
    EXP_PATH="./runs/${DATA_NAME}/${RUN_NAME}_dbg/run_${N}"
    if [ "$MIXED_PRECISION" != "no" ]; then
      EXP_PATH="${EXP_PATH}_${MIXED_PRECISION}"
    fi

    if ! prepare_locked_run "$EXP_PATH" "$0" "$NP"; then
      continue
    fi

    PORT="$(find_free_port)"

    # Build accelerate command as array (safe quoting)
    CMD=(
      accelerate launch
        --main_process_port "$PORT"
        --num_processes "$NP"
        --mixed_precision "$MIXED_PRECISION"
        --multi_gpu
        --config_file accelerate.yaml
      run_gradmemgpt_on_kv_retrieval.py
        --exp_path "$EXP_PATH"
        --per_device_batch_size "$PER_DEVICE_BATCH_SIZE"
        --gradient_accumulation_steps "$GRAD_ACC_STEPS"
        --total_batch_size "$TBS"
        --data_path "$DATA_PATH"
        --learning_rate "$LR"
        --pretrained_model "$PRETRAINED_MODEL"
        --n_mem_tokens "$N_MEM_TOKENS"
        --K "$K"
        --inner_lr "$INNER_LR"
        --use_adam "$USE_ADAM"
        --grad_mode "$GRAD_MODE"
        --max_steps 200000
        --eval_steps 500
        --logging_steps 500
        --warmup_steps 10000
        --early_stopping_patience 250
        --seed "$((142+N))"
    )

    # Optional args
    if [ -n "${INIT_CHECKPOINT:-}" ]; then
      CMD+=( --init_checkpoint "$INIT_CHECKPOINT" )
    fi
    if [ "$INNER_CLIP_VALUE" != "None" ]; then
      CMD+=( --inner_clip_value "$INNER_CLIP_VALUE" )
    fi
    if [ "$INNER_CLIP_NORM" != "None" ]; then
      CMD+=( --inner_clip_norm "$INNER_CLIP_NORM" )
    fi
    if [ "$USE_MEM_PROJ" = true ]; then
      CMD+=( --use_mem_proj --mem_proj_mode "$MEM_PROJ_MODE" )
    fi
    if [ "$USE_WRITE_HEAD" = true ]; then
      CMD+=( --use_write_head )
    fi
    if [ -n "${ATTN_IMPL:-}" ]; then
      CMD+=( --attn_implementation "$ATTN_IMPL" )
    fi
    if [ -n "${MAX_CONTEXT_LENGTH:-}" ]; then
      CMD+=( --max_context_length "$MAX_CONTEXT_LENGTH" )
    fi

    print_run_header "$EXP_PATH" "$PORT" "$NP" "$MIXED_PRECISION" "${CMD[@]}"

    start_run_timer

    set +e
    # actually running command
    "${CMD[@]}" 2>&1 | tee -a "$RUN_LOCK_LOG"
    RC=${PIPESTATUS[0]}
    set -e

    finalize_locked_run "$EXP_PATH" "$RUN_LOCK_DIR" "$RUN_LOCK_LOG" "$RC"

    # If you want the loop to continue even if one run fails, comment the next line.
    # exit $RC
  done
done

echo "Done"