#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/collect_env_state.sh"

# Define arguments for the script
NP=${NP:-1}  # Default to 1 process if not set
LR=1e-04
ADAM_BETA2=${ADAM_BETA2:-}
# ADAM_BETA2=0.98
TBS=64
PER_DEVICE_BATCH_SIZE=64
GRAD_ACC_STEPS=$(($TBS/($PER_DEVICE_BATCH_SIZE*$NP)))
STOP_ON_METRIC_VALUE=${STOP_ON_METRIC_VALUE:-1.00}
STOP_ON_METRIC_VALUE=0.99
MIXED_PRECISION=${MIXED_PRECISION:-no}

L=4
H=4
D=128
MAX_POSITION_EMBEDDINGS=1024
BASE_MODEL=llama

# INIT_CHECKPOINT=./runs/N64-K2V2-V62_1M/llama_L4H4D128_L1024_bs_64_lr_5e-05_b2_0.98/run_3/checkpoint-168500/model.safetensors
# RUN_NAME_SUFFIX=init_N64
RUN_NAME_SUFFIX=0.99

V=62
# Dataset parameters
# DATA_NAME="N2-K4V4-S4(32-64)_1M"
# DATA_NAME="N1-K4V4-S1(16-32)_1M"
# DATA_NAME="N10-K2V2-S4(32-64)_1M"
# DATA_NAME="N8-K1V1-vocab512_1M"
DATA_NAME="N8-K2V2-V${V}_1M"
# DATA_NAME="N4-K1V1-vocab512_1M"
# copy task
# DATA_NAME="N0-S1(4-4)_1M"
DATA_PATH="./data/${DATA_NAME}"
TOKENIZER_PATH="./tokenizers/kv_alphabet_${V}/"

if [ "$BASE_MODEL" == "mamba" ]; then
  RUN_NAME="${BASE_MODEL}_L${L}D${D}"
else
  RUN_NAME="${BASE_MODEL}_L${L}H${H}D${D}"
  if [ -n "$MAX_POSITION_EMBEDDINGS" ]; then
    RUN_NAME="${RUN_NAME}_L${MAX_POSITION_EMBEDDINGS}"
  fi
fi

RUN_NAME=${RUN_NAME}_bs_${TBS}_lr_${LR}

if [ -n "$ADAM_BETA2" ]; then
  RUN_NAME=${RUN_NAME}_b2_${ADAM_BETA2}
fi

if [ -n "$RUN_NAME_SUFFIX" ]; then
  RUN_NAME=${RUN_NAME}_${RUN_NAME_SUFFIX}
fi


# Run ID
N_VALUES=(1 2)
for N in "${N_VALUES[@]}"; do
  # Path to save experiment results
  # RND=$(date +%Y%m%d%H%M%S)
  # EXP_PATH="./runs/${DATA_NAME}/${RUN_NAME}_${RND}_DBG/run_$N"
  # EXP_PATH="./runs/${DATA_NAME}/mamba_L4D128_bs_64_lr_3e-04_20251005001255_DBG/run_1"
  EXP_PATH="./runs/${DATA_NAME}/${RUN_NAME}/run_$N"

  if [ "$MIXED_PRECISION" != "no" ]; then
    EXP_PATH="${EXP_PATH}_${MIXED_PRECISION}"
  fi

  if ! prepare_locked_run "$EXP_PATH" "$0" "$NP"; then
    continue
  fi

  PORT="$(find_free_port)"

  CMD=(
    accelerate launch
      --main_process_port "$PORT"
      --num_processes "$NP"
      --mixed_precision "$MIXED_PRECISION"
      --config_file accelerate.yaml
    run_gpt2_on_kv_retrieval.py
      --exp_path "$EXP_PATH"
      --per_device_batch_size "$PER_DEVICE_BATCH_SIZE"
      --gradient_accumulation_steps "$GRAD_ACC_STEPS"
      --total_batch_size "$TBS"
      --data_path "$DATA_PATH"
      --tokenizer_path "$TOKENIZER_PATH"
      --learning_rate "$LR"
      --n_layer "$L"
      --n_head "$H"
      --n_embd "$D"
      --base_model "$BASE_MODEL"
      --max_steps 200000
      --eval_steps 500
      --logging_steps 500
      --warmup_steps 10000
      --early_stopping_patience 500
      --stop_on_metric_value "$STOP_ON_METRIC_VALUE"
      --seed "$((142+N))"
  )

  if [ -n "$ADAM_BETA2" ]; then
    CMD+=(--adam_beta2 "$ADAM_BETA2")
  fi
  if [ -n "$MAX_POSITION_EMBEDDINGS" ]; then
    CMD+=(--max_position_embeddings "$MAX_POSITION_EMBEDDINGS")
  fi
  if [ -n "${INIT_CHECKPOINT:-}" ]; then
    CMD+=(--init_checkpoint "$INIT_CHECKPOINT")
  fi

  print_run_header "$EXP_PATH" "$PORT" "$NP" "$MIXED_PRECISION" "${CMD[@]}"
  start_run_timer

  set +e
  "${CMD[@]}" 2>&1 | tee -a "$RUN_LOCK_LOG"
  RC=${PIPESTATUS[0]}
  set -e

  finalize_locked_run "$EXP_PATH" "$RUN_LOCK_DIR" "$RUN_LOCK_LOG" "$RC"
done

echo "Done"
