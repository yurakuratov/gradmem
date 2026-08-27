#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/collect_env_state.sh"

NP=${NP:-1}
LR=${LR:-1e-04}
TBS=${TBS:-64}
PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE:-64}
STOP_ON_METRIC_VALUE=${STOP_ON_METRIC_VALUE:-0.99}
MIXED_PRECISION=${MIXED_PRECISION:-no}
RUN_NAME_SUFFIX=${RUN_NAME_SUFFIX:-}

L=${L:-4}
H=${H:-4}
D=${D:-256}
MAX_POSITION_EMBEDDINGS=${MAX_POSITION_EMBEDDINGS:-1024}
BASE_MODEL=${BASE_MODEL:-llama}

VOCAB_SIZE=${VOCAB_SIZE:-8192}
NUM_KV_PAIRS=8
INPUT_SEQ_LEN=$((3 * NUM_KV_PAIRS))
MQAR_NOISE_LVL=${MQAR_NOISE_LVL:-0.0}
QUERY_SAMPLING=${QUERY_SAMPLING:-uniform}
POWER_A=${POWER_A:-0.01}
TRAIN_NUM_EXAMPLES=${TRAIN_NUM_EXAMPLES:-1000000}
VALID_NUM_EXAMPLES=${VALID_NUM_EXAMPLES:-5000}
DATA_SEED=${DATA_SEED:-123}

case "$QUERY_SAMPLING" in
  uniform)
    DATA_NAME="mqar_N${NUM_KV_PAIRS}_V${VOCAB_SIZE}_L${INPUT_SEQ_LEN}"
    ;;
  power_law)
    DATA_NAME="mqar_nonuniform_A${POWER_A}_N${NUM_KV_PAIRS}_V${VOCAB_SIZE}_L${INPUT_SEQ_LEN}"
    ;;
  zoology)
    DATA_NAME="mqar_zoology_A${POWER_A}_N${NUM_KV_PAIRS}_V${VOCAB_SIZE}_L${INPUT_SEQ_LEN}"
    ;;
  *)
    echo "QUERY_SAMPLING must be uniform, power_law, or zoology, got: $QUERY_SAMPLING" >&2
    exit 2
    ;;
esac
if [ "$MQAR_NOISE_LVL" != "0.0" ]; then
  DATA_NAME="${DATA_NAME}_noise${MQAR_NOISE_LVL}"
fi
MQAR_DATA_PATH=${MQAR_DATA_PATH:-./data/${DATA_NAME}}

if (( TBS % (PER_DEVICE_BATCH_SIZE*NP) != 0 )); then
  echo "TBS must be divisible by PER_DEVICE_BATCH_SIZE*NP" >&2
  exit 2
fi
GRAD_ACC_STEPS=$((TBS/(PER_DEVICE_BATCH_SIZE*NP)))

N_MEM_TOKENS=${N_MEM_TOKENS:-8}
K=1
N_CTRL_TOKENS=${N_CTRL_TOKENS:-0}
USE_MEM_PROJ=true
MEM_PROJ_MODE=proj
USE_RECONSTRUCTION_LOSS=true
RECONSTRUCTION_LOSS_WEIGHT=1.0
USE_WRITE_HEAD=true
USE_MEM_RESIDUAL=${USE_MEM_RESIDUAL:-false}
ATTN_IMPLEMENTATION=${ATTN_IMPLEMENTATION:-eager}
MAX_STEPS=300000

RUN_NAME="rmt2segm_${BASE_MODEL}_L${L}H${H}D${D}_mem${N_MEM_TOKENS}_K${K}"
if [ "$N_CTRL_TOKENS" -gt 0 ]; then
  RUN_NAME=${RUN_NAME}_c${N_CTRL_TOKENS}
fi
if [ "$USE_MEM_PROJ" = true ]; then
  RUN_NAME=${RUN_NAME}_mem_${MEM_PROJ_MODE}
fi
if [ "$USE_RECONSTRUCTION_LOSS" = true ]; then
  RUN_NAME=${RUN_NAME}_rec_loss_w${RECONSTRUCTION_LOSS_WEIGHT}
fi
if [ "$USE_WRITE_HEAD" = true ]; then
  RUN_NAME=${RUN_NAME}_whead
fi
if [ "$USE_MEM_RESIDUAL" = true ]; then
  RUN_NAME=${RUN_NAME}_res
fi
RUN_NAME=${RUN_NAME}_bs_${TBS}_lr_${LR}
if [ "$MQAR_NOISE_LVL" != "0.0" ]; then
  RUN_NAME=${RUN_NAME}_mqarnoise${MQAR_NOISE_LVL}
fi
if [ -n "$RUN_NAME_SUFFIX" ]; then
  RUN_NAME=${RUN_NAME}_${RUN_NAME_SUFFIX}
fi

N_VALUES=(1 2 3)
for N in "${N_VALUES[@]}"; do
  EXP_PATH="./runs/${DATA_NAME}/${RUN_NAME}/run_${N}"
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
    run_rmt_on_mqar.py
      --exp_path "$EXP_PATH"
      --per_device_batch_size "$PER_DEVICE_BATCH_SIZE"
      --gradient_accumulation_steps "$GRAD_ACC_STEPS"
      --total_batch_size "$TBS"
      --vocab_size "$VOCAB_SIZE"
      --input_seq_len "$INPUT_SEQ_LEN"
      --num_kv_pairs "$NUM_KV_PAIRS"
      --mqar_noise_lvl "$MQAR_NOISE_LVL"
      --mqar_data_path "$MQAR_DATA_PATH"
      --query_sampling "$QUERY_SAMPLING"
      --power_a "$POWER_A"
      --train_num_examples "$TRAIN_NUM_EXAMPLES"
      --valid_num_examples "$VALID_NUM_EXAMPLES"
      --data_seed "$DATA_SEED"
      --dense_queries true
      --learning_rate "$LR"
      --n_layer "$L"
      --n_head "$H"
      --n_embd "$D"
      --max_position_embeddings "$MAX_POSITION_EMBEDDINGS"
      --base_model "$BASE_MODEL"
      --n_mem_tokens "$N_MEM_TOKENS"
      --K "$K"
      --n_ctrl_tokens "$N_CTRL_TOKENS"
      --use_mem_proj "$USE_MEM_PROJ"
      --mem_proj_mode "$MEM_PROJ_MODE"
      --use_reconstruction_loss "$USE_RECONSTRUCTION_LOSS"
      --reconstruction_loss_weight "$RECONSTRUCTION_LOSS_WEIGHT"
      --use_write_head "$USE_WRITE_HEAD"
      --use_mem_residual "$USE_MEM_RESIDUAL"
      --attn_implementation "$ATTN_IMPLEMENTATION"
      --max_steps "$MAX_STEPS"
      --eval_steps 500
      --logging_steps 500
      --warmup_steps 10000
      --early_stopping_patience 500
      --stop_on_metric_value "$STOP_ON_METRIC_VALUE"
      --seed "$((142+N))"
  )

  if [ -n "${INIT_CHECKPOINT:-}" ]; then
    CMD+=( --init_checkpoint "$INIT_CHECKPOINT" )
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
