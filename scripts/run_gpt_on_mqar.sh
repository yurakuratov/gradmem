#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/collect_env_state.sh"

# Define arguments for the script
NP=${NP:-1}  # Default to 1 process if not set
LR=${LR:-1e-04}
ADAM_BETA2=${ADAM_BETA2:-}
TBS=${TBS:-64}
PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE:-64}
STOP_ON_METRIC_VALUE=${STOP_ON_METRIC_VALUE:-0.99}
MIXED_PRECISION=${MIXED_PRECISION:-no}
RUN_NAME_SUFFIX=${RUN_NAME_SUFFIX:-$STOP_ON_METRIC_VALUE}

L=4
H=4
D=256
MAX_POSITION_EMBEDDINGS=1024
BASE_MODEL=llama

# Dense MQAR query-order distribution.
VOCAB_SIZE=8192
NUM_KV_PAIRS=8
INPUT_SEQ_LEN=$((3 * NUM_KV_PAIRS))
# zoology uses power_law with 0.01
QUERY_SAMPLING=${QUERY_SAMPLING:-uniform}
POWER_A=${POWER_A:-0.01}
TRAIN_NUM_EXAMPLES=100000
VALID_NUM_EXAMPLES=3000
DATA_SEED=123

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

# For sparse upstream MQAR, add --dense_queries false and set INPUT_SEQ_LEN to
# at least 4*NUM_KV_PAIRS. query_sampling only controls dense query ordering.

if (( TBS % (PER_DEVICE_BATCH_SIZE*NP) != 0 )); then
  echo "TBS must be divisible by PER_DEVICE_BATCH_SIZE*NP" >&2
  exit 2
fi
GRAD_ACC_STEPS=$((TBS/(PER_DEVICE_BATCH_SIZE*NP)))

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

if [ -n "${RUN_NAME_SUFFIX:-}" ]; then
  RUN_NAME=${RUN_NAME}_${RUN_NAME_SUFFIX}
fi


# Run ID
N_VALUES=(1 2 3)
for N in "${N_VALUES[@]}"; do
  # Path to save experiment results
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
    run_gpt2_on_mqar.py
      --exp_path "$EXP_PATH"
      --per_device_batch_size "$PER_DEVICE_BATCH_SIZE"
      --gradient_accumulation_steps "$GRAD_ACC_STEPS"
      --total_batch_size "$TBS"
      --vocab_size "$VOCAB_SIZE"
      --input_seq_len "$INPUT_SEQ_LEN"
      --num_kv_pairs "$NUM_KV_PAIRS"
      --query_sampling "$QUERY_SAMPLING"
      --power_a "$POWER_A"
      --train_num_examples "$TRAIN_NUM_EXAMPLES"
      --valid_num_examples "$VALID_NUM_EXAMPLES"
      --data_seed "$DATA_SEED"
      --learning_rate "$LR"
      --n_layer "$L"
      --n_head "$H"
      --n_embd "$D"
      --max_position_embeddings "$MAX_POSITION_EMBEDDINGS"
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
