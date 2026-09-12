#!/usr/bin/env bash

# Shared implementation for the three Zoology multihop curriculum entry points.
# This file is sourced after SEGMENT_REGIME and the three curriculum arrays are
# defined; it is not itself an experiment entry point.

if [ "${BASH_SOURCE[0]}" = "$0" ]; then
  echo "Source this helper from a Zoology curriculum entry point." >&2
  exit 2
fi

REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$REPO_ROOT"

PYTHON_BIN=${PYTHON_BIN:-python}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export WANDB_PROJECT=${WANDB_PROJECT:-gradmem_zoology_multihop}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}

if [ ${#CURRICULUM_N[@]} -eq 0 ]; then
  echo "CURRICULUM_N must not be empty" >&2
  exit 1
fi
if [ ${#CE_WEIGHTS[@]} -ne ${#CURRICULUM_N[@]} ]; then
  echo "CE_WEIGHTS must have one value per curriculum stage" >&2
  exit 1
fi
if [ ${#INNER_LRS[@]} -ne ${#CURRICULUM_N[@]} ]; then
  echo "INNER_LRS must have one value per curriculum stage" >&2
  exit 1
fi
if [ "$SEGMENT_REGIME" != "8" ] && [ "$SEGMENT_REGIME" != "32" ] \
  && [ "$SEGMENT_REGIME" != "all" ]; then
  echo "SEGMENT_REGIME must be 8, 32, or all; got $SEGMENT_REGIME" >&2
  exit 1
fi

NP=${NP:-$(($(tr -cd ',' <<< "$CUDA_VISIBLE_DEVICES" | wc -c) + 1))}
TBS=${TBS:-64}
PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE:-64}
if [ $((PER_DEVICE_BATCH_SIZE * NP)) -gt "$TBS" ] \
  || [ $((TBS % (PER_DEVICE_BATCH_SIZE * NP))) -ne 0 ]; then
  echo "TBS must be divisible by PER_DEVICE_BATCH_SIZE * NP" >&2
  exit 1
fi
GRAD_ACC_STEPS=${GRAD_ACC_STEPS:-$((TBS / (PER_DEVICE_BATCH_SIZE * NP)))}
MIXED_PRECISION=${MIXED_PRECISION:-no}

# Small Zoology-compatible decoder.
L=${L:-2}
N_HEAD=${N_HEAD:-1}
D=${D:-128}
N_MEM_TOKENS=${N_MEM_TOKENS:-4}
ATTENTION_DROPOUT=${ATTENTION_DROPOUT:-0.1}

# Established EnergyGradMem curriculum defaults.
K=${K:-2}
LAST_K_SECOND_ORDER=${LAST_K_SECOND_ORDER:-$K}
GRAD_MODE=${GRAD_MODE:-second}
INNER_CLIP_NORM=${INNER_CLIP_NORM:-1.0}
ENERGY_MODEL_TYPE=${ENERGY_MODEL_TYPE:-segment_delta_gru}
ENERGY_HIDDEN_SIZE=${ENERGY_HIDDEN_SIZE:-$D}
ENERGY_SEGMENT_STATE_SIZE=${ENERGY_SEGMENT_STATE_SIZE:-$D}
ENERGY_WEIGHT_RMS_REG=${ENERGY_WEIGHT_RMS_REG:-0.1}
ENERGY_WEIGHT_RMS_THRESHOLD=${ENERGY_WEIGHT_RMS_THRESHOLD:-$(awk "BEGIN { print sqrt($D) / 2 }")}
ENERGY_DELTA_REG=${ENERGY_DELTA_REG:-0.1}
ENERGY_DELTA_MAX=${ENERGY_DELTA_MAX:-1.0}
ENERGY_REPLAY_WEIGHT=0.0
READING_OPTIMIZATION=false

LR=${LR:-1e-4}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.0}
LR_SCHEDULER_TYPE=${LR_SCHEDULER_TYPE:-constant_with_warmup}
MAX_STEPS=${MAX_STEPS:-50000}
EVAL_STEPS=${EVAL_STEPS:-100}
LOGGING_STEPS=${LOGGING_STEPS:-100}
WARMUP_STEPS=${WARMUP_STEPS:-1000}
EARLY_STOPPING_PATIENCE=${EARLY_STOPPING_PATIENCE:-500}
STOP_EXACT_MATCH_VALUE=${STOP_EXACT_MATCH_VALUE:-0.99}
DATALOADER_NUM_WORKERS=${DATALOADER_NUM_WORKERS:-4}
HF_DATASET=${HF_DATASET:-irodkin/zoology_multihop}
OUTPUT_ROOT=${OUTPUT_ROOT:-$REPO_ROOT/runs/energy_gradmem_zoology_mh_curriculum}
START_STAGE=${START_STAGE:-1}
RUN_NUMBERS=${RUN_NUMBERS:-"1 2 3"}

if ! [[ "$START_STAGE" =~ ^[0-9]+$ ]] || [ "$START_STAGE" -lt 1 ]; then
  echo "START_STAGE must be a positive 1-based integer" >&2
  exit 1
fi

resolve_progression_checkpoint() {
  local stage_path=$1
  "$PYTHON_BIN" - "$stage_path" <<'PY'
import json
import sys
from pathlib import Path

stage_path = Path(sys.argv[1])
progression_model = stage_path / "progression_checkpoint" / "model.safetensors"
if progression_model.is_file():
    print(progression_model.resolve())
    raise SystemExit(0)

state_path = stage_path / "trainer_state.json"
if not state_path.is_file():
    raise SystemExit(f"missing trainer state: {state_path}")
state = json.loads(state_path.read_text(encoding="utf-8"))
checkpoint = state.get("best_model_checkpoint")
if not checkpoint:
    raise SystemExit(f"trainer state has no best_model_checkpoint: {state_path}")
checkpoint_path = Path(checkpoint)
if not checkpoint_path.is_absolute():
    checkpoint_path = Path.cwd() / checkpoint_path
model_path = checkpoint_path / "model.safetensors"
if not model_path.is_file():
    raise SystemExit(f"best checkpoint is incomplete or missing: {model_path}")
print(model_path)
PY
}

case "$SEGMENT_REGIME" in
  8) PORT_OFFSET=80 ;;
  32) PORT_OFFSET=320 ;;
  all) PORT_OFFSET=640 ;;
esac

for RUN_ID in $RUN_NUMBERS; do
  if ! [[ "$RUN_ID" =~ ^[0-9]+$ ]]; then
    echo "RUN_NUMBERS must contain non-negative integers; got $RUN_ID" >&2
    exit 1
  fi
  RUN_SEED=$((RUN_ID + 42))
  INIT_CKPT=""

  for STAGE_INDEX in "${!CURRICULUM_N[@]}"; do
    STAGE=$((STAGE_INDEX + 1))
    DATASET_N=${CURRICULUM_N[$STAGE_INDEX]}
    CE_WEIGHT=${CE_WEIGHTS[$STAGE_INDEX]}
    INNER_LR=${INNER_LRS[$STAGE_INDEX]}
    HF_SUBSET=N${DATASET_N}-H1-V4096

    if [ "$SEGMENT_REGIME" = "all" ] || [ "$SEGMENT_REGIME" -gt "$DATASET_N" ]; then
      KV_PAIRS_PER_SEGMENT=$DATASET_N
    else
      KV_PAIRS_PER_SEGMENT=$SEGMENT_REGIME
    fi
    SEGMENT_COUNT=$(((DATASET_N + KV_PAIRS_PER_SEGMENT - 1) / KV_PAIRS_PER_SEGMENT))

    RUN_NAME=energy_gradmem_zoology_mh_seg${SEGMENT_REGIME}_llama_L${L}H${N_HEAD}D${D}_attndrop${ATTENTION_DROPOUT}_${HF_SUBSET}_mem${N_MEM_TOKENS}_K${K}_ilr${INNER_LR}_ce${CE_WEIGHT}_grad_${GRAD_MODE}_energy_${ENERGY_MODEL_TYPE}_state${ENERGY_SEGMENT_STATE_SIZE}_replay0
    STAGE_PATH=$OUTPUT_ROOT/seg${SEGMENT_REGIME}/${HF_SUBSET}/${RUN_NAME}/run_${RUN_ID}/stage_${STAGE}
    WANDB_NAME=energy_gradmem_zoology_mh_seg${SEGMENT_REGIME}_N${DATASET_N}_${SEGMENT_COUNT}segments_ce${CE_WEIGHT}_ilr${INNER_LR}_run${RUN_ID}

    if [ "$STAGE" -lt "$START_STAGE" ]; then
      echo "Skipping stage $STAGE; resolving its best checkpoint from $STAGE_PATH"
    else
      INIT_ARGS=()
      if [ -n "$INIT_CKPT" ]; then
        INIT_ARGS=(--init_checkpoint "$INIT_CKPT")
      fi
      PORT=$((29500 + PORT_OFFSET + RUN_ID))
      echo "Starting seg=$SEGMENT_REGIME run=$RUN_ID stage=$STAGE N=$DATASET_N segments=$SEGMENT_COUNT CE=$CE_WEIGHT inner_lr=$INNER_LR"
      WANDB_NAME="$WANDB_NAME" "$PYTHON_BIN" -m accelerate.commands.launch \
        --main_process_port "$PORT" \
        --num_processes "$NP" \
        --mixed_precision "$MIXED_PRECISION" \
        --config_file "$REPO_ROOT/accelerate.yaml" \
        "$REPO_ROOT/run_energy_gradmem_on_zoology_multihop.py" \
        --exp_path "$STAGE_PATH" \
        --per_device_batch_size "$PER_DEVICE_BATCH_SIZE" \
        --gradient_accumulation_steps "$GRAD_ACC_STEPS" \
        --total_batch_size "$TBS" \
        --hf_dataset "$HF_DATASET" \
        --hf_subset "$HF_SUBSET" \
        --n_pairs "$DATASET_N" \
        --hop_length 1 \
        --kv_pairs_per_segment "$KV_PAIRS_PER_SEGMENT" \
        --vocab_size 4096 \
        --base_model llama \
        --n_layer "$L" \
        --n_head "$N_HEAD" \
        --n_embd "$D" \
        --attention_dropout "$ATTENTION_DROPOUT" \
        --memory_backend prefix \
        --n_mem_tokens "$N_MEM_TOKENS" \
        --K "$K" \
        --last_K_second_order "$LAST_K_SECOND_ORDER" \
        --inner_lr "$INNER_LR" \
        --use_adam false \
        --grad_mode "$GRAD_MODE" \
        --inner_clip_norm "$INNER_CLIP_NORM" \
        --segment_write_mode sequential \
        --memory_rotation none \
        --inner_objective neural \
        --energy_future_mode none \
        --energy_inner_ce_weight "$CE_WEIGHT" \
        --energy_ce_guidance false \
        --energy_model_type "$ENERGY_MODEL_TYPE" \
        --energy_hidden_size "$ENERGY_HIDDEN_SIZE" \
        --energy_segment_state_size "$ENERGY_SEGMENT_STATE_SIZE" \
        --energy_weight_rms_reg "$ENERGY_WEIGHT_RMS_REG" \
        --energy_weight_rms_threshold "$ENERGY_WEIGHT_RMS_THRESHOLD" \
        --energy_delta_reg "$ENERGY_DELTA_REG" \
        --energy_delta_max "$ENERGY_DELTA_MAX" \
        --energy_replay_weight "$ENERGY_REPLAY_WEIGHT" \
        --energy_pretrain_steps 0 \
        --energy_freezed_steps 0 \
        --reading_optimization "$READING_OPTIMIZATION" \
        --learning_rate "$LR" \
        --weight_decay "$WEIGHT_DECAY" \
        --lr_scheduler_type "$LR_SCHEDULER_TYPE" \
        --metric_for_best_model query_accuracy \
        --stop_exact_match_value "$STOP_EXACT_MATCH_VALUE" \
        --max_steps "$MAX_STEPS" \
        --eval_steps "$EVAL_STEPS" \
        --logging_steps "$LOGGING_STEPS" \
        --warmup_steps "$WARMUP_STEPS" \
        --early_stopping_patience "$EARLY_STOPPING_PATIENCE" \
        --dataloader_num_workers "$DATALOADER_NUM_WORKERS" \
        --seed "$RUN_SEED" \
        "${INIT_ARGS[@]}"
    fi

    if ! INIT_CKPT=$(resolve_progression_checkpoint "$STAGE_PATH"); then
      echo "Could not resolve a complete best checkpoint for stage $STAGE" >&2
      exit 1
    fi
  done
done
