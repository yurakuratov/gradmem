#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/collect_env_state.sh"

# Define arguments for the script
NP=${NP:-1}  # Default to 1 process if not set
LR=${LR:-1e-04}
TBS=64
PER_DEVICE_BATCH_SIZE=64
GRAD_ACC_STEPS=$(($TBS/($PER_DEVICE_BATCH_SIZE*$NP)))

L=4
H=4
D=256
BASE_MODEL=llama

# Dense MQAR query-order distribution.
VOCAB_SIZE=8192
NUM_KV_PAIRS=16
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

# GradMemGPT specific parameters
MEMORY_BACKEND="prefix"

# memory params
# - n_mem_tokens is used by prefix and kv_cache backends.
# - lora backend ignores n_mem_tokens.
N_MEM_TOKENS=8
N_CTRL_TOKENS=0
K=2
LAST_K_SECOND_ORDER=${K}
INNER_LR=0.4
INNER_CLIP_VALUE=None
INNER_CLIP_NORM=None
USE_ADAM=false
GRAD_MODE="second"
USE_MEM_PROJ=false
MEM_PROJ_MODE="none"
USE_WRITE_HEAD=false
USE_WRITE_LORA=false
WRITE_LORA_R=8
WRITE_LORA_ALPHA=16
WRITE_LORA_DROPOUT=0.0
WRITE_LORA_TARGETS=""
FREEZE_BACKBONE=false

# LoRA memory params (active only when MEMORY_BACKEND="lora")
LORA_MEM_PLACEMENT="between_layers"
# "gate_proj,up_proj,down_proj"
LORA_MEM_TARGET_MODULES=""
LORA_MEM_R=8
LORA_MEM_ALPHA=16
LORA_MEM_DROPOUT=0.0
LORA_MEM_LAYERS="all"

# KV-cache memory params (active only when MEMORY_BACKEND="kv_cache")
KV_MEM_LAYERS="all"
# INNER_LR=10.0 for kv_cache backend

# Prefix backend notes (active only when MEMORY_BACKEND="prefix"):
# - Uses N_MEM_TOKENS (+ optional N_CTRL_TOKENS).
# - Does not use LoRA/KV-cache layer settings.
ADD_INNER_LOSS_TO_OUTER=false
INNER_LOSS_WEIGHT=0.5
STOP_ON_METRIC_VALUE=${STOP_ON_METRIC_VALUE:-1.00}
STOP_ON_METRIC_VALUE=0.99
RUN_NAME_SUFFIX=0.99

ATTN_IMPL="eager"
MIXED_PRECISION='no'

# INIT_CHECKPOINT=./runs/N16-K2V2-V62_1M/gradmem_llama_L4H4D128_mem8_K2_ilr0.04_whead_grad_second_bs_64_lr_1e-04/run_1/checkpoint-198000/model.safetensors
# INIT_CHECKPOINT=./runs/N32-K2V2-V62_1M/gradmem_llama_L4H4D128_mem8_K2_ilr0.12_whead_grad_second_bs_64_lr_1e-04/run_1/checkpoint-196500/model.safetensors
# INIT_CHECKPOINT=./runs/N64-K2V2-V62_1M/gradmem_llama_L4H4D128_mem8_K5_ilr0.08_whead_grad_second_bs_64_lr_1e-04_init_N16_K2_ilr0.04/run_2/checkpoint-199000/model.safetensors
# RUN_NAME_SUFFIX=init_N64_K5_ilr0.08
# INIT_CHECKPOINT=./runs/N64-K2V2-V62_1M/gradmem_llama_L4H4D128_mem8_K1_ilr0.08_whead_grad_second_bs_64_lr_1e-04_init_N16_K2_ilr0.04/run_1/checkpoint-196000/model.safetensors
# RUN_NAME_SUFFIX=init_N64_K1_ilr0.08_N16_K2

RUN_NAME=gradmem_${BASE_MODEL}_L${L}H${H}D${D}
if [ "$MEMORY_BACKEND" = "prefix" ]; then
  RUN_NAME=${RUN_NAME}_mem${N_MEM_TOKENS}
fi
if [ "$MEMORY_BACKEND" = "lora" ]; then
  RUN_NAME=${RUN_NAME}_lora_${LORA_MEM_PLACEMENT}_r${LORA_MEM_R}a${LORA_MEM_ALPHA}
  if [ "$LORA_MEM_DROPOUT" != "0.0" ]; then
    RUN_NAME=${RUN_NAME}d${LORA_MEM_DROPOUT}
  fi
  RUN_NAME=${RUN_NAME}_layers_${LORA_MEM_LAYERS}
fi
if [ "$MEMORY_BACKEND" = "kv_cache" ]; then
  RUN_NAME=${RUN_NAME}_kvmem${N_MEM_TOKENS}
  if [ "$KV_MEM_LAYERS" != "all" ]; then
    RUN_NAME=${RUN_NAME}_layers_${KV_MEM_LAYERS}
  fi
fi
if [ "$N_CTRL_TOKENS" -gt 0 ]; then
  RUN_NAME=${RUN_NAME}_c${N_CTRL_TOKENS}
fi
RUN_NAME=${RUN_NAME}_K${K}_ilr${INNER_LR}
if [ "$LAST_K_SECOND_ORDER" != "$K" ] && [ "$GRAD_MODE" == "second" ]; then
  RUN_NAME=${RUN_NAME}_last_K${LAST_K_SECOND_ORDER}
fi
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
if [ "$USE_WRITE_LORA" = true ]; then
  RUN_NAME=${RUN_NAME}_wlora_r${WRITE_LORA_R}a${WRITE_LORA_ALPHA}
  if [ "$WRITE_LORA_DROPOUT" != "0.0" ]; then
    RUN_NAME=${RUN_NAME}d${WRITE_LORA_DROPOUT}
  fi
fi
RUN_NAME=${RUN_NAME}_grad_${GRAD_MODE}
if [ "$ADD_INNER_LOSS_TO_OUTER" = true ]; then
  RUN_NAME=${RUN_NAME}_add_inner
  if [ "$INNER_LOSS_WEIGHT" != "None" ]; then
    RUN_NAME=${RUN_NAME}_w${INNER_LOSS_WEIGHT}
  fi
fi
if [ "$USE_ADAM" = true ]; then
  RUN_NAME=${RUN_NAME}_with_adam
fi
RUN_NAME=${RUN_NAME}_bs_${TBS}_lr_${LR}

if [ "$MIXED_PRECISION" == "no" ]; then
  RUN_NAME=${RUN_NAME}_fp32
fi

if [ -n "${RUN_NAME_SUFFIX:-}" ]; then
  RUN_NAME=${RUN_NAME}_${RUN_NAME_SUFFIX}
fi

# Run ID
N_VALUES=(1 2 3)
for N in "${N_VALUES[@]}"; do
  # Path to save experiment results
  RND=$(date +%Y%m%d%H%M%S)
  EXP_PATH="./runs/${DATA_NAME}/${RUN_NAME}/run_${N}"

  if [ "$MIXED_PRECISION" != "no" ]; then
    EXP_PATH="${EXP_PATH}_${MIXED_PRECISION}"
  fi

  if ! prepare_locked_run "$EXP_PATH" "$0" "$NP"; then
    continue
  fi

  PORT="$(find_free_port)"
  
  # --multi_gpu \
  # --mixed_precision 'bf16' \
  # Execute the script using accelerate for parallel processing
  CMD=(
    accelerate launch
      --main_process_port "$PORT"
      --num_processes "$NP"
      --mixed_precision "$MIXED_PRECISION"
      # --multi_gpu \
      --config_file accelerate.yaml
    run_gradmemgpt_on_mqar.py
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
    --base_model "$BASE_MODEL"
    --n_mem_tokens "$N_MEM_TOKENS"
    --memory_backend "$MEMORY_BACKEND"
    --K "$K"
    --last_K_second_order "$LAST_K_SECOND_ORDER"
    --inner_lr "$INNER_LR"
    --use_adam "$USE_ADAM"
    --grad_mode "$GRAD_MODE"
    --freeze_backbone "$FREEZE_BACKBONE"
    --max_steps 200000
    --eval_steps 500
    --logging_steps 500
    --warmup_steps 10000
    --early_stopping_patience 500
    --stop_on_metric_value "$STOP_ON_METRIC_VALUE"
    --seed "$((142+$N))"
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

  if [ "$USE_WRITE_LORA" = true ]; then
    CMD+=( --use_write_lora )
    CMD+=( --write_lora_r "$WRITE_LORA_R" )
    CMD+=( --write_lora_alpha "$WRITE_LORA_ALPHA" )
    CMD+=( --write_lora_dropout "$WRITE_LORA_DROPOUT" )
    if [ -n "$WRITE_LORA_TARGETS" ]; then
      CMD+=( --write_lora_target_modules "$WRITE_LORA_TARGETS" )
    fi
  fi

  if [ "$MEMORY_BACKEND" = "lora" ]; then
    CMD+=( --lora_mem_placement "$LORA_MEM_PLACEMENT" )
    CMD+=( --lora_mem_target_modules "$LORA_MEM_TARGET_MODULES" )
    CMD+=( --lora_mem_r "$LORA_MEM_R" )
    CMD+=( --lora_mem_alpha "$LORA_MEM_ALPHA" )
    CMD+=( --lora_mem_dropout "$LORA_MEM_DROPOUT" )
    CMD+=( --lora_mem_layers "$LORA_MEM_LAYERS" )
  fi

  if [ "$MEMORY_BACKEND" = "kv_cache" ]; then
    CMD+=( --kv_mem_layers "$KV_MEM_LAYERS" )
  fi

  if [ "$ADD_INNER_LOSS_TO_OUTER" = true ]; then
    CMD+=( --add_inner_loss_to_outer )
    if [ "$INNER_LOSS_WEIGHT" != "None" ]; then
      CMD+=( --inner_loss_weight "$INNER_LOSS_WEIGHT" )
    fi
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

echo "Done"
