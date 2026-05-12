#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/collect_env_state.sh"

# Define arguments for the script
NP=${NP:-1}
LR=1e-04
TBS=64
PER_DEVICE_BATCH_SIZE=32
GRAD_ACC_STEPS=$(($TBS/($PER_DEVICE_BATCH_SIZE*$NP)))

L=4
H=4
D=128
BASE_MODEL=llama

V=62
DATA_NAME="N64-K2V2-V${V}_1M"
DATA_PATH="./data/${DATA_NAME}"
TOKENIZER_PATH="./tokenizers/kv_alphabet_${V}/"

CHUNK_SIZE=7
CONCAT_CHUNK_MEMORY=false
PARALLEL_CHUNKS=false

MEMORY_BACKEND="lora"

# Shared memory params
# - n_mem_tokens is used by prefix and kv_cache backends.
# - lora backend ignores n_mem_tokens.

N_MEM_TOKENS=1
N_CTRL_TOKENS=0
K=1
LAST_K_SECOND_ORDER=${K}
INNER_LR=0.04
INNER_CLIP_VALUE=None
INNER_CLIP_NORM=None
USE_ADAM=false
GRAD_MODE="second"
USE_MEM_PROJ=false
MEM_PROJ_MODE="none"
USE_WRITE_HEAD=true
USE_WRITE_LORA=false
WRITE_LORA_R=8
WRITE_LORA_ALPHA=16
WRITE_LORA_DROPOUT=0.0
WRITE_LORA_TARGETS=""
FREEZE_BACKBONE=false

# LoRA memory params (active only when MEMORY_BACKEND="lora")
# LORA_MEM_PLACEMENT="between_layers"
LORA_MEM_PLACEMENT="target_modules"
LORA_MEM_TARGET_MODULES="gate_proj,up_proj,down_proj"
LORA_MEM_R=8
LORA_MEM_ALPHA=16
LORA_MEM_DROPOUT=0.0
LORA_MEM_LAYERS="all"

# KV-cache memory params (active only when MEMORY_BACKEND="kv_cache")
KV_MEM_LAYERS="all"

ADD_INNER_LOSS_TO_OUTER=false
INNER_LOSS_WEIGHT=None

# INIT_CHECKPOINT="./runs/N16-K2V2-V62_1M/gradrmt_llama_L4H4D128_chunk7_lora_target_modules_r8a16_gate_proj_up_proj_down_proj_layers_all_K1_ilr0.04_whead_grad_second_bs_64_lr_1e-04/run_1_bf16/checkpoint-13000/model.safetensors"
# RUN_NAME_SUFFIX="init_N16"
INIT_CHECKPOINT="./runs/N32-K2V2-V62_1M/gradrmt_llama_L4H4D128_chunk7_lora_target_modules_r8a16_gate_proj_up_proj_down_proj_layers_all_K1_ilr0.04_whead_grad_second_bs_64_lr_1e-04_init_N16/run_1_bf16/checkpoint-29500/model.safetensors"
RUN_NAME_SUFFIX="init_N32"

ATTN_IMPL="eager"
MIXED_PRECISION='bf16'

RUN_NAME=gradrmt_${BASE_MODEL}_L${L}H${H}D${D}_chunk${CHUNK_SIZE}

if [ "$MEMORY_BACKEND" = "prefix" ]; then
  RUN_NAME=${RUN_NAME}_mem${N_MEM_TOKENS}
fi
if [ "$MEMORY_BACKEND" = "lora" ]; then
  RUN_NAME=${RUN_NAME}_lora_${LORA_MEM_PLACEMENT}_r${LORA_MEM_R}a${LORA_MEM_ALPHA}
  if [ "$LORA_MEM_DROPOUT" != "0.0" ]; then
    RUN_NAME=${RUN_NAME}d${LORA_MEM_DROPOUT}
  fi
  if [ -n "${LORA_MEM_TARGET_MODULES:-}" ]; then
    # replace commas with underscores
    LORA_MEM_TARGET_MODULES_STR=$(echo $LORA_MEM_TARGET_MODULES | tr ',' '_')
    RUN_NAME=${RUN_NAME}_${LORA_MEM_TARGET_MODULES_STR}
  fi
  RUN_NAME=${RUN_NAME}_layers_${LORA_MEM_LAYERS}
fi
if [ "$MEMORY_BACKEND" = "kv_cache" ]; then
  RUN_NAME=${RUN_NAME}_kvmem${N_MEM_TOKENS}
  if [ "$KV_MEM_LAYERS" != "all" ]; then
    RUN_NAME=${RUN_NAME}_layers_${KV_MEM_LAYERS}
  fi
fi
if [ "$CONCAT_CHUNK_MEMORY" = true ]; then
  RUN_NAME=${RUN_NAME}_cat
fi
if [ "$PARALLEL_CHUNKS" = true ]; then
  RUN_NAME=${RUN_NAME}_prl
fi
if [ "$N_CTRL_TOKENS" -gt 0 ]; then
  RUN_NAME=${RUN_NAME}_c${N_CTRL_TOKENS}
fi
RUN_NAME=${RUN_NAME}_K${K}_ilr${INNER_LR}

if [ "$USE_MEM_PROJ" = true ]; then
  RUN_NAME=${RUN_NAME}_mem_proj
fi
if [ "$USE_WRITE_HEAD" = true ]; then
  RUN_NAME=${RUN_NAME}_whead
fi
if [ "$USE_WRITE_LORA" = true ]; then
  RUN_NAME=${RUN_NAME}_wlora_r${WRITE_LORA_R}a${WRITE_LORA_ALPHA}
  if [ "$WRITE_LORA_DROPOUT" != "0.0" ]; then
    RUN_NAME=${RUN_NAME}_wd${WRITE_LORA_DROPOUT}
  fi
fi
if [ "$FREEZE_BACKBONE" = true ]; then
  RUN_NAME=${RUN_NAME}_frozen
fi
RUN_NAME=${RUN_NAME}_grad_${GRAD_MODE}_bs_${TBS}_lr_${LR}

if [ "$MIXED_PRECISION" == "no" ]; then
  RUN_NAME=${RUN_NAME}_fp32
fi

if [ -n "${RUN_NAME_SUFFIX:-}" ]; then
  RUN_NAME=${RUN_NAME}_${RUN_NAME_SUFFIX}
fi

# Run ID
N_VALUES=(1 2 3)
for N in "${N_VALUES[@]}"; do
  RND=$(date +%Y%m%d%H%M%S)
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
      # --multi_gpu \
      --config_file accelerate.yaml
    run_gradrmt_on_kv_retrieval.py
    --exp_path "$EXP_PATH"
    --per_device_batch_size "$PER_DEVICE_BATCH_SIZE"
    --gradient_accumulation_steps "$GRAD_ACC_STEPS"
    --total_batch_size "$TBS"
    --data_path "$DATA_PATH"
    --tokenizer_path "$TOKENIZER_PATH"
    --learning_rate "$LR"
    --n_layer $L \
    --n_head $H \
    --n_embd $D \
    --base_model $BASE_MODEL \
    --n_mem_tokens "$N_MEM_TOKENS"
    --memory_backend "$MEMORY_BACKEND"
    --K "$K"
    --last_K_second_order "$LAST_K_SECOND_ORDER"
    --inner_lr "$INNER_LR"
    --use_adam "$USE_ADAM"
    --grad_mode "$GRAD_MODE"
    --freeze_backbone "$FREEZE_BACKBONE"
    --chunk_size "$CHUNK_SIZE"
    --max_steps 100000
    --eval_steps 500
    --logging_steps 500
    --warmup_steps 1000
    --early_stopping_patience 500
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
    if [ -n "${LORA_MEM_TARGET_MODULES:-}" ]; then
      CMD+=( --lora_mem_target_modules "$LORA_MEM_TARGET_MODULES" )
    fi
    CMD+=( --lora_mem_r "$LORA_MEM_R" )
    CMD+=( --lora_mem_alpha "$LORA_MEM_ALPHA" )
    CMD+=( --lora_mem_dropout "$LORA_MEM_DROPOUT" )
    CMD+=( --lora_mem_layers "$LORA_MEM_LAYERS" )
  fi

  if [ "$MEMORY_BACKEND" = "kv_cache" ]; then
    CMD+=( --kv_mem_layers "$KV_MEM_LAYERS" )
  fi

  if [ "$CONCAT_CHUNK_MEMORY" = true ]; then
    CMD+=( --concat_chunk_memory )
  fi
  if [ "$PARALLEL_CHUNKS" = true ]; then
    CMD+=( --parallel_chunks )
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
