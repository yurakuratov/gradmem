#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/collect_env_state.sh"

# Define arguments for the script
NP=${NP:-1}
LR=1e-04
TBS=64
PER_DEVICE_BATCH_SIZE=64
GRAD_ACC_STEPS=$(($TBS/($PER_DEVICE_BATCH_SIZE*$NP)))

L=4
H=4
D=128
BASE_MODEL=llama

V=62
DATA_NAME="N8-K2V2-V${V}_1M"
DATA_PATH="./data/${DATA_NAME}"
TOKENIZER_PATH="./tokenizers/kv_alphabet_${V}/"

# Energy-GradMem currently supports prefix memory only.
MEMORY_BACKEND="prefix"
WRITE_OBJECTIVE=${WRITE_OBJECTIVE:-"energy"}

# Memory/write params. Start from known-good GradMem N8 setup.
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
FREEZE_BACKBONE=false
MEMORY_ALIGNMENT_WEIGHT=${MEMORY_ALIGNMENT_WEIGHT:-0.0}
STEP_ALIGNMENT_WEIGHT=${STEP_ALIGNMENT_WEIGHT:-0.0}
GRAD_ALIGN_NORM=${GRAD_ALIGN_NORM:-none}
INTERMEDIATE_READ_WEIGHT=${INTERMEDIATE_READ_WEIGHT:-0.0}
ORTHOGONAL_LOSS_WEIGHT=${ORTHOGONAL_LOSS_WEIGHT:-0.0}

# Energy head and optional landscape-shaping losses. Environment overrides let
# dedicated experiment wrappers reuse this launcher without duplicating it.
ENERGY_HEAD_HIDDEN_DIM=None
WRITE_RECONSTRUCTION_WEIGHT=1.0
WRITE_ENERGY_WEIGHT=1.0
ENERGY_RANK_WEIGHT=${ENERGY_RANK_WEIGHT:-0.0}
ENERGY_TRAJ_WEIGHT=${ENERGY_TRAJ_WEIGHT:-0.0}
ENERGY_MARGIN=${ENERGY_MARGIN:-0.1}
ENERGY_TRAJ_MARGIN=${ENERGY_TRAJ_MARGIN:-0.0}
ENERGY_RANK_TEMPERATURE=${ENERGY_RANK_TEMPERATURE:-1.0}
ENERGY_MIX_ALPHA=${ENERGY_MIX_ALPHA:-0.75}
ENERGY_ANCHOR_WEIGHT=${ENERGY_ANCHOR_WEIGHT:-0.0}
ENERGY_MEMORY_SEARCH_WEIGHT=${ENERGY_MEMORY_SEARCH_WEIGHT:-0.0}
ENERGY_MEMORY_SEARCH_NUM_SAMPLES=${ENERGY_MEMORY_SEARCH_NUM_SAMPLES:-4}
ENERGY_MEMORY_SEARCH_RADIUS_SCALE=${ENERGY_MEMORY_SEARCH_RADIUS_SCALE:-0.25}
ENERGY_MEMORY_SEARCH_USE_GAIN_WEIGHTING=${ENERGY_MEMORY_SEARCH_USE_GAIN_WEIGHTING:-false}
ENERGY_MEMORY_SEARCH_GAIN_EMA_DECAY=${ENERGY_MEMORY_SEARCH_GAIN_EMA_DECAY:-0.99}

STOP_ON_METRIC_VALUE=0.99

ADD_INNER_LOSS_TO_OUTER=false
INNER_LOSS_WEIGHT=0.5

ATTN_IMPL="eager"
MIXED_PRECISION='no'

# INIT_CHECKPOINT=./runs/N8-K2V2-V62_1M/energygradmem_llama_L4H4D128_mem8_K2_ilr0.4_energy_recon1.0_energy1.0_grad_second_bs_64_lr_1e-04_fp32/run_1/checkpoint-15000/model.safetensors
# RUN_NAME_SUFFIX=init_N8_K2_ilr0.4_recon1.0_energy1.0

# INIT_CHECKPOINT=runs/N8-K2V2-V62_1M/energygradmem_llama_L4H4D128_mem8_K2_ilr0.4_energy_recon1.0_energy1.0_rank0.01_m0.1_t1.0_mix0.75_anchor0.001_grad_second_bs_64_lr_1e-04_fp32/run_1/checkpoint-15500/model.safetensors
# RUN_NAME_SUFFIX=init_N8_K2_ilr0.4_recon1.0_energy1.0_shaping
# INIT_CHECKPOINT=./runs/N8-K2V2-V62_1M/gradmem_llama_L4H4D128_mem8_K2_ilr0.4_grad_second_bs_64_lr_1e-04_fp32/run_1/checkpoint-12500/model.safetensors
# INIT_CHECKPOINT=./runs/N8-K2V2-V62_1M/gradmem_llama_L4H4D128_mem8_K2_ilr0.4_grad_second_bs_64_lr_1e-04_fp32/run_2/checkpoint-14500/model.safetensors
# RUN_NAME_SUFFIX=init_gradmem_N8_K2_ilr0.4

RUN_NAME=energygradmem_${BASE_MODEL}_L${L}H${H}D${D}_mem${N_MEM_TOKENS}
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
RUN_NAME=${RUN_NAME}_energy
if [ "$WRITE_OBJECTIVE" = "energy_with_reconstruction" ]; then
  RUN_NAME=${RUN_NAME}_recon${WRITE_RECONSTRUCTION_WEIGHT}_energy${WRITE_ENERGY_WEIGHT}
fi
if [ "$ENERGY_HEAD_HIDDEN_DIM" != "None" ]; then
  RUN_NAME=${RUN_NAME}_eh${ENERGY_HEAD_HIDDEN_DIM}
fi
if [ "$ENERGY_RANK_WEIGHT" != "0.0" ]; then
  RUN_NAME=${RUN_NAME}_rank${ENERGY_RANK_WEIGHT}_m${ENERGY_MARGIN}
  RUN_NAME=${RUN_NAME}_t${ENERGY_RANK_TEMPERATURE}_mix${ENERGY_MIX_ALPHA}
fi
if [ "$ENERGY_TRAJ_WEIGHT" != "0.0" ]; then
  RUN_NAME=${RUN_NAME}_traj${ENERGY_TRAJ_WEIGHT}_m${ENERGY_TRAJ_MARGIN}
fi
if [ "$ENERGY_ANCHOR_WEIGHT" != "0.0" ]; then
  RUN_NAME=${RUN_NAME}_anchor${ENERGY_ANCHOR_WEIGHT}
fi
if [ "$ENERGY_MEMORY_SEARCH_WEIGHT" != "0.0" ]; then
  RUN_NAME=${RUN_NAME}_msearch${ENERGY_MEMORY_SEARCH_WEIGHT}_n${ENERGY_MEMORY_SEARCH_NUM_SAMPLES}
  if [ "$ENERGY_MEMORY_SEARCH_USE_GAIN_WEIGHTING" = true ]; then
    RUN_NAME=${RUN_NAME}_gainema${ENERGY_MEMORY_SEARCH_GAIN_EMA_DECAY}
  fi
fi
RUN_NAME=${RUN_NAME}_grad_${GRAD_MODE}
if [ "$ADD_INNER_LOSS_TO_OUTER" = true ]; then
  RUN_NAME=${RUN_NAME}_add_inner
  if [ "$INNER_LOSS_WEIGHT" != "None" ]; then
    RUN_NAME=${RUN_NAME}_w${INNER_LOSS_WEIGHT}
  fi
fi
if [ "$MEMORY_ALIGNMENT_WEIGHT" != "0.0" ]; then
  RUN_NAME=${RUN_NAME}_align${MEMORY_ALIGNMENT_WEIGHT}
fi
if [ "$STEP_ALIGNMENT_WEIGHT" != "0.0" ]; then
  RUN_NAME=${RUN_NAME}_stepalign${STEP_ALIGNMENT_WEIGHT}_${GRAD_ALIGN_NORM}
fi
if [ "$INTERMEDIATE_READ_WEIGHT" != "0.0" ]; then
  RUN_NAME=${RUN_NAME}_iread${INTERMEDIATE_READ_WEIGHT}
fi
if [ "$ORTHOGONAL_LOSS_WEIGHT" != "0.0" ]; then
  RUN_NAME=${RUN_NAME}_orth${ORTHOGONAL_LOSS_WEIGHT}
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

N_VALUES=(2)
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
    run_gradmemgpt_on_kv_retrieval.py
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
    --n_mem_tokens "$N_MEM_TOKENS"
    --n_ctrl_tokens "$N_CTRL_TOKENS"
    --memory_backend "$MEMORY_BACKEND"
    --write_objective "$WRITE_OBJECTIVE"
    --K "$K"
    --last_K_second_order "$LAST_K_SECOND_ORDER"
    --inner_lr "$INNER_LR"
    --use_adam "$USE_ADAM"
    --grad_mode "$GRAD_MODE"
    --memory_alignment_weight "$MEMORY_ALIGNMENT_WEIGHT"
    --step_alignment_weight "$STEP_ALIGNMENT_WEIGHT"
    --grad_align_norm "$GRAD_ALIGN_NORM"
    --intermediate_read_weight "$INTERMEDIATE_READ_WEIGHT"
    --orthogonal_loss_weight "$ORTHOGONAL_LOSS_WEIGHT"
    --freeze_backbone "$FREEZE_BACKBONE"
    --energy_rank_weight "$ENERGY_RANK_WEIGHT"
    --energy_traj_weight "$ENERGY_TRAJ_WEIGHT"
    --write_reconstruction_weight "$WRITE_RECONSTRUCTION_WEIGHT"
    --write_energy_weight "$WRITE_ENERGY_WEIGHT"
    --energy_margin "$ENERGY_MARGIN"
    --energy_traj_margin "$ENERGY_TRAJ_MARGIN"
    --energy_rank_temperature "$ENERGY_RANK_TEMPERATURE"
    --energy_mix_alpha "$ENERGY_MIX_ALPHA"
    --energy_anchor_weight "$ENERGY_ANCHOR_WEIGHT"
    --energy_memory_search_weight "$ENERGY_MEMORY_SEARCH_WEIGHT"
    --energy_memory_search_num_samples "$ENERGY_MEMORY_SEARCH_NUM_SAMPLES"
    --energy_memory_search_radius_scale "$ENERGY_MEMORY_SEARCH_RADIUS_SCALE"
    --energy_memory_search_gain_ema_decay "$ENERGY_MEMORY_SEARCH_GAIN_EMA_DECAY"
    --max_steps 200000
    --eval_steps 500
    --logging_steps 500
    --warmup_steps 10000
    --early_stopping_patience 500
    --stop_on_metric_value "$STOP_ON_METRIC_VALUE"
    --seed "$((142+$N))"
  )

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
  if [ -n "${ATTN_IMPL:-}" ]; then
    CMD+=( --attn_implementation "$ATTN_IMPL" )
  fi
  if [ "$ENERGY_HEAD_HIDDEN_DIM" != "None" ]; then
    CMD+=( --energy_head_hidden_dim "$ENERGY_HEAD_HIDDEN_DIM" )
  fi
  if [ -n "${MAX_CONTEXT_LENGTH:-}" ]; then
    CMD+=( --max_context_length "$MAX_CONTEXT_LENGTH" )
  fi
  if [ "$ADD_INNER_LOSS_TO_OUTER" = true ]; then
    CMD+=( --add_inner_loss_to_outer )
    if [ "$INNER_LOSS_WEIGHT" != "None" ]; then
      CMD+=( --inner_loss_weight "$INNER_LOSS_WEIGHT" )
    fi
  fi
  if [ "$ENERGY_MEMORY_SEARCH_USE_GAIN_WEIGHTING" = true ]; then
    CMD+=( --energy_memory_search_use_gain_weighting )
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
