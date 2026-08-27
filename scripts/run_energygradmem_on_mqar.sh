#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/collect_env_state.sh"

# Define arguments for the script
RESUME_FROM_CHECKPOINT=${RESUME_FROM_CHECKPOINT:-}
while [[ $# -gt 0 ]]; do
  case "$1" in
    --resume_from_checkpoint|--resume-from-checkpoint)
      if [[ $# -lt 2 ]]; then
        echo "[ERROR] $1 requires a checkpoint directory" >&2
        exit 2
      fi
      RESUME_FROM_CHECKPOINT="$2"
      shift 2
      ;;
    --resume_from_checkpoint=*|--resume-from-checkpoint=*)
      RESUME_FROM_CHECKPOINT="${1#*=}"
      shift
      ;;
    *)
      echo "[ERROR] unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

NP=${NP:-1}
LR=${LR:-1e-04}
ADAM_BETA1=${ADAM_BETA1:-0.9}
ADAM_BETA2=${ADAM_BETA2:-0.999}
TBS=64
PER_DEVICE_BATCH_SIZE=64
GRAD_ACC_STEPS=$(($TBS/($PER_DEVICE_BATCH_SIZE*$NP)))

L=${L:-4}
H=${H:-4}
D=${D:-256}
BASE_MODEL=llama

# Dense MQAR task: 4K framed context tokens + K contiguous queries.
VOCAB_SIZE=8192
NUM_KV_PAIRS=${NUM_KV_PAIRS:-16}
INPUT_SEQ_LEN=$((3 * NUM_KV_PAIRS))
DENSE_QUERIES=true
MQAR_NOISE_LVL=${MQAR_NOISE_LVL:-0.0}
# zoology uses power_law with 0.01
QUERY_SAMPLING=${QUERY_SAMPLING:-uniform}
TRAIN_NUM_EXAMPLES=1000000
VALID_NUM_EXAMPLES=5000
POWER_A=${POWER_A:-0.01}
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
if [ "$MQAR_NOISE_LVL" != "0.0" ]; then
  DATA_NAME="${DATA_NAME}_noise${MQAR_NOISE_LVL}"
fi
MQAR_DATA_PATH=${MQAR_DATA_PATH:-./data/${DATA_NAME}}

# Energy-GradMem currently supports prefix memory only.
MEMORY_BACKEND="prefix"
WRITE_OBJECTIVE="energy"

# Memory/write params. Start from known-good GradMem N8 setup.
N_MEM_TOKENS=8
N_CTRL_TOKENS=0
K=1
LAST_K_SECOND_ORDER=${K}
INNER_LR=5.0
INNER_CLIP_VALUE=None
INNER_CLIP_NORM=10.0
USE_ADAM=false
GRAD_MODE="second"
USE_MEM_PROJ=false
MEM_PROJ_MODE="none"
USE_WRITE_HEAD=true
USE_WRITE_LORA=false
WRITE_LORA_R=${WRITE_LORA_R:-8}
WRITE_LORA_ALPHA=${WRITE_LORA_ALPHA:-16}
WRITE_LORA_DROPOUT=${WRITE_LORA_DROPOUT:-0.0}
WRITE_LORA_TARGETS=${WRITE_LORA_TARGETS:-}
FREEZE_BACKBONE=false

LORA_MEM_PLACEMENT=${LORA_MEM_PLACEMENT:-between_layers}
LORA_MEM_TARGET_MODULES=${LORA_MEM_TARGET_MODULES:-}
LORA_MEM_R=${LORA_MEM_R:-8}
LORA_MEM_ALPHA=${LORA_MEM_ALPHA:-16}
LORA_MEM_DROPOUT=${LORA_MEM_DROPOUT:-0.0}
LORA_MEM_LAYERS=${LORA_MEM_LAYERS:-all}
KV_MEM_LAYERS=${KV_MEM_LAYERS:-all}

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
USE_LAYERWISE_ENERGY=${USE_LAYERWISE_ENERGY:-false}
LIPSCHITZ_WEIGHT=${LIPSCHITZ_WEIGHT:-0.0}
LIPSCHITZ_CONSTRAINT=${LIPSCHITZ_CONSTRAINT:-1.0}
ENERGY_MEMORY_SEARCH_WEIGHT=${ENERGY_MEMORY_SEARCH_WEIGHT:-0.0}
ENERGY_MEMORY_SEARCH_NUM_SAMPLES=${ENERGY_MEMORY_SEARCH_NUM_SAMPLES:-4}
ENERGY_MEMORY_SEARCH_RADIUS_SCALE=${ENERGY_MEMORY_SEARCH_RADIUS_SCALE:-0.25}
ENERGY_MEMORY_SEARCH_USE_GAIN_WEIGHTING=${ENERGY_MEMORY_SEARCH_USE_GAIN_WEIGHTING:-false}
ENERGY_MEMORY_SEARCH_GAIN_EMA_DECAY=${ENERGY_MEMORY_SEARCH_GAIN_EMA_DECAY:-0.99}
ENERGY_MEMORY_SEARCH_MIN_RELATIVE_TARGET_GAIN=${ENERGY_MEMORY_SEARCH_MIN_RELATIVE_TARGET_GAIN:-0.0}
ENERGY_MEMORY_SEARCH_USE_BEST_FOR_NEXT_STEP=${ENERGY_MEMORY_SEARCH_USE_BEST_FOR_NEXT_STEP:-false}
READ_FOCAL_GAMMA=${READ_FOCAL_GAMMA:-0.0}
MEMORY_ALIGNMENT_WEIGHT=${MEMORY_ALIGNMENT_WEIGHT:-0.0}
STEP_ALIGNMENT_WEIGHT=0.0
ALIGN_LAST_STEP=${ALIGN_LAST_STEP:-false}
GRAD_ALIGN_NORM=${GRAD_ALIGN_NORM:-none}
INTERMEDIATE_READ_WEIGHT=0.0
MEMORY_NOISE_SIGMA=${MEMORY_NOISE_SIGMA:-0.0}
ORTHOGONAL_LOSS_WEIGHT=${ORTHOGONAL_LOSS_WEIGHT:-0.0}
IVAN_LOSS_WEIGHT=${IVAN_LOSS_WEIGHT:-0.0}

STOP_ON_METRIC_VALUE=0.99
SAVE_STEPS=${SAVE_STEPS:-}
MAX_STEPS=500000

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

# INIT_CHECKPOINT=/cephfs/home/mkairov/gradim/energy_shaping/runs/mqar_N16_V8192_L48/energygradmem_llama_L4H4D256_mem8_K2_ilr0.4_icn10.0_energy_grad_second_bs_64_lr_1e-04_stepalign0.1_fp32/run_1/checkpoint-131000/model.safetensors
# INIT_CHECKPOINT=/cephfs/home/mkairov/gradim/energy_shaping/runs/mqar_N16_V8192_L48/energygradmem_llama_L4H4D256_mem8_K2_ilr0.4_icn10.0_energy_grad_second_bs_64_lr_1e-04_stepalign0.1_fp32/run_2/checkpoint-53000/model.safetensors
# RUN_NAME_SUFFIX=init_N16

if [ "$WRITE_OBJECTIVE" = "reconstruction" ]; then
  RUN_NAME=gradmem_${BASE_MODEL}_L${L}H${H}D${D}_mem${N_MEM_TOKENS}
else
  RUN_NAME=energygradmem_${BASE_MODEL}_L${L}H${H}D${D}_mem${N_MEM_TOKENS}
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
if [ "$WRITE_OBJECTIVE" != "reconstruction" ]; then
  RUN_NAME=${RUN_NAME}_energy
fi
if [ "$MQAR_NOISE_LVL" != "0.0" ]; then
  RUN_NAME=${RUN_NAME}_mqarnoise${MQAR_NOISE_LVL}
fi
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
if [ "$ADAM_BETA1" != "0.9" ] || [ "$ADAM_BETA2" != "0.999" ]; then
  RUN_NAME=${RUN_NAME}_b1${ADAM_BETA1}_b2${ADAM_BETA2}
fi
if [ "$LIPSCHITZ_WEIGHT" != "0.0" ]; then
  RUN_NAME=${RUN_NAME}_lip${LIPSCHITZ_WEIGHT}_L${LIPSCHITZ_CONSTRAINT}
fi
if [ "$ENERGY_MEMORY_SEARCH_WEIGHT" != "0.0" ]; then
  RUN_NAME=${RUN_NAME}_msearch${ENERGY_MEMORY_SEARCH_WEIGHT}
fi
if [ "$READ_FOCAL_GAMMA" != "0.0" ]; then
  RUN_NAME=${RUN_NAME}_focal${READ_FOCAL_GAMMA}
fi
if [ "$STEP_ALIGNMENT_WEIGHT" != "0.0" ]; then
  RUN_NAME=${RUN_NAME}_stepalign${STEP_ALIGNMENT_WEIGHT}
fi
if [ "$MEMORY_NOISE_SIGMA" != "0.0" ]; then
  RUN_NAME=${RUN_NAME}_mnoise${MEMORY_NOISE_SIGMA}
fi

if [ "$MIXED_PRECISION" == "no" ]; then
  RUN_NAME=${RUN_NAME}_fp32
fi

if [ -n "${RUN_NAME_SUFFIX:-}" ]; then
  RUN_NAME=${RUN_NAME}_${RUN_NAME_SUFFIX}
fi

N_VALUES=(1)
if [ -n "$RESUME_FROM_CHECKPOINT" ]; then
  if [ ! -d "$RESUME_FROM_CHECKPOINT" ]; then
    echo "[ERROR] resume checkpoint directory does not exist: $RESUME_FROM_CHECKPOINT" >&2
    exit 1
  fi
  RESUME_FROM_CHECKPOINT="$(realpath "$RESUME_FROM_CHECKPOINT")"
  RESUME_EXP_PATH="$(dirname "$RESUME_FROM_CHECKPOINT")"
  RESUME_RUN_DIR="$(basename "$RESUME_EXP_PATH")"
  if [[ ! "$RESUME_RUN_DIR" =~ ^run_([0-9]+)(_(bf16|fp16))?$ ]]; then
    echo "[ERROR] could not infer run number from resume path: $RESUME_EXP_PATH" >&2
    exit 1
  fi
  N_VALUES=("${BASH_REMATCH[1]}")
fi
for N in "${N_VALUES[@]}"; do
  EXP_PATH="./runs/${DATA_NAME}/${RUN_NAME}/run_${N}"

  if [ "$MIXED_PRECISION" != "no" ]; then
    EXP_PATH="${EXP_PATH}_${MIXED_PRECISION}"
  fi

  ALLOW_EXISTING=false
  if [ -n "$RESUME_FROM_CHECKPOINT" ]; then
    EXP_PATH="$RESUME_EXP_PATH"
    ALLOW_EXISTING=true
  fi

  if ! prepare_locked_run "$EXP_PATH" "$0" "$NP" "$ALLOW_EXISTING"; then
    continue
  fi

  PORT="$(find_free_port)"

  CMD=(
    accelerate launch
      --main_process_port "$PORT"
      --num_processes "$NP"
      --mixed_precision "$MIXED_PRECISION"
      --config_file accelerate.yaml
    run_gradmemgpt_on_mqar.py
    --exp_path "$EXP_PATH"
    --per_device_batch_size "$PER_DEVICE_BATCH_SIZE"
    --gradient_accumulation_steps "$GRAD_ACC_STEPS"
    --total_batch_size "$TBS"
    --vocab_size "$VOCAB_SIZE"
    --input_seq_len "$INPUT_SEQ_LEN"
    --num_kv_pairs "$NUM_KV_PAIRS"
    --dense_queries "$DENSE_QUERIES"
    --mqar_noise_lvl "$MQAR_NOISE_LVL"
    --mqar_data_path "$MQAR_DATA_PATH"
    --query_sampling "$QUERY_SAMPLING"
    --train_num_examples "$TRAIN_NUM_EXAMPLES"
    --valid_num_examples "$VALID_NUM_EXAMPLES"
    --power_a "$POWER_A"
    --data_seed "$DATA_SEED"
    --learning_rate "$LR"
    --adam_beta1 "$ADAM_BETA1"
    --adam_beta2 "$ADAM_BETA2"
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
    --lipschitz_weight "$LIPSCHITZ_WEIGHT"
    --lipschitz_constraint "$LIPSCHITZ_CONSTRAINT"
    --energy_memory_search_weight "$ENERGY_MEMORY_SEARCH_WEIGHT"
    --energy_memory_search_num_samples "$ENERGY_MEMORY_SEARCH_NUM_SAMPLES"
    --energy_memory_search_radius_scale "$ENERGY_MEMORY_SEARCH_RADIUS_SCALE"
    --energy_memory_search_gain_ema_decay "$ENERGY_MEMORY_SEARCH_GAIN_EMA_DECAY"
    --energy_memory_search_min_relative_target_gain "$ENERGY_MEMORY_SEARCH_MIN_RELATIVE_TARGET_GAIN"
    --read_focal_gamma "$READ_FOCAL_GAMMA"
    --memory_alignment_weight "$MEMORY_ALIGNMENT_WEIGHT"
    --step_alignment_weight "$STEP_ALIGNMENT_WEIGHT"
    --grad_align_norm "$GRAD_ALIGN_NORM"
    --intermediate_read_weight "$INTERMEDIATE_READ_WEIGHT"
    --memory_noise_sigma "$MEMORY_NOISE_SIGMA"
    --orthogonal_loss_weight "$ORTHOGONAL_LOSS_WEIGHT"
    --ivan_loss_weight "$IVAN_LOSS_WEIGHT"
    --max_steps "$MAX_STEPS"
    --eval_steps 500
    --logging_steps 500
    --warmup_steps 10000
    --early_stopping_patience 2000
    --stop_on_metric_value "$STOP_ON_METRIC_VALUE"
    --seed "$((142+$N))"
  )

  if [ -n "${INIT_CHECKPOINT:-}" ] && [ -z "$RESUME_FROM_CHECKPOINT" ]; then
    CMD+=( --init_checkpoint "$INIT_CHECKPOINT" )
  fi
  if [ -n "${INIT_BASE_CHECKPOINT:-}" ]; then
    CMD+=( --init_base_checkpoint "$INIT_BASE_CHECKPOINT" )
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
  if [ "$USE_WRITE_LORA" = true ]; then
    CMD+=( --use_write_lora --write_lora_r "$WRITE_LORA_R" )
    CMD+=( --write_lora_alpha "$WRITE_LORA_ALPHA" --write_lora_dropout "$WRITE_LORA_DROPOUT" )
    if [ -n "$WRITE_LORA_TARGETS" ]; then
      CMD+=( --write_lora_target_modules "$WRITE_LORA_TARGETS" )
    fi
  fi
  if [ "$MEMORY_BACKEND" = "lora" ]; then
    CMD+=( --lora_mem_placement "$LORA_MEM_PLACEMENT" --lora_mem_r "$LORA_MEM_R" )
    CMD+=( --lora_mem_alpha "$LORA_MEM_ALPHA" --lora_mem_dropout "$LORA_MEM_DROPOUT" )
    CMD+=( --lora_mem_layers "$LORA_MEM_LAYERS" --lora_mem_target_modules "$LORA_MEM_TARGET_MODULES" )
  fi
  if [ "$MEMORY_BACKEND" = "kv_cache" ]; then
    CMD+=( --kv_mem_layers "$KV_MEM_LAYERS" )
  fi
  if [ "$USE_LAYERWISE_ENERGY" = true ]; then
    CMD+=( --use_layerwise_energy )
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
  if [ -n "$SAVE_STEPS" ]; then
    CMD+=( --save_steps "$SAVE_STEPS" )
  fi
  if [ -n "$RESUME_FROM_CHECKPOINT" ]; then
    CMD+=( --resume_from_checkpoint "$RESUME_FROM_CHECKPOINT" )
  fi
  if [ "$ADD_INNER_LOSS_TO_OUTER" = true ]; then
    CMD+=( --add_inner_loss_to_outer )
    if [ "$INNER_LOSS_WEIGHT" != "None" ]; then
      CMD+=( --inner_loss_weight "$INNER_LOSS_WEIGHT" )
    fi
  fi
  if [ "$ALIGN_LAST_STEP" = true ]; then
    CMD+=( --align_last_step )
  fi
  if [ "$ENERGY_MEMORY_SEARCH_USE_GAIN_WEIGHTING" = true ]; then
    CMD+=( --energy_memory_search_use_gain_weighting )
  fi
  if [ "$ENERGY_MEMORY_SEARCH_USE_BEST_FOR_NEXT_STEP" = true ]; then
    CMD+=( --energy_memory_search_use_best_for_next_step )
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
