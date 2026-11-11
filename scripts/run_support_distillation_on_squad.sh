#!/bin/bash
# Standalone launcher. Every runner argument is also accepted as a CLI override:
# bash scripts/run_support_distillation_on_squad.sh --n_support_questions 3 --K 2
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
NP=${NP:-1}
if [ "$NP" != 1 ]; then
  echo 'This launcher is configured for one GPU; configure distributed training explicitly.' >&2
  exit 1
fi
PRETRAINED_MODEL=${PRETRAINED_MODEL-unsloth/Llama-3.2-1B}
BASE_MODEL=${BASE_MODEL:-llama}
N_SUPPORT_QUESTIONS=${N_SUPPORT_QUESTIONS:-2}
K=${K:-3}
INNER_LR=${INNER_LR:-0.01}
INNER_RECONSTRUCTION_WEIGHT=${INNER_RECONSTRUCTION_WEIGHT:-0.0}
N_MEM_TOKENS=${N_MEM_TOKENS:-8}
SEED=${SEED:-142}
MIXED_PRECISION=${MIXED_PRECISION:-bf16}
PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE:-1}
GRAD_ACC_STEPS=${GRAD_ACC_STEPS:-1}
# Unique paths prevent collisions even for arguments supplied through "$@".
RUN_NAME=${RUN_NAME:-supportkl_n${N_SUPPORT_QUESTIONS}_mem${N_MEM_TOKENS}_K${K}_ilr${INNER_LR}_ce${INNER_RECONSTRUCTION_WEIGHT}_s${SEED}_$(date +%Y%m%d-%H%M%S)_$$}
EXP_PATH=${EXP_PATH:-runs/squad/$RUN_NAME}
if [ -e "$EXP_PATH" ]; then
  echo "Refusing to overwrite $EXP_PATH. Resume by invoking the Python runner explicitly." >&2
  exit 1
fi
CMD=(run_support_distillation_on_squad.py
  --exp_path "$EXP_PATH"
  --per_device_batch_size "$PER_DEVICE_BATCH_SIZE"
  --gradient_accumulation_steps "$GRAD_ACC_STEPS"
  --dataset_name squad --data_path "${DATA_PATH:-./data/squad}"
  --tokenizer_path "${TOKENIZER_PATH:-unsloth/Llama-3.2-1B}"
  --n_layer "${L:-4}" --n_head "${H:-4}" --n_embd "${D:-256}"
  --n_support_questions "$N_SUPPORT_QUESTIONS"
  --inner_distill_query_weight "${INNER_DISTILL_QUERY_WEIGHT:-0.0}"
  --inner_distill_target_weight "${INNER_DISTILL_TARGET_WEIGHT:-1.0}"
  --inner_reconstruction_weight "$INNER_RECONSTRUCTION_WEIGHT"
  --memory_backend prefix --n_mem_tokens "$N_MEM_TOKENS"
  --n_ctrl_tokens "${N_CTRL_TOKENS:-0}"
  --K "$K" --inner_lr "$INNER_LR" --use_adam false --grad_mode second
  --freeze_backbone "${FREEZE_BACKBONE:-false}"
  --use_write_lora "${USE_WRITE_LORA:-false}"
  --write_lora_r "${WRITE_LORA_R:-8}" --write_lora_alpha "${WRITE_LORA_ALPHA:-16}"
  --write_lora_dropout "${WRITE_LORA_DROPOUT:-0.0}"
  --use_mem_proj "${USE_MEM_PROJ:-false}" --mem_proj_mode "${MEM_PROJ_MODE:-none}"
  --attn_implementation "${ATTN_IMPL:-eager}"
  --use_gradient_checkpointing "${USE_GRADIENT_CHECKPOINTING:-false}"
  --read_ce_weight "${READ_CE_WEIGHT:-1.0}"
  --distill_logits_query_weight "${DISTILL_LOGITS_QUERY_WEIGHT:-0.0}"
  --distill_logits_target_weight "${DISTILL_LOGITS_TARGET_WEIGHT:-0.0}"
  --distill_hidden_query_weight "${DISTILL_HIDDEN_QUERY_WEIGHT:-0.0}"
  --distill_hidden_target_weight "${DISTILL_HIDDEN_TARGET_WEIGHT:-0.0}"
  --learning_rate "${LR:-1e-4}" --weight_decay "${WEIGHT_DECAY:-0.0}"
  --adam_beta1 "${ADAM_BETA1:-0.9}" --adam_beta2 "${ADAM_BETA2:-0.999}"
  --lr_scheduler_type "${LR_SCHEDULER_TYPE:-constant_with_warmup}"
  --max_steps "${MAX_STEPS:-50000}" --warmup_steps "${WARMUP_STEPS:-1000}"
  --eval_steps "${EVAL_STEPS:-100}" --logging_steps "${LOGGING_STEPS:-100}"
  --early_stopping_patience "${EARLY_STOPPING_PATIENCE:-50}"
  --metric_for_best_model "${METRIC_FOR_BEST_MODEL:-token_accuracy}"
  --dataloader_num_workers "${DATALOADER_NUM_WORKERS:-0}" --seed "$SEED"
)
if [ -n "$PRETRAINED_MODEL" ]; then
  CMD+=(--pretrained_model "$PRETRAINED_MODEL")
else
  CMD+=(--base_model "$BASE_MODEL")
fi
if [ -n "${MAX_CONTEXT_LENGTH:-}" ]; then CMD+=(--max_context_length "$MAX_CONTEXT_LENGTH"); fi
if [ -n "${INNER_CLIP_VALUE:-}" ]; then CMD+=(--inner_clip_value "$INNER_CLIP_VALUE"); fi
if [ -n "${INNER_CLIP_NORM:-}" ]; then CMD+=(--inner_clip_norm "$INNER_CLIP_NORM"); fi
if [ -n "${INIT_CHECKPOINT:-}" ]; then CMD+=(--init_checkpoint "$INIT_CHECKPOINT"); fi
if [ -n "${INIT_BASE_CHECKPOINT:-}" ]; then CMD+=(--init_base_checkpoint "$INIT_BASE_CHECKPOINT"); fi
if [ -n "${WRITE_LORA_TARGET_MODULES:-}" ]; then CMD+=(--write_lora_target_modules "$WRITE_LORA_TARGET_MODULES"); fi
# No other experiment script or launcher is sourced or executed.
exec accelerate launch --num_processes 1 --num_machines 1 --mixed_precision "$MIXED_PRECISION" "${CMD[@]}" "$@"
