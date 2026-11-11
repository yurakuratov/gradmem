#!/bin/bash
# Based on run_energygradmem_on_squad.sh. Keep its full launcher surface and
# lifecycle; support distillation is an additive experiment variant.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/collect_env_state.sh"

# NP=${NP:-1}
NP=2
LR=${LR:-1e-04}
ADAM_BETA1=${ADAM_BETA1:-0.9}
ADAM_BETA2=${ADAM_BETA2:-0.999}
TBS=64
PER_DEVICE_BATCH_SIZE=16
GRAD_ACC_STEPS=$(($TBS/($PER_DEVICE_BATCH_SIZE*$NP)))

# Model settings match the MQAR energy launcher.
L=${L:-4}
H=${H:-4}
D=${D:-256}
BASE_MODEL=${BASE_MODEL:-llama}
PRETRAINED_MODEL=unsloth/Llama-3.2-1B
SEPARATE_WRITER=${SEPARATE_WRITER:-false}
WRITER_BASE_MODEL=${WRITER_BASE_MODEL:-$BASE_MODEL}
WRITER_PRETRAINED_MODEL=${WRITER_PRETRAINED_MODEL:-}
WRITER_L=${WRITER_L:-$L}
WRITER_H=${WRITER_H:-$H}
WRITER_D=${WRITER_D:-$D}

# SQuAD uses a text tokenizer rather than the synthetic MQAR vocabulary.
DATA_NAME=squad
DATA_PATH="./data/${DATA_NAME}"
TOKENIZER_PATH=${TOKENIZER_PATH:-gpt2}
# Maximum context tokens (including special tokens); empty leaves contexts uncapped.
MAX_CONTEXT_LENGTH=${MAX_CONTEXT_LENGTH:-256}

MEMORY_BACKEND=prefix
# Support KL augments energy WRITE; reconstruction-only mode remains available.
WRITE_OBJECTIVE=${WRITE_OBJECTIVE:-energy}
N_MEM_TOKENS=8
N_CTRL_TOKENS=0
K=1
LAST_K_SECOND_ORDER=${K}
INNER_LR=0.4
K_SAMPLING=false
K_MIN=1
K_MAX=4
ILR_MIN=0.1
ILR_MAX=2.5
ILR_SAMPLE_UNIQUE=true
INNER_CLIP_VALUE=None
INNER_CLIP_NORM=10.0
USE_ADAM=false
GRAD_MODE=second
USE_MEM_PROJ=false
MEM_PROJ_MODE=none
USE_WRITE_HEAD=false
USE_WRITE_LORA=true
WRITE_LORA_R=16
WRITE_LORA_ALPHA=32
WRITE_LORA_DROPOUT=${WRITE_LORA_DROPOUT:-0.0}
WRITE_LORA_TARGETS=${WRITE_LORA_TARGETS:-}
FREEZE_BACKBONE=true
LORA_MEM_PLACEMENT=${LORA_MEM_PLACEMENT:-between_layers}
LORA_MEM_TARGET_MODULES=${LORA_MEM_TARGET_MODULES:-}
LORA_MEM_R=${LORA_MEM_R:-8}
LORA_MEM_ALPHA=${LORA_MEM_ALPHA:-16}
LORA_MEM_DROPOUT=${LORA_MEM_DROPOUT:-0.0}
LORA_MEM_LAYERS=${LORA_MEM_LAYERS:-all}
KV_MEM_LAYERS=${KV_MEM_LAYERS:-all}

ENERGY_HEAD_HIDDEN_DIM=None
WRITE_RECONSTRUCTION_WEIGHT=1.0
WRITE_ENERGY_WEIGHT=1.0
ENERGY_RANK_WEIGHT=0.0
ENERGY_TRAJ_WEIGHT=${ENERGY_TRAJ_WEIGHT:-0.0}
ENERGY_MARGIN=${ENERGY_MARGIN:-0.1}
ENERGY_TRAJ_MARGIN=${ENERGY_TRAJ_MARGIN:-0.0}
ENERGY_RANK_TEMPERATURE=${ENERGY_RANK_TEMPERATURE:-1.0}
ENERGY_MIX_ALPHA=${ENERGY_MIX_ALPHA:-0.75}
ENERGY_ANCHOR_WEIGHT=0.0
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
ALIGNMENT_SAMPLING=false
ALIGN_LAST_STEP=${ALIGN_LAST_STEP:-false}
GRAD_ALIGN_NORM=${GRAD_ALIGN_NORM:-none}
INTERMEDIATE_READ_WEIGHT=0.0
MEMORY_NOISE_SIGMA=${MEMORY_NOISE_SIGMA:-0.0}
ORTHOGONAL_LOSS_WEIGHT=${ORTHOGONAL_LOSS_WEIGHT:-0.0}
IVAN_LOSS_WEIGHT=${IVAN_LOSS_WEIGHT:-0.0}
ADD_INNER_LOSS_TO_OUTER=false
INNER_LOSS_WEIGHT=0.5
ATTN_IMPL=flashbob
MIXED_PRECISION=bf16
MAX_STEPS=100000
INIT_CHECKPOINT=${INIT_CHECKPOINT:-}
INIT_BASE_CHECKPOINT=${INIT_BASE_CHECKPOINT:-}

# Explicit training and support-distillation controls (not hidden CLI defaults).
WEIGHT_DECAY=${WEIGHT_DECAY:-0.0}
LR_SCHEDULER_TYPE=${LR_SCHEDULER_TYPE:-constant_with_warmup}
METRIC_FOR_BEST_MODEL=${METRIC_FOR_BEST_MODEL:-token_accuracy}
EVAL_STEPS=${EVAL_STEPS:-500}
LOGGING_STEPS=${LOGGING_STEPS:-500}
WARMUP_STEPS=${WARMUP_STEPS:-10000}
EARLY_STOPPING_PATIENCE=${EARLY_STOPPING_PATIENCE:-500}
SEED=${SEED:-142}
USE_GRADIENT_CHECKPOINTING=${USE_GRADIENT_CHECKPOINTING:-false}
DATALOADER_NUM_WORKERS=${DATALOADER_NUM_WORKERS:-0}
RESUME_FROM_CHECKPOINT=${RESUME_FROM_CHECKPOINT:-}
READ_CE_WEIGHT=${READ_CE_WEIGHT:-1.0}
DISTILL_LOGITS_QUERY_WEIGHT=${DISTILL_LOGITS_QUERY_WEIGHT:-0.0}
DISTILL_LOGITS_TARGET_WEIGHT=${DISTILL_LOGITS_TARGET_WEIGHT:-0.0}
DISTILL_HIDDEN_QUERY_WEIGHT=${DISTILL_HIDDEN_QUERY_WEIGHT:-0.0}
DISTILL_HIDDEN_TARGET_WEIGHT=${DISTILL_HIDDEN_TARGET_WEIGHT:-0.0}
N_SUPPORT_QUESTIONS=${N_SUPPORT_QUESTIONS:-2}
INNER_DISTILL_QUERY_WEIGHT=${INNER_DISTILL_QUERY_WEIGHT:-0.0}
INNER_DISTILL_TARGET_WEIGHT=${INNER_DISTILL_TARGET_WEIGHT:-1.0}
INNER_RECONSTRUCTION_WEIGHT=${INNER_RECONSTRUCTION_WEIGHT:-0.0}

N_VALUES=(2)

if [ "$WRITE_OBJECTIVE" = reconstruction ]; then
  RUN_NAME=gradmem_${BASE_MODEL}_L${L}H${H}D${D}_mem${N_MEM_TOKENS}
else
  RUN_NAME=energygradmem_${BASE_MODEL}_L${L}H${H}D${D}_mem${N_MEM_TOKENS}
fi
if [ "$SEPARATE_WRITER" = true ]; then
  RUN_NAME=${RUN_NAME}_writer_${WRITER_BASE_MODEL}_L${WRITER_L}H${WRITER_H}D${WRITER_D}
fi
RUN_NAME=${RUN_NAME}_K${K}_ilr${INNER_LR}
if [ "$K_SAMPLING" = true ]; then RUN_NAME=${RUN_NAME}_Ksample${K_MIN}-${K_MAX}_ilr${ILR_MIN}-${ILR_MAX}_unique${ILR_SAMPLE_UNIQUE}; fi
if [ "$LAST_K_SECOND_ORDER" != "$K" ]; then RUN_NAME=${RUN_NAME}_last_K${LAST_K_SECOND_ORDER}; fi
if [ "$INNER_CLIP_VALUE" != None ]; then RUN_NAME=${RUN_NAME}_icv${INNER_CLIP_VALUE}; fi
if [ "$INNER_CLIP_NORM" != None ]; then RUN_NAME=${RUN_NAME}_icn${INNER_CLIP_NORM}; fi
if [ "$USE_MEM_PROJ" = true ]; then RUN_NAME=${RUN_NAME}_mem_proj_${MEM_PROJ_MODE}; fi
if [ "$USE_WRITE_HEAD" = true ]; then RUN_NAME=${RUN_NAME}_whead; fi
if [ "$USE_WRITE_LORA" = true ]; then
  RUN_NAME=${RUN_NAME}_write_lora_r${WRITE_LORA_R}_a${WRITE_LORA_ALPHA}
  if [ "$WRITE_LORA_DROPOUT" != 0.0 ]; then RUN_NAME=${RUN_NAME}_drop${WRITE_LORA_DROPOUT}; fi
fi
if [ "$FREEZE_BACKBONE" = true ]; then RUN_NAME=${RUN_NAME}_freeze_backbone; fi
if [ "$WRITE_OBJECTIVE" != reconstruction ]; then RUN_NAME=${RUN_NAME}_energy; fi
if [ "$WRITE_OBJECTIVE" = energy_with_reconstruction ]; then RUN_NAME=${RUN_NAME}_recon${WRITE_RECONSTRUCTION_WEIGHT}_energy${WRITE_ENERGY_WEIGHT}; fi
if [ "$ENERGY_HEAD_HIDDEN_DIM" != None ]; then RUN_NAME=${RUN_NAME}_eh${ENERGY_HEAD_HIDDEN_DIM}; fi
if [ "$USE_LAYERWISE_ENERGY" = true ]; then RUN_NAME=${RUN_NAME}_layerwise; fi
if [ "$ENERGY_RANK_WEIGHT" != 0.0 ]; then RUN_NAME=${RUN_NAME}_rank${ENERGY_RANK_WEIGHT}_m${ENERGY_MARGIN}_t${ENERGY_RANK_TEMPERATURE}_mix${ENERGY_MIX_ALPHA}; fi
if [ "$ENERGY_TRAJ_WEIGHT" != 0.0 ]; then RUN_NAME=${RUN_NAME}_traj${ENERGY_TRAJ_WEIGHT}_m${ENERGY_TRAJ_MARGIN}; fi
if [ "$ENERGY_ANCHOR_WEIGHT" != 0.0 ]; then RUN_NAME=${RUN_NAME}_anchor${ENERGY_ANCHOR_WEIGHT}; fi
if [ "$LIPSCHITZ_WEIGHT" != 0.0 ]; then RUN_NAME=${RUN_NAME}_lip${LIPSCHITZ_WEIGHT}_L${LIPSCHITZ_CONSTRAINT}; fi
if [ "$ENERGY_MEMORY_SEARCH_WEIGHT" != 0.0 ]; then
  RUN_NAME=${RUN_NAME}_msearch${ENERGY_MEMORY_SEARCH_WEIGHT}_n${ENERGY_MEMORY_SEARCH_NUM_SAMPLES}_r${ENERGY_MEMORY_SEARCH_RADIUS_SCALE}
  if [ "$ENERGY_MEMORY_SEARCH_USE_GAIN_WEIGHTING" = true ]; then RUN_NAME=${RUN_NAME}_gain${ENERGY_MEMORY_SEARCH_GAIN_EMA_DECAY}; fi
  if [ "$ENERGY_MEMORY_SEARCH_MIN_RELATIVE_TARGET_GAIN" != 0.0 ]; then RUN_NAME=${RUN_NAME}_mingain${ENERGY_MEMORY_SEARCH_MIN_RELATIVE_TARGET_GAIN}; fi
  if [ "$ENERGY_MEMORY_SEARCH_USE_BEST_FOR_NEXT_STEP" = true ]; then RUN_NAME=${RUN_NAME}_bestnext; fi
fi
RUN_NAME=${RUN_NAME}_grad_${GRAD_MODE}
if [ "$ADD_INNER_LOSS_TO_OUTER" = true ]; then RUN_NAME=${RUN_NAME}_inner${INNER_LOSS_WEIGHT}; fi
if [ "$USE_ADAM" = true ]; then RUN_NAME=${RUN_NAME}_with_adam; fi
if [ "$MEMORY_ALIGNMENT_WEIGHT" != 0.0 ]; then RUN_NAME=${RUN_NAME}_memalign${MEMORY_ALIGNMENT_WEIGHT}; fi
if [ "$STEP_ALIGNMENT_WEIGHT" != 0.0 ]; then RUN_NAME=${RUN_NAME}_stepalign${STEP_ALIGNMENT_WEIGHT}_${ALIGNMENT_SAMPLING}_${ALIGN_LAST_STEP}_${GRAD_ALIGN_NORM}; fi
if [ "$INTERMEDIATE_READ_WEIGHT" != 0.0 ]; then RUN_NAME=${RUN_NAME}_iread${INTERMEDIATE_READ_WEIGHT}; fi
if [ "$MEMORY_NOISE_SIGMA" != 0.0 ]; then RUN_NAME=${RUN_NAME}_mnoise${MEMORY_NOISE_SIGMA}; fi
if [ "$ORTHOGONAL_LOSS_WEIGHT" != 0.0 ]; then RUN_NAME=${RUN_NAME}_orth${ORTHOGONAL_LOSS_WEIGHT}; fi
if [ "$IVAN_LOSS_WEIGHT" != 0.0 ]; then RUN_NAME=${RUN_NAME}_ivan${IVAN_LOSS_WEIGHT}; fi
if [ "$READ_FOCAL_GAMMA" != 0.0 ]; then RUN_NAME=${RUN_NAME}_focal${READ_FOCAL_GAMMA}; fi
RUN_NAME=${RUN_NAME}_bs_${TBS}_lr_${LR}
# Keep inherited naming, then encode every support/outer-distillation weight.
RUN_NAME=${RUN_NAME}_support${N_SUPPORT_QUESTIONS}_iklq${INNER_DISTILL_QUERY_WEIGHT}_iklt${INNER_DISTILL_TARGET_WEIGHT}_ice${INNER_RECONSTRUCTION_WEIGHT}
RUN_NAME=${RUN_NAME}_ce${READ_CE_WEIGHT}_klq${DISTILL_LOGITS_QUERY_WEIGHT}_klt${DISTILL_LOGITS_TARGET_WEIGHT}_hq${DISTILL_HIDDEN_QUERY_WEIGHT}_ht${DISTILL_HIDDEN_TARGET_WEIGHT}
if [ "$WEIGHT_DECAY" != 0.0 ]; then RUN_NAME=${RUN_NAME}_wd${WEIGHT_DECAY}; fi
if [ "$LR_SCHEDULER_TYPE" != constant_with_warmup ]; then RUN_NAME=${RUN_NAME}_${LR_SCHEDULER_TYPE}; fi
if [ "$USE_GRADIENT_CHECKPOINTING" = true ]; then RUN_NAME=${RUN_NAME}_gckpt; fi
if [ "$SEED" != 142 ]; then RUN_NAME=${RUN_NAME}_seed${SEED}; fi
if [ "$WARMUP_STEPS" != 10000 ]; then RUN_NAME=${RUN_NAME}_warm${WARMUP_STEPS}; fi
if [ "$EVAL_STEPS" != 500 ]; then RUN_NAME=${RUN_NAME}_eval${EVAL_STEPS}; fi
if [ "$LOGGING_STEPS" != 500 ]; then RUN_NAME=${RUN_NAME}_log${LOGGING_STEPS}; fi
if [ "$EARLY_STOPPING_PATIENCE" != 500 ]; then RUN_NAME=${RUN_NAME}_pat${EARLY_STOPPING_PATIENCE}; fi
if [ "$METRIC_FOR_BEST_MODEL" != token_accuracy ]; then RUN_NAME=${RUN_NAME}_best_${METRIC_FOR_BEST_MODEL}; fi
if [ "$DATALOADER_NUM_WORKERS" != 0 ]; then RUN_NAME=${RUN_NAME}_workers${DATALOADER_NUM_WORKERS}; fi
if [ -n "$RESUME_FROM_CHECKPOINT" ]; then RUN_NAME=${RUN_NAME}_resume_$(basename "$RESUME_FROM_CHECKPOINT"); fi
if [ "$MIXED_PRECISION" = no ]; then RUN_NAME=${RUN_NAME}_fp32; fi
if [ -n "${RUN_NAME_SUFFIX:-}" ]; then RUN_NAME=${RUN_NAME}_${RUN_NAME_SUFFIX}; fi

for N in "${N_VALUES[@]}"; do
  EXP_PATH="./runs/${DATA_NAME}/${RUN_NAME}/run_${N}"
  if ! prepare_locked_run "$EXP_PATH" "$0" "$NP" false; then continue; fi
  PORT="$(find_free_port)"
  CMD=(
    accelerate launch --main_process_port "$PORT" --num_processes "$NP"
    --mixed_precision "$MIXED_PRECISION" --config_file accelerate.yaml
    run_support_distillation_on_squad.py
    --exp_path "$EXP_PATH" --per_device_batch_size "$PER_DEVICE_BATCH_SIZE"
    --gradient_accumulation_steps "$GRAD_ACC_STEPS" --total_batch_size "$TBS"
    --dataset_name "$DATA_NAME" --data_path "$DATA_PATH" --tokenizer_path "$TOKENIZER_PATH"
    --learning_rate "$LR" --adam_beta1 "$ADAM_BETA1" --adam_beta2 "$ADAM_BETA2"
    --weight_decay "$WEIGHT_DECAY" --lr_scheduler_type "$LR_SCHEDULER_TYPE"
    --metric_for_best_model "$METRIC_FOR_BEST_MODEL"
    --use_gradient_checkpointing "$USE_GRADIENT_CHECKPOINTING"
    --dataloader_num_workers "$DATALOADER_NUM_WORKERS"
    --n_support_questions "$N_SUPPORT_QUESTIONS"
    --inner_distill_query_weight "$INNER_DISTILL_QUERY_WEIGHT"
    --inner_distill_target_weight "$INNER_DISTILL_TARGET_WEIGHT"
    --inner_reconstruction_weight "$INNER_RECONSTRUCTION_WEIGHT"
    --read_ce_weight "$READ_CE_WEIGHT"
    --distill_logits_query_weight "$DISTILL_LOGITS_QUERY_WEIGHT"
    --distill_logits_target_weight "$DISTILL_LOGITS_TARGET_WEIGHT"
    --distill_hidden_query_weight "$DISTILL_HIDDEN_QUERY_WEIGHT"
    --distill_hidden_target_weight "$DISTILL_HIDDEN_TARGET_WEIGHT"
    --n_layer "$L" --n_head "$H" --n_embd "$D"
    --separate_writer "$SEPARATE_WRITER" --memory_backend "$MEMORY_BACKEND"
    --n_mem_tokens "$N_MEM_TOKENS" --n_ctrl_tokens "$N_CTRL_TOKENS"
    --write_objective "$WRITE_OBJECTIVE" --K "$K" --K_sampling "$K_SAMPLING"
    --K_min "$K_MIN" --K_max "$K_MAX" --ilr_min "$ILR_MIN" --ilr_max "$ILR_MAX"
    --ilr_sample_unique "$ILR_SAMPLE_UNIQUE" --last_K_second_order "$LAST_K_SECOND_ORDER"
    --inner_lr "$INNER_LR" --use_adam "$USE_ADAM" --grad_mode "$GRAD_MODE"
    --freeze_backbone "$FREEZE_BACKBONE"
    --energy_rank_weight "$ENERGY_RANK_WEIGHT" --energy_traj_weight "$ENERGY_TRAJ_WEIGHT"
    --write_reconstruction_weight "$WRITE_RECONSTRUCTION_WEIGHT" --write_energy_weight "$WRITE_ENERGY_WEIGHT"
    --energy_margin "$ENERGY_MARGIN" --energy_traj_margin "$ENERGY_TRAJ_MARGIN"
    --energy_rank_temperature "$ENERGY_RANK_TEMPERATURE" --energy_mix_alpha "$ENERGY_MIX_ALPHA"
    --energy_anchor_weight "$ENERGY_ANCHOR_WEIGHT" --use_layerwise_energy "$USE_LAYERWISE_ENERGY"
    --lipschitz_weight "$LIPSCHITZ_WEIGHT" --lipschitz_constraint "$LIPSCHITZ_CONSTRAINT"
    --energy_memory_search_weight "$ENERGY_MEMORY_SEARCH_WEIGHT"
    --energy_memory_search_num_samples "$ENERGY_MEMORY_SEARCH_NUM_SAMPLES"
    --energy_memory_search_radius_scale "$ENERGY_MEMORY_SEARCH_RADIUS_SCALE"
    --energy_memory_search_use_gain_weighting "$ENERGY_MEMORY_SEARCH_USE_GAIN_WEIGHTING"
    --energy_memory_search_gain_ema_decay "$ENERGY_MEMORY_SEARCH_GAIN_EMA_DECAY"
    --energy_memory_search_min_relative_target_gain "$ENERGY_MEMORY_SEARCH_MIN_RELATIVE_TARGET_GAIN"
    --energy_memory_search_use_best_for_next_step "$ENERGY_MEMORY_SEARCH_USE_BEST_FOR_NEXT_STEP"
    --read_focal_gamma "$READ_FOCAL_GAMMA" --memory_alignment_weight "$MEMORY_ALIGNMENT_WEIGHT"
    --step_alignment_weight "$STEP_ALIGNMENT_WEIGHT" --alignment_sampling "$ALIGNMENT_SAMPLING"
    --align_last_step "$ALIGN_LAST_STEP" --grad_align_norm "$GRAD_ALIGN_NORM"
    --intermediate_read_weight "$INTERMEDIATE_READ_WEIGHT" --memory_noise_sigma "$MEMORY_NOISE_SIGMA"
    --orthogonal_loss_weight "$ORTHOGONAL_LOSS_WEIGHT" --ivan_loss_weight "$IVAN_LOSS_WEIGHT"
    --attn_implementation "$ATTN_IMPL" --max_steps "$MAX_STEPS" --eval_steps "$EVAL_STEPS"
    --logging_steps "$LOGGING_STEPS" --warmup_steps "$WARMUP_STEPS" --early_stopping_patience "$EARLY_STOPPING_PATIENCE" --seed "$((SEED+N))"
  )
  if [ -n "$PRETRAINED_MODEL" ]; then CMD+=( --pretrained_model "$PRETRAINED_MODEL" ); else CMD+=( --base_model "$BASE_MODEL" ); fi
  if [ "$SEPARATE_WRITER" = true ]; then
    CMD+=( --writer_base_model "$WRITER_BASE_MODEL" --writer_n_layer "$WRITER_L" --writer_n_head "$WRITER_H" --writer_n_embd "$WRITER_D" )
    if [ -n "$WRITER_PRETRAINED_MODEL" ]; then CMD+=( --writer_pretrained_model "$WRITER_PRETRAINED_MODEL" ); fi
  fi
  if [ -n "$INIT_CHECKPOINT" ]; then CMD+=( --init_checkpoint "$INIT_CHECKPOINT" ); fi
  if [ -n "$MAX_CONTEXT_LENGTH" ]; then CMD+=( --max_context_length "$MAX_CONTEXT_LENGTH" ); fi
  if [ -n "$INIT_BASE_CHECKPOINT" ]; then CMD+=( --init_base_checkpoint "$INIT_BASE_CHECKPOINT" ); fi
  if [ -n "$RESUME_FROM_CHECKPOINT" ]; then CMD+=( --resume_from_checkpoint "$RESUME_FROM_CHECKPOINT" ); fi
  if [ "$INNER_CLIP_VALUE" != None ]; then CMD+=( --inner_clip_value "$INNER_CLIP_VALUE" ); fi
  if [ "$INNER_CLIP_NORM" != None ]; then CMD+=( --inner_clip_norm "$INNER_CLIP_NORM" ); fi
  if [ "$USE_MEM_PROJ" = true ]; then CMD+=( --use_mem_proj --mem_proj_mode "$MEM_PROJ_MODE" ); fi
  if [ "$USE_WRITE_HEAD" = true ]; then CMD+=( --use_write_head ); fi
  if [ "$USE_WRITE_LORA" = true ]; then
    CMD+=( --use_write_lora --write_lora_r "$WRITE_LORA_R" --write_lora_alpha "$WRITE_LORA_ALPHA" --write_lora_dropout "$WRITE_LORA_DROPOUT" )
    if [ -n "$WRITE_LORA_TARGETS" ]; then CMD+=( --write_lora_target_modules "$WRITE_LORA_TARGETS" ); fi
  fi
  if [ "$MEMORY_BACKEND" = lora ]; then CMD+=( --lora_mem_placement "$LORA_MEM_PLACEMENT" --lora_mem_r "$LORA_MEM_R" --lora_mem_alpha "$LORA_MEM_ALPHA" --lora_mem_dropout "$LORA_MEM_DROPOUT" --lora_mem_layers "$LORA_MEM_LAYERS" --lora_mem_target_modules "$LORA_MEM_TARGET_MODULES" ); fi
  if [ "$MEMORY_BACKEND" = kv_cache ]; then CMD+=( --kv_mem_layers "$KV_MEM_LAYERS" ); fi
  if [ "$ENERGY_HEAD_HIDDEN_DIM" != None ]; then CMD+=( --energy_head_hidden_dim "$ENERGY_HEAD_HIDDEN_DIM" ); fi
  if [ "$ADD_INNER_LOSS_TO_OUTER" = true ]; then CMD+=( --add_inner_loss_to_outer --inner_loss_weight "$INNER_LOSS_WEIGHT" ); fi
  print_run_header "$EXP_PATH" "$PORT" "$NP" "$MIXED_PRECISION" "${CMD[@]}"
  start_run_timer
  set +e
  "${CMD[@]}" 2>&1 | tee -a "$RUN_LOCK_LOG"
  RC=${PIPESTATUS[0]}
  set -e
  finalize_locked_run "$EXP_PATH" "$RUN_LOCK_DIR" "$RUN_LOCK_LOG" "$RC"
done
