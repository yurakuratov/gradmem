#!/bin/bash

set -euo pipefail

# Environment is expected to be already activated.
PYTHON_BIN=${PYTHON_BIN:-python}
export WANDB_PROJECT=${WANDB_PROJECT:-gradmem}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}

# Distributed / batching
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}
NP=${NP:-$(($(tr -cd ',' <<< "$CUDA_VISIBLE_DEVICES" | wc -c) + 1))}
LR=${LR:-1e-04}
TBS=${TBS:-64}
PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE:-64}
GRAD_ACC_STEPS=${GRAD_ACC_STEPS:-$((TBS/(PER_DEVICE_BATCH_SIZE*NP)))}
MIXED_PRECISION=${MIXED_PRECISION:-no}

# Base model
BASE_MODEL=${BASE_MODEL:-llama}
L=${L:-4}
H=${H:-4}
D=${D:-128}

# HuggingFace KV-retrieval dataset: irodkin/kv_retrieval, subset N${N_PAIRS}-K${K_SIZE}V${V_SIZE}-V${VOCAB_SIZE}
HF_DATASET=${HF_DATASET:-irodkin/kv_retrieval}
N_PAIRS=${N_PAIRS:-8}
K_SIZE=${K_SIZE:-2}
V_SIZE=${V_SIZE:-2}
VOCAB_SIZE=${VOCAB_SIZE:-62}
HF_SUBSET=${HF_SUBSET:-N${N_PAIRS}-K${K_SIZE}V${V_SIZE}-V${VOCAB_SIZE}}
TOKENIZER_PATH=${TOKENIZER_PATH:-./tokenizers/kv_alphabet_${VOCAB_SIZE}/}
MAX_CONTEXT_LENGTH=${MAX_CONTEXT_LENGTH:-None}

# EnergyGradMem memory / inner loop
MEMORY_BACKEND=${MEMORY_BACKEND:-prefix}
N_MEM_TOKENS=${N_MEM_TOKENS:-8}
N_CTRL_TOKENS=${N_CTRL_TOKENS:-0}
K=${K:-2}
LAST_K_SECOND_ORDER=${LAST_K_SECOND_ORDER:-$K}
INNER_LR=${INNER_LR:-0.04}
INNER_CLIP_VALUE=${INNER_CLIP_VALUE:-None}
INNER_CLIP_NORM=${INNER_CLIP_NORM:-None}
USE_ADAM=${USE_ADAM:-false}
GRAD_MODE=${GRAD_MODE:-second}
USE_MEM_PROJ=${USE_MEM_PROJ:-false}
MEM_PROJ_MODE=${MEM_PROJ_MODE:-none}
USE_WRITE_HEAD=${USE_WRITE_HEAD:-true}
USE_WRITE_LORA=${USE_WRITE_LORA:-false}
WRITE_LORA_R=${WRITE_LORA_R:-8}
WRITE_LORA_ALPHA=${WRITE_LORA_ALPHA:-16}
WRITE_LORA_DROPOUT=${WRITE_LORA_DROPOUT:-0.0}
WRITE_LORA_TARGET_MODULES=${WRITE_LORA_TARGET_MODULES:-None}
FREEZE_BACKBONE=${FREEZE_BACKBONE:-false}
USE_GRADIENT_CHECKPOINTING=${USE_GRADIENT_CHECKPOINTING:-false}
ATTN_IMPLEMENTATION=${ATTN_IMPLEMENTATION:-eager}
ADD_INNER_LOSS_TO_OUTER=${ADD_INNER_LOSS_TO_OUTER:-true}
INNER_LOSS_WEIGHT=${INNER_LOSS_WEIGHT:-0.5}

# LoRA / KV-cache memory backend options
LORA_MEM_PLACEMENT=${LORA_MEM_PLACEMENT:-between_layers}
LORA_MEM_R=${LORA_MEM_R:-8}
LORA_MEM_ALPHA=${LORA_MEM_ALPHA:-16}
LORA_MEM_DROPOUT=${LORA_MEM_DROPOUT:-0.0}
LORA_MEM_LAYERS=${LORA_MEM_LAYERS:-all}
LORA_MEM_TARGET_MODULES=${LORA_MEM_TARGET_MODULES:-None}
KV_MEM_LAYERS=${KV_MEM_LAYERS:-all}

# Energy objective. Parameter names intentionally use energy_* rather than lstm_*.
INNER_OBJECTIVE=${INNER_OBJECTIVE:-lstm}
ENERGY_HIDDEN_SIZE=${ENERGY_HIDDEN_SIZE:-$D}
ENERGY_NUM_LAYERS=${ENERGY_NUM_LAYERS:-2}
ENERGY_DROPOUT=${ENERGY_DROPOUT:-0.0}

# Training control
MAX_STEPS=${MAX_STEPS:-50000}
EVAL_STEPS=${EVAL_STEPS:-100}
LOGGING_STEPS=${LOGGING_STEPS:-100}
WARMUP_STEPS=${WARMUP_STEPS:-1000}
EARLY_STOPPING_PATIENCE=${EARLY_STOPPING_PATIENCE:-500}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.0}
LR_SCHEDULER_TYPE=${LR_SCHEDULER_TYPE:-constant_with_warmup}
METRIC_FOR_BEST_MODEL=${METRIC_FOR_BEST_MODEL:-token_accuracy}
INIT_CHECKPOINT=${INIT_CHECKPOINT:-}

RUN_NAME=${RUN_NAME:-energy_gradmem_${BASE_MODEL}_L${L}H${H}D${D}_${HF_SUBSET}_mem${N_MEM_TOKENS}_K${K}_ilr${INNER_LR}_grad_${GRAD_MODE}_bs_${TBS}_lr_${LR}}
RUN_NAME_SUFFIX=${RUN_NAME_SUFFIX:-}
if [ -n "$RUN_NAME_SUFFIX" ]; then
  RUN_NAME=${RUN_NAME}_${RUN_NAME_SUFFIX}
fi
export WANDB_NAME=${WANDB_NAME:-$RUN_NAME}

N_VALUES=${N_VALUES:-1}
for N in $N_VALUES; do
  EXP_PATH=${EXP_PATH:-./runs/energy_gradmem_kv/${HF_SUBSET}/${RUN_NAME}/run_${N}}
  PORT=$((29500 + TBS + N + 17))

  $PYTHON_BIN -m accelerate.commands.launch \
    --main_process_port $PORT \
    --num_processes $NP \
    --mixed_precision "$MIXED_PRECISION" \
    --config_file accelerate.yaml \
    run_energy_gradmem_on_kv_retrieval.py \
    --exp_path "$EXP_PATH" \
    --per_device_batch_size "$PER_DEVICE_BATCH_SIZE" \
    --gradient_accumulation_steps "$GRAD_ACC_STEPS" \
    --total_batch_size "$TBS" \
    --hf_dataset "$HF_DATASET" \
    --hf_subset "$HF_SUBSET" \
    --tokenizer_path "$TOKENIZER_PATH" \
    --learning_rate "$LR" \
    --weight_decay "$WEIGHT_DECAY" \
    --lr_scheduler_type "$LR_SCHEDULER_TYPE" \
    --metric_for_best_model "$METRIC_FOR_BEST_MODEL" \
    --n_layer "$L" \
    --n_head "$H" \
    --n_embd "$D" \
    --base_model "$BASE_MODEL" \
    $( [ -n "$INIT_CHECKPOINT" ] && echo "--init_checkpoint $INIT_CHECKPOINT" ) \
    $( [ "$MAX_CONTEXT_LENGTH" != "None" ] && echo "--max_context_length $MAX_CONTEXT_LENGTH" ) \
    --memory_backend "$MEMORY_BACKEND" \
    --n_mem_tokens "$N_MEM_TOKENS" \
    --K "$K" \
    --last_K_second_order "$LAST_K_SECOND_ORDER" \
    --inner_lr "$INNER_LR" \
    --use_adam "$USE_ADAM" \
    --grad_mode "$GRAD_MODE" \
    --n_ctrl_tokens "$N_CTRL_TOKENS" \
    $( [ "$INNER_CLIP_VALUE" != "None" ] && echo "--inner_clip_value $INNER_CLIP_VALUE" ) \
    $( [ "$INNER_CLIP_NORM" != "None" ] && echo "--inner_clip_norm $INNER_CLIP_NORM" ) \
    $( [ "$USE_MEM_PROJ" = true ] && echo "--use_mem_proj" ) \
    --mem_proj_mode "$MEM_PROJ_MODE" \
    --use_write_head "$USE_WRITE_HEAD" \
    --use_write_lora "$USE_WRITE_LORA" \
    --write_lora_r "$WRITE_LORA_R" \
    --write_lora_alpha "$WRITE_LORA_ALPHA" \
    --write_lora_dropout "$WRITE_LORA_DROPOUT" \
    $( [ "$WRITE_LORA_TARGET_MODULES" != "None" ] && echo "--write_lora_target_modules $WRITE_LORA_TARGET_MODULES" ) \
    --lora_mem_placement "$LORA_MEM_PLACEMENT" \
    --lora_mem_r "$LORA_MEM_R" \
    --lora_mem_alpha "$LORA_MEM_ALPHA" \
    --lora_mem_dropout "$LORA_MEM_DROPOUT" \
    --lora_mem_layers "$LORA_MEM_LAYERS" \
    $( [ "$LORA_MEM_TARGET_MODULES" != "None" ] && echo "--lora_mem_target_modules $LORA_MEM_TARGET_MODULES" ) \
    --kv_mem_layers "$KV_MEM_LAYERS" \
    --freeze_backbone "$FREEZE_BACKBONE" \
    --use_gradient_checkpointing "$USE_GRADIENT_CHECKPOINTING" \
    --attn_implementation "$ATTN_IMPLEMENTATION" \
    --add_inner_loss_to_outer "$ADD_INNER_LOSS_TO_OUTER" \
    $( [ "$INNER_LOSS_WEIGHT" != "None" ] && echo "--inner_loss_weight $INNER_LOSS_WEIGHT" ) \
    --inner_objective "$INNER_OBJECTIVE" \
    --energy_hidden_size "$ENERGY_HIDDEN_SIZE" \
    --energy_num_layers "$ENERGY_NUM_LAYERS" \
    --energy_dropout "$ENERGY_DROPOUT" \
    --max_steps "$MAX_STEPS" \
    --eval_steps "$EVAL_STEPS" \
    --logging_steps "$LOGGING_STEPS" \
    --warmup_steps "$WARMUP_STEPS" \
    --early_stopping_patience "$EARLY_STOPPING_PATIENCE" \
    --seed $((142 + N))
done
