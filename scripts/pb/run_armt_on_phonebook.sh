#!/bin/bash
set -e

# Finetune ARMT on Phonebook (HF dataset), mirroring hyperparams/sweep structure of:
# - scripts/pb/run_rmt_on_phonebook.sh (LR/TBS/steps/warmup/eval cadence)
# and using the ARMT runner:
# - run_armt_on_squad.py (supports "phonebook" datasets too)

export WANDB_PROJECT=phonebook
export HF_Trainer=1

export CUDA_VISIBLE_DEVICES=0

NP=$(echo $CUDA_VISIBLE_DEVICES | awk -F',' '{print NF}')

# Phonebook hyperparams (match run_rmt_on_phonebook.sh)
LR_SCHEDULER_TYPE="constant_with_warmup"
TBS=256

WARMUP_STEPS=10000

# Backbone
MODEL_NAME=gpt2
PRETRAINED_MODEL=gpt2
LAYERS_ATTR="transformer.h"

# ARMT parameters (match run_armt_on_squad.sh defaults)
NUM_MEM_TOKENS=32
D_MEM=64
SEGMENT_SIZE=128
SEGMENT_ALIGNMENT="left"
READING_DEPTH_MULTIPLIER=1
WRITING_DEPTH_MULTIPLIER=1

# Run ID
N=1

# Resume control:
# - set START_I to start from a later curriculum stage without editing arrays (0-based index)
# - the script will deterministically set --model_cpt based on (i-1), like scripts/run_armt_on_kv_retrieval.sh
START_I=${START_I:-0}

# Curriculum schedule (edit as needed). Arrays are aligned by index.
N_PAIRS_LIST=(2 4 8 16 32 64)
PER_DEVICE_BATCH_SIZES=(64 32 16 8 4 2)
MAX_STEPS_LIST=(20000 20000 20000 20000 20000 20000)
LR_LIST=(1e-03 1e-03 7e-04 5e-04 3e-04 3e-04)

for (( i=0; i<${#N_PAIRS_LIST[@]}; i++ )); do
  if [ "$i" -lt "$START_I" ]; then
    continue
  fi

  N_PAIRS=${N_PAIRS_LIST[i]}
  PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZES[i]}
  MAX_STEPS=${MAX_STEPS_LIST[i]}
  LR=${LR_LIST[i]}

  GRAD_ACC_STEPS=$(($TBS/($PER_DEVICE_BATCH_SIZE*$NP)))

  DATA_NAME="booydar/phonebook_N${N_PAIRS}"

  RUN_NAME=armt2segm_${MODEL_NAME}_mem${NUM_MEM_TOKENS}_dmem${D_MEM}_seg${SEGMENT_SIZE}_wdm${WRITING_DEPTH_MULTIPLIER}_rdm${READING_DEPTH_MULTIPLIER}_bs_${TBS}_lr_${LR}_${LR_SCHEDULER_TYPE}

  # Path to save experiment results (mirrors run_rmt_on_phonebook.sh structure)
  EXP_PATH="./runs/${DATA_NAME}/N${N_PAIRS}/${RUN_NAME}/run_$N"

  export WANDB_NAME=armt_phonebook_N${N_PAIRS}_run_${N}

  # Deterministic curriculum checkpoint: load from the *previous* stage output directory.
  # We point to the specific HF Trainer folder: checkpoint-<prev_max_steps>.
  # (If you want a different checkpoint, change MAX_STEPS_LIST for the previous stage
  #  or override manually by running the python with --model_cpt PATH.)
  if [ "$i" -eq 0 ]; then
    MODEL_CPT=None
  else
    PREV_N_PAIRS=${N_PAIRS_LIST[i-1]}
    PREV_LR=${LR_LIST[i-1]}
    PREV_MAX_STEPS=${MAX_STEPS_LIST[i-1]}
    PREV_DATA_NAME="booydar/phonebook_N${PREV_N_PAIRS}"
    PREV_RUN_NAME=armt2segm_${MODEL_NAME}_mem${NUM_MEM_TOKENS}_dmem${D_MEM}_seg${SEGMENT_SIZE}_wdm${WRITING_DEPTH_MULTIPLIER}_rdm${READING_DEPTH_MULTIPLIER}_bs_${TBS}_lr_${PREV_LR}_${LR_SCHEDULER_TYPE}
    PREV_EXP_PATH="./runs/${PREV_DATA_NAME}/N${PREV_N_PAIRS}/${PREV_RUN_NAME}/run_$N"
    MODEL_CPT="${PREV_EXP_PATH}/checkpoint-${PREV_MAX_STEPS}"
  fi

  accelerate launch \
    --main_process_port $((29500+$TBS+$N+21)) \
    --num_processes $NP \
    --mixed_precision bf16 \
    --config_file accelerate.yaml \
    run_armt_on_squad.py \
    --exp_path $EXP_PATH \
    --per_device_batch_size $PER_DEVICE_BATCH_SIZE \
    --gradient_accumulation_steps $GRAD_ACC_STEPS \
    --total_batch_size $TBS \
    --dataset_name $DATA_NAME \
    --learning_rate $LR \
    --lr_scheduler_type $LR_SCHEDULER_TYPE \
    --pretrained_model $PRETRAINED_MODEL \
    --layers_attr "$LAYERS_ATTR" \
    --num_mem_tokens $NUM_MEM_TOKENS \
    --d_mem $D_MEM \
    --segment_size $SEGMENT_SIZE \
    --segment_alignment $SEGMENT_ALIGNMENT \
    --reading_depth_multiplier $READING_DEPTH_MULTIPLIER \
    --writing_depth_multiplier $WRITING_DEPTH_MULTIPLIER \
    --max_steps $MAX_STEPS \
    --eval_steps 500 \
    --logging_steps 100 \
    --warmup_steps $WARMUP_STEPS \
    --early_stopping_patience 500 \
    --seed $((142+$N)) \
    --armt_impl old \
    --model_cpt $MODEL_CPT
done

echo "Done"

