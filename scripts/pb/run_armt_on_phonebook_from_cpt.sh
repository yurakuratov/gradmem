#!/bin/bash
set -e

# Finetune ARMT on Phonebook (HF dataset), mirroring hyperparams/sweep structure of:
# - scripts/pb/run_rmt_on_phonebook.sh (LR/TBS/steps/warmup/eval cadence)
# and using the ARMT runner:
# - run_armt_on_squad.py (supports "phonebook" datasets too)

export WANDB_PROJECT=phonebook
export HF_Trainer=1

export CUDA_VISIBLE_DEVICES=1

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
N=7

# Resume control:
# - set START_I to start from a later curriculum stage without editing arrays (0-based index)
# - the script will set --model_cpt by picking the latest checkpoint-* from the previous stage
START_I=${START_I:-0}

# Curriculum schedule (edit as needed). Arrays are aligned by index.
N_PAIRS_LIST=(2 4 8 16 32 64)
PER_DEVICE_BATCH_SIZES=(64 32 16 8 4 2)
MAX_STEPS_LIST=(20000 30000 50000 50000 50000 50000)
LR_LIST=(2e-4 1e-4 1e-4 1e-4 1e-4 1e-4)

latest_checkpoint_dir() {
  # Returns the newest checkpoint dir under $1 (HF Trainer convention: checkpoint-<step>).
  local exp_path="$1"
  local ckpt
  ckpt=$(ls -d "${exp_path}"/checkpoint-* 2>/dev/null | sort -V | tail -n 1 || true)
  if [ -n "$ckpt" ]; then
    echo "$ckpt"
  else
    echo ""
  fi
}
START_CPT=./runs/booydar/phonebook_N4/N4/armt2segm_gpt2_mem32_dmem64_seg128_wdm1_rdm1_bs_256_lr_1e-4_constant_with_warmup/run_5/checkpoint-20000
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

  # Curriculum checkpoint: load the latest checkpoint-* from the previous stage output.
  if [ "$i" -eq "$START_I" ]; then
    MODEL_CPT=$START_CPT
  else
    PREV_N_PAIRS=${N_PAIRS_LIST[i-1]}
    PREV_LR=${LR_LIST[i-1]}
    PREV_DATA_NAME="booydar/phonebook_N${PREV_N_PAIRS}"
    PREV_RUN_NAME=armt2segm_${MODEL_NAME}_mem${NUM_MEM_TOKENS}_dmem${D_MEM}_seg${SEGMENT_SIZE}_wdm${WRITING_DEPTH_MULTIPLIER}_rdm${READING_DEPTH_MULTIPLIER}_bs_${TBS}_lr_${PREV_LR}_${LR_SCHEDULER_TYPE}
    PREV_EXP_PATH="./runs/${PREV_DATA_NAME}/N${PREV_N_PAIRS}/${PREV_RUN_NAME}/run_$N"
    MODEL_CPT=$(latest_checkpoint_dir "$PREV_EXP_PATH")
    if [ -z "$MODEL_CPT" ]; then
      echo "WARNING: no checkpoint-* found under PREV_EXP_PATH=$PREV_EXP_PATH; starting from scratch" 1>&2
      MODEL_CPT=$START_CPT
    fi
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

