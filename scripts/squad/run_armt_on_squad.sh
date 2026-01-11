#!/bin/bash
export WANDB_PROJECT=squad
export HF_Trainer=1
# export armt_mask_2d=1
# export NOT_INVERT_ATTN_MASK=1
export CUDA_VISIBLE_DEVICES=0
# Train ARMT on SQuAD (mirrors run_armt_on_kv_retrieval.sh structure, but uses HF SQuAD dataset)

NP=$(echo $CUDA_VISIBLE_DEVICES | awk -F',' '{print NF}')
LR=3e-04
LR_SCHEDULER_TYPE="constant_with_warmup"
TBS=64
PER_DEVICE_BATCH_SIZE=16
GRAD_ACC_STEPS=$(($TBS/($PER_DEVICE_BATCH_SIZE*$NP)))

# Choose a backbone. For GPT-2, layers live under "transformer.h".
PRETRAINED_MODEL=mkairov/gpt2_short_squad
LAYERS_ATTR="transformer.h"

# ARMT parameters
NUM_MEM_TOKENS=32
D_MEM=64
SEGMENT_SIZE=128
SEGMENT_ALIGNMENT="left"
READING_DEPTH_MULTIPLIER=1
WRITING_DEPTH_MULTIPLIER=1

DATA_NAME="squad"

RUN_NAME=armt_thinking_${PRETRAINED_MODEL}_mem${NUM_MEM_TOKENS}_dmem${D_MEM}_seg${SEGMENT_SIZE}_wdm${WRITING_DEPTH_MULTIPLIER}_rdm${READING_DEPTH_MULTIPLIER}_bs_${TBS}_lr_${LR}

# Run ID
N_VALUES=(5)
for N in "${N_VALUES[@]}"; do
  EXP_PATH="./runs/${DATA_NAME}/${RUN_NAME}/run_$N"
  
  export WANDB_NAME=armt_squad_run_${N}

  accelerate launch \
    --main_process_port $((29500+$TBS+$N+11)) \
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
    --max_steps 75000 \
    --eval_steps 500 \
    --logging_steps 500 \
    --warmup_steps 10000 \
    --early_stopping_patience 500 \
    --seed $((142+$N)) \
    --armt_impl old
done

echo "Done"


