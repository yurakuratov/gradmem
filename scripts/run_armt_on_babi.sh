#!/bin/bash
set -e
export CUDA_VISIBLE_DEVICES=1
export WANDB_PROJECT=babi
# Define arguments for the script
NP=$(echo $CUDA_VISIBLE_DEVICES | awk -F',' '{print NF}')
LR=1e-04
# ADAM_BETA2=0.95


MODEL_NAME=gpt2
PRETRAINED_MODEL=gpt2

TBS=64
NUM_MEM_TOKENS=8
D_MEM=64

BSS=(16)
ITERSS=(30000)
NS=(22 23 24)

for N in "${NS[@]}"; do

MODEL_CPT=None

for task_name in "qa1" "qa4" "qa5"; do
  DATA_NAME="babilong_${task_name}_0k"
  DATA_PATH="./data/${DATA_NAME}"


PER_DEVICE_BATCH_SIZE=${BSS[i]}
GRAD_ACC_STEPS=$(($TBS/($PER_DEVICE_BATCH_SIZE*$NP)))

ITERS=${ITERSS[i]}
# ARMT specific parameters
SEGMENT_SIZE=-1
SEGMENT_ALIGNMENT="left"
LAYERS_ATTR="transformer.h"

READING_DEPTH_MULTIPLIER=1
WRITING_DEPTH_MULTIPLIER=1


RUN_NAME=armt_thinking_${MODEL_NAME}_mem${NUM_MEM_TOKENS}


# Run ID

  # Path to save experiment results
  EXP_PATH="./runs/${DATA_NAME}/${RUN_NAME}/run_$N"
  # Execute the script using accelerate for parallel processing
  export WANDB_NAME=armt_${task_name}
  accelerate launch \
    --main_process_port $((29500+$TBS+$N+1)) \
    --num_processes $NP \
    --mixed_precision 'bf16' \
    --config_file accelerate.yaml \
    run_armt_on_kv_retrieval.py \
    --exp_path $EXP_PATH \
    --per_device_batch_size $PER_DEVICE_BATCH_SIZE \
    --gradient_accumulation_steps $GRAD_ACC_STEPS \
    --total_batch_size $TBS \
    --data_path $DATA_PATH \
    --pretrained_model $PRETRAINED_MODEL \
    --tokenizer_path $PRETRAINED_MODEL \
    --learning_rate $LR \
    $( [ -n "$ADAM_BETA2" ] && echo "--adam_beta2 $ADAM_BETA2" ) \
    --weight_decay 5.0 \
    --num_mem_tokens $NUM_MEM_TOKENS \
    --d_mem $D_MEM \
    --segment_size $SEGMENT_SIZE \
    --segment_alignment $SEGMENT_ALIGNMENT \
    --layers_attr $LAYERS_ATTR \
    --reading_depth_multiplier $READING_DEPTH_MULTIPLIER \
    --writing_depth_multiplier $WRITING_DEPTH_MULTIPLIER \
    --max_steps $ITERS \
    --eval_steps 100 \
    --logging_steps 100 \
    --warmup_steps 1000 \
    --early_stopping_patience 500 \
    --seed $((142+$N)) \
    --armt_impl old
done
done

echo "Done"

