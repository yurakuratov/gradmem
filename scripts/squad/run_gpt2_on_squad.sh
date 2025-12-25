#!/bin/bash

# Define arguments for the script
NP=${NP:-1}  # Default to 1 process if not set
LR=1e-04
TBS=64
PER_DEVICE_BATCH_SIZE=32
GRAD_ACC_STEPS=$(($TBS/($PER_DEVICE_BATCH_SIZE*$NP)))

MODEL_NAME=gpt2
PRETRAINED_MODEL=gpt2

RUN_NAME=${MODEL_NAME}_bs_${TBS}_lr_${LR}

DATA_NAME="squad"

# Run ID
N_VALUES=(1)
for N in "${N_VALUES[@]}"; do
  # Path to save experiment results
  EXP_PATH="./runs/${DATA_NAME}/${RUN_NAME}/run_$N"

  # Execute the script using accelerate for parallel processing
  accelerate launch \
    --main_process_port $((29500+$TBS+$N+1)) \
    --num_processes $NP \
    --mixed_precision bf16 \
    --config_file accelerate.yaml \
    run_gpt2_on_squad.py \
    --exp_path $EXP_PATH \
    --per_device_batch_size $PER_DEVICE_BATCH_SIZE \
    --gradient_accumulation_steps $GRAD_ACC_STEPS \
    --total_batch_size $TBS \
    --dataset_name $DATA_NAME \
    --learning_rate $LR \
    --pretrained_model $PRETRAINED_MODEL \
    --max_steps 200000 \
    --eval_steps 500 \
    --logging_steps 500 \
    --warmup_steps 10000 \
    --early_stopping_patience 500 \
    --seed $((142+$N))
done

echo "Done"
