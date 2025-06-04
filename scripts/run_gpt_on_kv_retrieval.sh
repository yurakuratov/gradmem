#!/bin/bash

# Define arguments for the script
NP=${NP:-1}  # Default to 1 process if not set
LR=1e-04
TBS=64
PER_DEVICE_BATCH_SIZE=64
GRAD_ACC_STEPS=$(($TBS/($PER_DEVICE_BATCH_SIZE*$NP)))

L=4
H=4
D=128
BASE_MODEL=gpt2

# Dataset parameters
# DATA_NAME="N2-K4V4-S4(32-64)_1M"
# DATA_NAME="N1-K4V4-S1(16-32)_1M"
DATA_NAME="N10-K2V2-S4(32-64)_1M"
# copy task
# DATA_NAME="N0-S1(4-4)_1M"
DATA_PATH="./data/${DATA_NAME}"


# Run ID
N_VALUES=(1 2 3)
for N in "${N_VALUES[@]}"; do
  # Path to save experiment results
  EXP_PATH="./runs/${DATA_NAME}/${BASE_MODEL}_L${L}H${H}D${D}_bs_${TBS}_lr_${LR}/run_$N"

  # Execute the script using accelerate for parallel processing
  accelerate launch \
    --main_process_port $((29500+$TBS+$N+1)) \
    --num_processes $NP \
    --mixed_precision bf16 \
    --config_file accelerate.yaml \
    run_gpt2_on_kv_retrieval.py \
    --exp_path $EXP_PATH \
    --per_device_batch_size $PER_DEVICE_BATCH_SIZE \
    --gradient_accumulation_steps $GRAD_ACC_STEPS \
    --total_batch_size $TBS \
    --data_path $DATA_PATH \
    --learning_rate $LR \
    --n_layer $L \
    --n_head $H \
    --n_embd $D \
    --base_model $BASE_MODEL \
    --max_steps 200000 \
    --eval_steps 500 \
    --logging_steps 500 \
    --warmup_steps 10000 \
    --early_stopping_patience 500 \
    --seed $((142+$N))
done

echo "Done"