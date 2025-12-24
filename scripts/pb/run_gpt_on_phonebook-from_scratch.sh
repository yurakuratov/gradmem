#!/bin/bash

# Define arguments for the script
NP=${NP:-1}  # Default to 1 process if not set
LR=3e-04
# ADAM_BETA2=0.98
TBS=256
PER_DEVICE_BATCH_SIZE=64
GRAD_ACC_STEPS=$(($TBS/($PER_DEVICE_BATCH_SIZE*$NP)))
MAX_STEPS=200000
WARMUP_STEPS=10000

# MAX_POSITION_EMBEDDINGS=1024
# MODEL_NAME=mamba-130m-hf
# PRETRAINED_MODEL=state-spaces/mamba-130m-hf
MODEL_NAME=gpt2
# PRETRAINED_MODEL=gpt2

# Dataset parameters
# DATA_NAME="babilong_qa1_0k"
# DATA_PATH="./data/${DATA_NAME}"

RUN_NAME="${MODEL_NAME}"

RUN_NAME=${RUN_NAME}_bs_${TBS}_lr_${LR}

if [ -n "$ADAM_BETA2" ]; then
  RUN_NAME=${RUN_NAME}_b2_${ADAM_BETA2}
fi

RUN_NAME=${RUN_NAME}_from_scratch

for N_PAIRS in 4 8 16 32 64; do
  DATA_NAME="phonebook"
  DATA_PATH="booydar/${DATA_NAME}_N${N_PAIRS}"

  # Run ID
  N_VALUES=(1)
  for N in "${N_VALUES[@]}"; do
    # Path to save experiment results
    EXP_PATH="./runs/${DATA_NAME}/N${N_PAIRS}/${RUN_NAME}/run_$N"

    # Execute the script using accelerate for parallel processing
    accelerate launch \
      --main_process_port $((29500+$TBS+$N+1)) \
      --num_processes $NP \
      --mixed_precision bf16 \
      --config_file accelerate.yaml \
      run_gpt2_on_phonebook.py \
      --exp_path $EXP_PATH \
      --per_device_batch_size $PER_DEVICE_BATCH_SIZE \
      --gradient_accumulation_steps $GRAD_ACC_STEPS \
      --total_batch_size $TBS \
      --data_path $DATA_PATH \
      --learning_rate $LR \
      $( [ -n "$ADAM_BETA2" ] && echo "--adam_beta2 $ADAM_BETA2" ) \
      $( [ -n "$MAX_POSITION_EMBEDDINGS" ] && echo "--max_position_embeddings $MAX_POSITION_EMBEDDINGS" ) \
      --base_model $MODEL_NAME \
      --tokenizer_path $MODEL_NAME \
      --max_steps $MAX_STEPS \
      --eval_steps 500 \
      --logging_steps 500 \
      --warmup_steps $WARMUP_STEPS \
      --early_stopping_patience 500 \
      --seed $((142+$N))
  done
done

echo "Done"
