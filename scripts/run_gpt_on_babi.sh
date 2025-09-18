#!/bin/bash

# Define arguments for the script
NP=${NP:-1}  # Default to 1 process if not set
LR=1e-04
# ADAM_BETA2=0.98
TBS=64
PER_DEVICE_BATCH_SIZE=$TBS
GRAD_ACC_STEPS=$(($TBS/($PER_DEVICE_BATCH_SIZE*$NP)))

# MAX_POSITION_EMBEDDINGS=1024
MODEL_NAME=mamba-130m-hf
PRETRAINED_MODEL=state-spaces/mamba-130m-hf

# Dataset parameters
# DATA_NAME="babilong_qa1_0k"
# DATA_PATH="./data/${DATA_NAME}"

RUN_NAME="${MODEL_NAME}"

RUN_NAME=${RUN_NAME}_bs_${TBS}_lr_${LR}

if [ -n "$ADAM_BETA2" ]; then
  RUN_NAME=${RUN_NAME}_b2_${ADAM_BETA2}
fi

for task_name in "qa1" "qa2" "qa3" "qa4" "qa5"; do
  DATA_NAME="babilong_${task_name}_0k"
  DATA_PATH="./data/${DATA_NAME}"

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
      run_gpt2_on_kv_retrieval.py \
      --exp_path $EXP_PATH \
      --per_device_batch_size $PER_DEVICE_BATCH_SIZE \
      --gradient_accumulation_steps $GRAD_ACC_STEPS \
      --total_batch_size $TBS \
      --data_path $DATA_PATH \
      --learning_rate $LR \
      $( [ -n "$ADAM_BETA2" ] && echo "--adam_beta2 $ADAM_BETA2" ) \
      $( [ -n "$MAX_POSITION_EMBEDDINGS" ] && echo "--max_position_embeddings $MAX_POSITION_EMBEDDINGS" ) \
      --pretrained_model $PRETRAINED_MODEL \
      --max_steps 200000 \
      --eval_steps 500 \
      --logging_steps 500 \
      --warmup_steps 10000 \
      --early_stopping_patience 500 \
      --seed $((142+$N))
  done
done

echo "Done"
