#!/bin/bash

# Define arguments for the script
NP=${NP:-1}  # Default to 1 process if not set
LR=3e-04
LR_SCHEDULER_TYPE="cosine"
TBS=64
PER_DEVICE_BATCH_SIZE=16
GRAD_ACC_STEPS=$(($TBS/($PER_DEVICE_BATCH_SIZE*$NP)))
USE_GRAD_CKPT=false

MODEL_NAME=gpt2
PRETRAINED_MODEL=/workspace-SR006.nfs2/bulatov/rmt/models/gpt2_checkpoint_13500


# GradMemGPT specific parameters
N_CTRL_TOKENS=0
K=1
for N_MEM_TOKENS in 32; do
  USE_MEM_PROJ=true
  MEM_PROJ_MODE="proj"
  USE_WRITE_HEAD=true

  RUN_NAME=rmt2segm_${MODEL_NAME}_mem${N_MEM_TOKENS}
  if [ "$N_CTRL_TOKENS" -gt 0 ]; then
    RUN_NAME=${RUN_NAME}_c${N_CTRL_TOKENS}
  fi
  RUN_NAME=${RUN_NAME}_K${K}
  if [ "$USE_MEM_PROJ" = true ]; then
    RUN_NAME=${RUN_NAME}_mem_proj
    if [ "$MEM_PROJ_MODE" == "per_sample" ]; then
      RUN_NAME=${RUN_NAME}_ps
    fi
  fi
  if [ "$USE_WRITE_HEAD" = true ]; then
    RUN_NAME=${RUN_NAME}_whead
  fi
  RUN_NAME=${RUN_NAME}_bs_${TBS}_lr_${LR}
  RUN_NAME=${RUN_NAME}_${LR_SCHEDULER_TYPE}
  RUN_NAME=${RUN_NAME}_from_gpt2_13k

  DATA_NAME="squad"

  # Run ID
  N_VALUES=(1 2 3)
  for N in "${N_VALUES[@]}"; do
    RND=$(date +%Y%m%d%H%M%S)
    # Path to save experiment results
    EXP_PATH="./runs/${DATA_NAME}/${RUN_NAME}/run_$N"

    # Execute the script using accelerate for parallel processing
    accelerate launch \
      --main_process_port $((29500+$TBS+$N+1)) \
      --num_processes $NP \
      --mixed_precision bf16 \
      --config_file accelerate.yaml \
      run_rmt_on_squad.py \
      --exp_path $EXP_PATH \
      --per_device_batch_size $PER_DEVICE_BATCH_SIZE \
      --gradient_accumulation_steps $GRAD_ACC_STEPS \
      --total_batch_size $TBS \
      --dataset_name $DATA_NAME \
      --learning_rate $LR \
      --pretrained_model $PRETRAINED_MODEL \
      --tokenizer gpt2\
      --n_mem_tokens $N_MEM_TOKENS \
      --K $K \
      $( [ "$USE_MEM_PROJ" = true ] && echo "--use_mem_proj" ) \
      $( [ "$USE_MEM_PROJ" = true ] && echo "--mem_proj_mode $MEM_PROJ_MODE" ) \
      $( [ "$USE_WRITE_HEAD" = true ] && echo "--use_write_head" ) \
      --lr_scheduler_type $LR_SCHEDULER_TYPE \
      --max_steps 75000 \
      --eval_steps 500 \
      --logging_steps 500 \
      --warmup_steps 10000 \
      --early_stopping_patience 1000 \
      --seed $((142+$N))
  done
done

echo "Done"
