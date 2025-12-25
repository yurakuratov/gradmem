#!/bin/bash

# Define arguments for the script
NP=${NP:-1}  # Default to 1 process if not set
LR=1e-03
TBS=256
PER_DEVICE_BATCH_SIZE=256
GRAD_ACC_STEPS=$(($TBS/($PER_DEVICE_BATCH_SIZE*$NP)))
USE_GRAD_CKPT=false

MODEL_NAME=gpt2
PRETRAINED_MODEL=gpt2


# GradMemGPT specific parameters
N_CTRL_TOKENS=0
K=1
MAX_STEPS=200000
WARMUP_STEPS=10000
LR_SCHEDULER_TYPE=constant_with_warmup
USE_MEM_PROJ=true
MEM_PROJ_MODE="proj"
USE_WRITE_HEAD=true

INITIAL_CHECKPOINT_PATH=/workspace-SR006.nfs2/bulatov/rmt/test-time/test_time_gd/runs/phonebook/N4/rmt2segm_gpt2_mem32_K1_mem_proj_whead_bs_64_lr_1e-03_constant_with_warmup/run_1/checkpoint-171500/model.safetensors
/workspace-SR006.nfs2/bulatov/rmt/test-time/test_time_gd/runs/phonebook/N8/rmt2segm_gpt2_mem32_K1_mem_proj_whead_bs_256_lr_1e-03_constant_with_warmup-from_N4/run_1/checkpoint-118500/model.safetensors
N_MEM_TOKENS=32
for N_PAIRS in 16 32; do

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
    RUN_NAME=${RUN_NAME}_${LR_SCHEDULER_TYPE}-from_N4

    DATA_NAME="booydar/phonebook_N${N_PAIRS}"
    DATA_PATH="phonebook/N${N_PAIRS}"

    # Run ID
    N=1

    RND=$(date +%Y%m%d%H%M%S)
    # Path to save experiment results
    EXP_PATH="./runs/${DATA_PATH}/${RUN_NAME}/run_$N"

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
      --init_checkpoint $INITIAL_CHECKPOINT_PATH \
      --n_mem_tokens $N_MEM_TOKENS \
      --K $K \
      $( [ "$USE_MEM_PROJ" = true ] && echo "--use_mem_proj" ) \
      $( [ "$USE_MEM_PROJ" = true ] && echo "--mem_proj_mode $MEM_PROJ_MODE" ) \
      $( [ "$USE_WRITE_HEAD" = true ] && echo "--use_write_head" ) \
      --lr_scheduler_type $LR_SCHEDULER_TYPE \
      --max_steps $MAX_STEPS \
      --eval_steps 500 \
      --logging_steps 500 \
      --warmup_steps $WARMUP_STEPS \
      --early_stopping_patience 500 \
      --seed $((142+$N))
  done
done

echo "Done"
