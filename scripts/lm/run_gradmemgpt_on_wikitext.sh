#!/bin/bash

# Define arguments for the script
export CUDA_VISIBLE_DEVICES=0
NP=1
# NP=${NP:-1}  # Default to 1 process if not set

LR=1e-03
TBS=64
PER_DEVICE_BATCH_SIZE=64
GRAD_ACC_STEPS=$(($TBS/($PER_DEVICE_BATCH_SIZE*$NP)))
USE_GRAD_CKPT=false

MODEL_NAME=gpt2
PRETRAINED_MODEL=gpt2


# GradMemGPT specific parameters
N_MEM_TOKENS=32
N_CTRL_TOKENS=0
K=1
INNER_LR=0.4
INNER_CLIP_VALUE=None
INNER_CLIP_NORM=None
USE_ADAM=false
GRAD_MODE="second"
USE_MEM_PROJ=true
MEM_PROJ_MODE="proj"
USE_WRITE_HEAD=true
USE_CONTEXT_LM_LOSS=false

CONTEXT_SIZE=128
SEGMENT_SIZE=128


RUN_NAME=gradmem_${MODEL_NAME}_mem${N_MEM_TOKENS}
if [ "$N_CTRL_TOKENS" -gt 0 ]; then
  RUN_NAME=${RUN_NAME}_c${N_CTRL_TOKENS}
fi
RUN_NAME=${RUN_NAME}_K${K}_ilr${INNER_LR}
if [ "$INNER_CLIP_VALUE" != "None" ]; then
  RUN_NAME=${RUN_NAME}_icv${INNER_CLIP_VALUE}
fi
if [ "$INNER_CLIP_NORM" != "None" ]; then
  RUN_NAME=${RUN_NAME}_icn${INNER_CLIP_NORM}
fi
if [ "$USE_MEM_PROJ" = true ]; then
  RUN_NAME=${RUN_NAME}_mem_proj
  if [ "$MEM_PROJ_MODE" == "per_sample" ]; then
    RUN_NAME=${RUN_NAME}_ps
  fi
fi
if [ "$USE_WRITE_HEAD" = true ]; then
  RUN_NAME=${RUN_NAME}_whead
fi
RUN_NAME=${RUN_NAME}_grad_${GRAD_MODE}
if [ "$USE_ADAM" = true ]; then
  RUN_NAME=${RUN_NAME}_with_adam
fi
if [ "$USE_GRAD_CKPT" = true ]; then
  RUN_NAME=${RUN_NAME}_gc
fi
if [ "$USE_CONTEXT_LM_LOSS" = true ]; then
  RUN_NAME=${RUN_NAME}_closs
fi
RUN_NAME=${RUN_NAME}_bs_${TBS}_lr_${LR}_linear_init_rmt

RUN_NAME=${RUN_NAME}_ctx${CONTEXT_SIZE}-${SEGMENT_SIZE}

DATA_NAME="wikitext"

# Run ID
N_VALUES=(1)
for N in "${N_VALUES[@]}"; do
  RND=$(date +%Y%m%d%H%M%S)
  # Path to save experiment results
  EXP_PATH="/cephfs/home/mkairov/gd_runs/${DATA_NAME}/${RUN_NAME}/run_$N"

  # Execute the script using accelerate for parallel processing
  accelerate launch \
    --main_process_port $((29500+$TBS+$N+1)) \
    --num_processes $NP \
    --mixed_precision bf16 \
    --config_file accelerate.yaml \
    run_gradmemgpt_on_lm.py \
    --exp_path $EXP_PATH \
    --per_device_batch_size $PER_DEVICE_BATCH_SIZE \
    --gradient_accumulation_steps $GRAD_ACC_STEPS \
    --total_batch_size $TBS \
    --dataset_name $DATA_NAME \
    --context_size $CONTEXT_SIZE \
    --segment_size $SEGMENT_SIZE \
    --learning_rate $LR \
    --pretrained_model $PRETRAINED_MODEL \
    --n_mem_tokens $N_MEM_TOKENS \
    --K $K \
    --inner_lr $INNER_LR \
    --use_adam $USE_ADAM \
    --grad_mode $GRAD_MODE \
    $( [ "$INNER_CLIP_VALUE" != "None" ] && echo "--inner_clip_value $INNER_CLIP_VALUE" ) \
    $( [ "$INNER_CLIP_NORM" != "None" ] && echo "--inner_clip_norm $INNER_CLIP_NORM" ) \
    $( [ "$USE_MEM_PROJ" = true ] && echo "--use_mem_proj" ) \
    $( [ "$USE_MEM_PROJ" = true ] && echo "--mem_proj_mode $MEM_PROJ_MODE" ) \
    $( [ "$USE_WRITE_HEAD" = true ] && echo "--use_write_head" ) \
    $( [ "$USE_GRAD_CKPT" = true ] && echo "--use_gradient_checkpointing" ) \
    $( [ "$USE_CONTEXT_LM_LOSS" = true ] && echo "--use_context_lm_loss" ) \
    --lr_scheduler_type linear \
    --max_steps 36000 \
    --eval_steps 500 \
    --logging_steps 500 \
    --warmup_steps 1000 \
    --early_stopping_patience 500 \
    --attn_implementation hvp_manual \
    --seed $((142+$N))
done

echo "Done"
