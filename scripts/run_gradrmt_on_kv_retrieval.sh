#!/bin/bash
set -e
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUBLAS_WORKSPACE_CONFIG=:4096:2
export CUDA_LAUNCH_BLOCKING=1
export CUDA_VISIBLE_DEVICES=1
NP=${NP:-1}  # Default to 1 process if not set

# Define arguments for the script
LR=1e-04
TBS=64
PER_DEVICE_BATCH_SIZE=64
GRAD_ACC_STEPS=$(($TBS/($PER_DEVICE_BATCH_SIZE*$NP)))

L=4
H=4
D=128
BASE_MODEL=llama

V=62
# Dataset parameters
# DATA_NAME="N2-K4V4-S4(32-64)_1M"
# DATA_NAME="N2-K4V4-S1(16-32)_1M"
# DATA_NAME="N2-K4V4-S2(16-32)_1M"
# DATA_NAME="N0-S1(4-4)_1M"
# DATA_NAME="N10-K2V2-S4(32-64)_1M"
# DATA_NAME="N16-K1V1-vocab512_1M"
DATA_NAME="N4-K2V2-V${V}_1M"
SEGMENT_SIZE=14
DATA_PATH="./data/${DATA_NAME}"
TOKENIZER_PATH="./tokenizers/kv_alphabet_${V}/"

# GradMemGPT specific parameters
N_MEM_TOKENS=8
N_CTRL_TOKENS=0
K=3
INNER_LR=0.04
INNER_CLIP_VALUE=None
INNER_CLIP_NORM=None
INNER_OPTIM="sgd"
GRAD_MODE="second"
MOMENTUM_MODE="second"

USE_MEM_PROJ=true
MEM_PROJ_MODE="proj"
USE_WRITE_HEAD=true

RUN_NAME=gradmem_${BASE_MODEL}_L${L}H${H}D${D}_mem${N_MEM_TOKENS}
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

RUN_NAME=${RUN_NAME}_${INNER_OPTIM}_grad_${GRAD_MODE}_m_${MOMENTUM_MODE}_bs_${TBS}_lr_${LR}

# Run ID
N_VALUES=(1 2)
for N in "${N_VALUES[@]}"; do
  # Path to save experiment results
  EXP_PATH="./runs/${DATA_NAME}/${RUN_NAME}/run_$N"

  # Execute the script using accelerate for parallel processing
  accelerate launch \
    --main_process_port $((29500+$TBS+$N+1)) \
    --num_processes $NP \
    --mixed_precision 'no' \
    --config_file accelerate.yaml \
    run_gradrmt_on_kv_retrieval.py \
    --exp_path $EXP_PATH \
    --per_device_batch_size $PER_DEVICE_BATCH_SIZE \
    --gradient_accumulation_steps $GRAD_ACC_STEPS \
    --total_batch_size $TBS \
    --data_path $DATA_NAME \
    --tokenizer_path $TOKENIZER_PATH \
    --learning_rate $LR \
    --n_layer $L \
    --n_head $H \
    --n_embd $D \
    --base_model $BASE_MODEL \
    --n_mem_tokens $N_MEM_TOKENS \
    --K $K \
    --inner_lr $INNER_LR \
    --inner_optim $INNER_OPTIM \
    --grad_mode $GRAD_MODE \
    --momentum_mode $MOMENTUM_MODE \
    --segment_size $SEGMENT_SIZE \
    $( [ "$INNER_CLIP_VALUE" != "None" ] && echo "--inner_clip_value $INNER_CLIP_VALUE" ) \
    $( [ "$INNER_CLIP_NORM" != "None" ] && echo "--inner_clip_norm $INNER_CLIP_NORM" ) \
    $( [ "$USE_MEM_PROJ" = true ] && echo "--use_mem_proj" ) \
    $( [ "$USE_MEM_PROJ" = true ] && echo "--mem_proj_mode $MEM_PROJ_MODE" ) \
    $( [ "$USE_WRITE_HEAD" = true ] && echo "--use_write_head" ) \
    --max_steps 200000 \
    --eval_steps 500 \
    --logging_steps 500 \
    --warmup_steps 10000 \
    --early_stopping_patience 500 \
    --seed $((142+$N))
done

echo "Done"
