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
BASE_MODEL=pythia

# Dataset parameters
DATA_NAME="N2-K4V4-S4(32-64)_1M"
# DATA_NAME="N2-K4V4-S1(16-32)_1M"
# DATA_NAME="N2-K4V4-S2(16-32)_1M"
# DATA_NAME="N0-S1(4-4)_1M"
# DATA_NAME="N10-K2V2-S4(32-64)_1M"
DATA_PATH="./data/${DATA_NAME}"

# Run ID
N=1
# GradMemGPT specific parameters
N_MEM_TOKENS=8
N_CTRL_TOKENS=0
K=10
INNER_LR=1.0
INNER_CLIP_VALUE=None
INNER_CLIP_NORM=1.0
USE_ADAM=false
GRAD_MODE="second"

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
RUN_NAME=${RUN_NAME}_grad_${GRAD_MODE}
if [ "$USE_ADAM" = true ]; then
  RUN_NAME=${RUN_NAME}_with_adam
fi
RUN_NAME=${RUN_NAME}_bs_${TBS}_lr_${LR}

# Path to save experiment results
EXP_PATH="./runs/${DATA_NAME}/${RUN_NAME}/run_$N"

# Execute the script using accelerate for parallel processing
accelerate launch \
  --main_process_port $((29500+$TBS+$N+1)) \
  --num_processes $NP \
  --mixed_precision bf16 \
  --config_file accelerate.yaml \
  run_gradmemgpt_on_kv_retrieval.py \
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
  --n_mem_tokens $N_MEM_TOKENS \
  --K $K \
  --inner_lr $INNER_LR \
  --use_adam $USE_ADAM \
  --grad_mode $GRAD_MODE \
  $( [ "$INNER_CLIP_VALUE" != "None" ] && echo "--inner_clip_value $INNER_CLIP_VALUE" ) \
  $( [ "$INNER_CLIP_NORM" != "None" ] && echo "--inner_clip_norm $INNER_CLIP_NORM" ) \
  --max_steps 200000 \
  --eval_steps 500 \
  --logging_steps 500 \
  --warmup_steps 10000 \
  --early_stopping_patience 500 \
  --seed $((142+$N))

echo "Done"