#!/bin/bash
set -e
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUBLAS_WORKSPACE_CONFIG=:4096:2
export CUDA_LAUNCH_BLOCKING=1
export CUDA_VISIBLE_DEVICES=2,3
NP=2

LR=1e-04
# ADAM_BETA2=0.95
TBS=64
PER_DEVICE_BATCH_SIZE=32
GRAD_ACC_STEPS=$(($TBS/($PER_DEVICE_BATCH_SIZE*$NP)))

MODEL_NAME=gpt2
PRETRAINED_MODEL=gpt2


# RMT specific parameters
N_MEM_TOKENS=32
N_CTRL_TOKENS=0
K=2
USE_MEM_PROJ=true
MEM_PROJ_MODE="proj"
USE_RECONSTRUCTION_LOSS=true
RECONSTRUCTION_LOSS_WEIGHT=1.0
USE_WRITE_HEAD=false
USE_MEM_RESIDUAL=false
ATTN_IMPLEMENTATION="sdpa"

RUN_NAME=rmt2segm_${MODEL_NAME}_L${L}H${H}D${D}_mem${N_MEM_TOKENS}_K${K}
if [ "$N_CTRL_TOKENS" -gt 0 ]; then
  RUN_NAME=${RUN_NAME}_c${N_CTRL_TOKENS}
fi
if [ "$USE_MEM_PROJ" = true ]; then
  RUN_NAME=${RUN_NAME}_mem_${MEM_PROJ_MODE}
fi

if [ "$USE_WRITE_HEAD" = true ]; then
  RUN_NAME=${RUN_NAME}_whead
fi

if [ "$USE_MEM_RESIDUAL" = true ]; then
  RUN_NAME=${RUN_NAME}_res
fi

if [ "$USE_RECONSTRUCTION_LOSS" = true ]; then
  RUN_NAME=${RUN_NAME}_rec_loss_w${RECONSTRUCTION_LOSS_WEIGHT}
fi

RUN_NAME=${RUN_NAME}_bs_${TBS}_lr_${LR}
if [ -n "$ADAM_BETA2" ]; then
  RUN_NAME=${RUN_NAME}_b2_${ADAM_BETA2}
fi

if [ -n "$RUN_NAME_SUFFIX" ]; then
  RUN_NAME=${RUN_NAME}_${RUN_NAME_SUFFIX}
fi

DATA_NAME="squad_short"
DATA_PATH="./data/${DATA_NAME}"
TOKENIZER_PATH="gpt2"

# Run ID
N_VALUES=(1)
for N in "${N_VALUES[@]}"; do
  # Path to save experiment results
  EXP_PATH="/cephfs/home/mkairov/gd_runs/${DATA_NAME}/${RUN_NAME}/run_$N"

  # Execute the script using accelerate for parallel processing
  accelerate launch \
    --main_process_port $((29500+100*$K+$TBS+$N+920)) \
    --num_processes $NP \
    --mixed_precision 'no' \
    --config_file accelerate.yaml \
    run_rmt_on_short_squad.py \
    --exp_path $EXP_PATH \
    --per_device_batch_size $PER_DEVICE_BATCH_SIZE \
    --gradient_accumulation_steps $GRAD_ACC_STEPS \
    --total_batch_size $TBS \
    --data_path $DATA_PATH \
    --tokenizer_path $TOKENIZER_PATH \
    --learning_rate $LR \
    $( [ -n "$ADAM_BETA2" ] && echo "--adam_beta2 $ADAM_BETA2" ) \
    --attn_implementation $ATTN_IMPLEMENTATION \
    $( [ -n "$INIT_CHECKPOINT" ] && echo "--init_checkpoint $INIT_CHECKPOINT" ) \
    --n_mem_tokens $N_MEM_TOKENS \
    --K $K \
    $( [ "$USE_MEM_PROJ" = true ] && echo "--use_mem_proj" ) \
    $( [ "$USE_MEM_PROJ" = true ] && echo "--mem_proj_mode $MEM_PROJ_MODE" ) \
    $( [ "$USE_RECONSTRUCTION_LOSS" = true ] && echo "--use_reconstruction_loss" ) \
    $( [ "$USE_RECONSTRUCTION_LOSS" = true ] && echo "--reconstruction_loss_weight $RECONSTRUCTION_LOSS_WEIGHT" ) \
    $( [ "$USE_WRITE_HEAD" = true ] && echo "--use_write_head" ) \
    $( [ "$USE_MEM_RESIDUAL" = true ] && echo "--use_mem_residual" ) \
    --max_steps 1000000 \
    --eval_steps 500 \
    --logging_steps 500 \
    --warmup_steps 2000 \
    --early_stopping_patience 500 \
    --seed $((142+$N)) \
    --pretrained_model $PRETRAINED_MODEL \
    --init_base_checkpoint "/cephfs/home/mkairov/gd_runs/short_squad/gpt2_L4H4D128_bs_64_lr_1e-04/run_2/checkpoint-5500/model.safetensors"

done

echo "Done"
