#!/bin/bash

# Define arguments for the script
NP=${NP:-1}  # Default to 1 process if not set
LR=1e-04
# ADAM_BETA2=0.95
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
# DATA_NAME="N8-K1V1-vocab512-no_noise_1M"
DATA_NAME="N16-K2V2-V${V}_1M"
DATA_PATH="./data/${DATA_NAME}"
TOKENIZER_PATH="./tokenizers/kv_alphabet_${V}/"

# RMT specific parameters
N_MEM_TOKENS=8
N_CTRL_TOKENS=0
K=2
USE_MEM_PROJ=true
MEM_PROJ_MODE="proj"
USE_RECONSTRUCTION_LOSS=false
RECONSTRUCTION_LOSS_WEIGHT=1.0
USE_WRITE_HEAD=false
USE_MEM_RESIDUAL=true
ATTN_IMPLEMENTATION="eager"
# ATTN_IMPLEMENTATION="flash_attention_2"

# INIT_CHECKPOINT=./runs/N16-K2V2-V62_1M/rmt2segm_llama_L4H4D128_mem8_K1_mem_proj_rw_whead_rec_loss_w1.0_bs_64_lr_1e-04/run_2/checkpoint-294500/model.safetensors
# INIT_CHECKPOINT=./runs/N16-K2V2-V62_1M/rmt2segm_llama_L4H4D128_mem8_K2_mem_proj_rw_whead_rec_loss_w1.0_bs_64_lr_1e-04/run_3/checkpoint-534500/model.safetensors
# RUN_NAME_SUFFIX=init_N16_K2_FA2

RUN_NAME=rmt2segm_${BASE_MODEL}_L${L}H${H}D${D}_mem${N_MEM_TOKENS}_K${K}
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

# Run ID
N_VALUES=(1 2)
for N in "${N_VALUES[@]}"; do
  # Path to save experiment results
  EXP_PATH="./runs/${DATA_NAME}/${RUN_NAME}/run_$N"

  # Execute the script using accelerate for parallel processing
  accelerate launch \
    --main_process_port $((29500+100*$K+$TBS+$N+920)) \
    --num_processes $NP \
    --mixed_precision 'no' \
    --config_file accelerate.yaml \
    run_rmt_on_kv_retrieval.py \
    --exp_path $EXP_PATH \
    --per_device_batch_size $PER_DEVICE_BATCH_SIZE \
    --gradient_accumulation_steps $GRAD_ACC_STEPS \
    --total_batch_size $TBS \
    --data_path $DATA_PATH \
    --tokenizer_path $TOKENIZER_PATH \
    --learning_rate $LR \
    $( [ -n "$ADAM_BETA2" ] && echo "--adam_beta2 $ADAM_BETA2" ) \
    --n_layer $L \
    --n_head $H \
    --n_embd $D \
    --base_model $BASE_MODEL \
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
    --seed $((142+$N))
done

echo "Done"
