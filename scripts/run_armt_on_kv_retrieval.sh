#!/bin/bash
export CUDA_VISIBLE_DEVICES=1
export WANDB_PROJECT=kv_retrieval
# Define arguments for the script
NP=${NP:-1}  # Default to 1 process if not set
LR=3e-04
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
NUM_PAIRS=32
NUM_PAIRS_PER_SEGMENT=1
DATA_NAME="N${NUM_PAIRS}-K2V2-V${V}"
# DATA_NAME is used as the subset name for HuggingFace Hub dataset irodkin/kv_retrieval
TOKENIZER_PATH="./tokenizers/kv_alphabet_${V}/"

# ARMT specific parameters
NUM_MEM_TOKENS=8
D_MEM=64
SEGMENT_SIZE=$((7*$NUM_PAIRS_PER_SEGMENT))
SEGMENT_ALIGNMENT="left"
LAYERS_ATTR="model.layers"
READING_DEPTH_MULTIPLIER=1
WRITING_DEPTH_MULTIPLIER=1

# INIT_CHECKPOINT=./runs/N32-K2V2-V62_1M/armt_thinking_llama_L4H4D128_mem8_dmem512_seg512_bs_64_lr_1e-04/run_2/checkpoint-195500/model.safetensors
RUN_NAME_SUFFIX=""

RUN_NAME=armt_thinking_${BASE_MODEL}_L${L}H${H}D${D}_mem${NUM_MEM_TOKENS}_dmem${D_MEM}_seg${SEGMENT_SIZE}
if [ "$WRAP_POS" = true ]; then
  RUN_NAME=${RUN_NAME}_wrappos
fi
if [ "$CORRECTION" = false ]; then
  RUN_NAME=${RUN_NAME}_nocorr
fi
if [ "$USE_DENOM" = false ]; then
  RUN_NAME=${RUN_NAME}_nodenom
fi
if [ "$READING_DEPTH_MULTIPLIER" != "1" ]; then
  RUN_NAME=${RUN_NAME}_rdm${READING_DEPTH_MULTIPLIER}
fi
if [ "$WRITING_DEPTH_MULTIPLIER" != "1" ]; then
  RUN_NAME=${RUN_NAME}_wdm${WRITING_DEPTH_MULTIPLIER}
fi

RUN_NAME=${RUN_NAME}_bs_${TBS}_lr_${LR}
if [ -n "$ADAM_BETA2" ]; then
  RUN_NAME=${RUN_NAME}_b2_${ADAM_BETA2}
fi

if [ -n "$RUN_NAME_SUFFIX" ]; then
  RUN_NAME=${RUN_NAME}_${RUN_NAME_SUFFIX}
fi

# Run ID
N_VALUES=(1)
for N in "${N_VALUES[@]}"; do
  # Path to save experiment results
  EXP_PATH="./runs/${DATA_NAME}/${RUN_NAME}/run_$N"

  # Execute the script using accelerate for parallel processing
  export WANDB_NAME=${RUN_NAME}_run_$N
  accelerate launch \
    --main_process_port $((29500+$TBS+$N+200)) \
    --num_processes $NP \
    --mixed_precision 'bf16' \
    --config_file accelerate.yaml \
    run_armt_on_kv_retrieval.py \
    --exp_path $EXP_PATH \
    --per_device_batch_size $PER_DEVICE_BATCH_SIZE \
    --gradient_accumulation_steps $GRAD_ACC_STEPS \
    --total_batch_size $TBS \
    --data_path $DATA_NAME \
    --tokenizer_path $TOKENIZER_PATH \
    --learning_rate $LR \
    $( [ -n "$ADAM_BETA2" ] && echo "--adam_beta2 $ADAM_BETA2" ) \
    --n_layer $L \
    --n_head $H \
    --n_embd $D \
    --base_model $BASE_MODEL \
    $( [ -n "$INIT_CHECKPOINT" ] && echo "--init_checkpoint $INIT_CHECKPOINT" ) \
    --num_mem_tokens $NUM_MEM_TOKENS \
    --d_mem $D_MEM \
    --segment_size $SEGMENT_SIZE \
    --segment_alignment $SEGMENT_ALIGNMENT \
    --layers_attr $LAYERS_ATTR \
    --reading_depth_multiplier $READING_DEPTH_MULTIPLIER \
    --writing_depth_multiplier $WRITING_DEPTH_MULTIPLIER \
    --max_steps 1000000 \
    --eval_steps 50 \
    --logging_steps 50 \
    --warmup_steps 10000 \
    --early_stopping_patience 500 \
    --seed $((142+$N))
done

echo "Done"

