#!/bin/bash
set -e
export CUDA_VISIBLE_DEVICES=0,1
export WANDB_PROJECT=kv_retrieval
# Define arguments for the script
NP=$(echo $CUDA_VISIBLE_DEVICES | awk -F',' '{print NF}')
LR=3e-04
# ADAM_BETA2=0.95


L=4
H=4
D=128
BASE_MODEL=llama
NUM_MEM_TOKENS=8
D_MEM=64

V=62
# Dataset parameters
# DATA_NAME="N2-K4V4-S4(32-64)_1M"
# DATA_NAME="N2-K4V4-S1(16-32)_1M"
# DATA_NAME="N2-K4V4-S2(16-32)_1M"
# DATA_NAME="N0-S1(4-4)_1M"
# DATA_NAME="N10-K2V2-S4(32-64)_1M"
# DATA_NAME="N8-K1V1-vocab512-no_noise_1M"

TBS=256

NUMS_PAIRS=(256)
NUMS_PAIRS_PER_SEGMENT=(1)
BSS=(32)
ITERSS=(30000)
NS=(22)

for N in "${NS[@]}"; do

MODEL_CPT=None

for (( i=0; i<${#NUMS_PAIRS[@]}; i++ ))
do

NUM_PAIRS=${NUMS_PAIRS[i]}
NUM_PAIRS_PER_SEGMENT=${NUMS_PAIRS_PER_SEGMENT[i]}
DATA_NAME="N${NUM_PAIRS}-K2V2-V${V}"
# DATA_NAME is used as the subset name for HuggingFace Hub dataset irodkin/kv_retrieval
TOKENIZER_PATH="./tokenizers/kv_alphabet_${V}/"

PER_DEVICE_BATCH_SIZE=${BSS[i]}
GRAD_ACC_STEPS=$(($TBS/($PER_DEVICE_BATCH_SIZE*$NP)))

ITERS=${ITERSS[i]}
# ARMT specific parameters
SEGMENT_SIZE=$((7*$NUM_PAIRS_PER_SEGMENT))
SEGMENT_ALIGNMENT="left"
LAYERS_ATTR="model.layers"

READING_DEPTH_MULTIPLIER=1
WRITING_DEPTH_MULTIPLIER=1
REPEAT_READ_SEGMENTS=1
REPEAT_WRITE_SEGMENTS=1


RUN_NAME=armt_thinking_${BASE_MODEL}_L${L}H${H}D${D}_mem${NUM_MEM_TOKENS}_dmem${D_MEM}_seg${SEGMENT_SIZE}_wdm${WRITING_DEPTH_MULTIPLIER}_rdm${READING_DEPTH_MULTIPLIER}_repW${REPEAT_WRITE_SEGMENTS}_repR${REPEAT_READ_SEGMENTS}


# Run ID

  # Path to save experiment results
  EXP_PATH="./runs/${DATA_NAME}/${RUN_NAME}/run_$N"
  if [ $i -eq 0 ]; then
    MODEL_CPT=../armt_pps1_N128/
  else
    PAST_SEGMENT_SIZE=$((7*NUMS_PAIRS_PER_SEGMENT[i-1]))
    MODEL_CPT="./runs/N${NUMS_PAIRS[i-1]}-K2V2-V${V}/armt_thinking_${BASE_MODEL}_L${L}H${H}D${D}_mem${NUM_MEM_TOKENS}_dmem${D_MEM}_seg${PAST_SEGMENT_SIZE}_wdm${WRITING_DEPTH_MULTIPLIER}_rdm${READING_DEPTH_MULTIPLIER}_repW${REPEAT_WRITE_SEGMENTS}_repR${REPEAT_READ_SEGMENTS}/run_$N"
  fi
  # Execute the script using accelerate for parallel processing
  export WANDB_NAME=armt_w${WRITING_DEPTH_MULTIPLIER}_r${READING_DEPTH_MULTIPLIER}_repW${REPEAT_WRITE_SEGMENTS}_repR${REPEAT_READ_SEGMENTS}_cur_N${NUM_PAIRS}_pps${NUM_PAIRS_PER_SEGMENT}
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
    --weight_decay 5.0 \
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
    --max_steps $ITERS \
    --eval_steps 100 \
    --logging_steps 100 \
    --warmup_steps 1000 \
    --early_stopping_patience 500 \
    --seed $((142+$N+$i)) \
    --model_cpt $MODEL_CPT \
    --repeat_read_segments $REPEAT_READ_SEGMENTS \
    --repeat_write_segments $REPEAT_WRITE_SEGMENTS
done
done

echo "Done"

