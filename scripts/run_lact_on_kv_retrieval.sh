#!/bin/bash
set -e
export CUDA_VISIBLE_DEVICES=0,1
export WANDB_PROJECT=kv_retrieval
NP=$(echo $CUDA_VISIBLE_DEVICES | awk -F',' '{print NF}')
cd ..

LR=3e-04

L=4
D=128

V=62

# LaCT-specific parameters
NUM_ATTN_HEADS=4
NUM_LACT_HEADS=4
INTER_MULTI=1
LACT_CHUNK_SIZE=16
WINDOW_SIZE=64
USE_MUON=true
USE_MOMENTUM=true

TBS=256

NUMS_PAIRS=(2 4 8 16 32 64 96)
BSS=(128 128 128 128 128 128 128)
ITERSS=(10000 10000 10000 20000 100000 100000 100000)
NS=(72 73 74)

for N in "${NS[@]}"; do

MODEL_CPT=None

for (( i=0; i<${#NUMS_PAIRS[@]}; i++ ))
do

NUM_PAIRS=${NUMS_PAIRS[i]}
DATA_NAME="N${NUM_PAIRS}-K2V2-V${V}"
TOKENIZER_PATH="./tokenizers/kv_alphabet_${V}/"

PER_DEVICE_BATCH_SIZE=${BSS[i]}
GRAD_ACC_STEPS=$(($TBS/($PER_DEVICE_BATCH_SIZE*$NP)))

ITERS=${ITERSS[i]}

RUN_NAME=lact_L${L}D${D}_ah${NUM_ATTN_HEADS}_lh${NUM_LACT_HEADS}_cs${LACT_CHUNK_SIZE}_w${WINDOW_SIZE}
if [ "$USE_MUON" = true ]; then
  RUN_NAME=${RUN_NAME}_muon
fi
if [ "$USE_MOMENTUM" = true ]; then
  RUN_NAME=${RUN_NAME}_mom
fi

  EXP_PATH="./runs/${DATA_NAME}/${RUN_NAME}/run_$N"
  if [ $i -eq 0 ]; then
    MODEL_CPT=None
  else
    MODEL_CPT="./runs/N${NUMS_PAIRS[i-1]}-K2V2-V${V}/${RUN_NAME}/run_$N"
  fi

  export WANDB_NAME=lact_cs${LACT_CHUNK_SIZE}_w${WINDOW_SIZE}_muon${USE_MUON}_mom${USE_MOMENTUM}_cur_N${NUM_PAIRS}
  accelerate launch \
    --main_process_port $((29500+$TBS+$N+400)) \
    --num_processes $NP \
    --mixed_precision 'bf16' \
    --config_file accelerate.yaml \
    run_lact_on_kv_retrieval.py \
    --exp_path $EXP_PATH \
    --per_device_batch_size $PER_DEVICE_BATCH_SIZE \
    --gradient_accumulation_steps $GRAD_ACC_STEPS \
    --total_batch_size $TBS \
    --data_path $DATA_NAME \
    --tokenizer_path $TOKENIZER_PATH \
    --learning_rate $LR \
    --weight_decay 5.0 \
    --n_layer $L \
    --n_embd $D \
    --num_attn_heads $NUM_ATTN_HEADS \
    --num_lact_heads $NUM_LACT_HEADS \
    --inter_multi $INTER_MULTI \
    --lact_chunk_size $LACT_CHUNK_SIZE \
    --window_size $WINDOW_SIZE \
    $( [ "$USE_MUON" = true ] && echo "--use_muon" ) \
    $( [ "$USE_MOMENTUM" = true ] && echo "--use_momentum" ) \
    --max_steps $ITERS \
    --eval_steps 100 \
    --logging_steps 100 \
    --warmup_steps 1000 \
    --early_stopping_patience 500 \
    --seed $((142+$N+$i)) \
    --model_cpt $MODEL_CPT
done
done

echo "Done"
