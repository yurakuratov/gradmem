#!/bin/bash
set -e
export CUDA_VISIBLE_DEVICES=0,1
export WANDB_PROJECT=kv_retrieval
NP=$(echo $CUDA_VISIBLE_DEVICES | awk -F',' '{print NF}')

cd .. 

LR=3e-04
L=4
H=4
D=128

V=62

# TTT-specific parameters
TTT_LAYER_TYPE=linear
TTT_BASE_LR=1.0
MINI_BATCH_SIZE=16

TBS=256

NUMS_PAIRS=(2 4 8 16 32 64 96)
BSS=(128 128 128 128 128 128 128)
ITERSS=(10000 10000 10000 20000 30000 30000 30000 30000)
NS=(73 74 75)

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

RUN_NAME=ttt_${TTT_LAYER_TYPE}_L${L}H${H}D${D}_mbs${MINI_BATCH_SIZE}_tttlr${TTT_BASE_LR}

  EXP_PATH="./runs/${DATA_NAME}/${RUN_NAME}/run_$N"
  if [ $i -eq 0 ]; then
    MODEL_CPT=None
  else
    # MODEL_CPT="./runs/N${NUMS_PAIRS[i-1]}-K2V2-V${V}/${RUN_NAME}/run_$N"
    MODEL_CPT=None
  fi

  export WANDB_NAME=ttt_${TTT_LAYER_TYPE}_mbs${MINI_BATCH_SIZE}_tttlr${TTT_BASE_LR}_N${NUM_PAIRS}
  accelerate launch \
    --main_process_port $((29500+$TBS+$N+300)) \
    --num_processes $NP \
    --mixed_precision 'bf16' \
    --config_file accelerate.yaml \
    run_ttt_on_kv_retrieval.py \
    --exp_path $EXP_PATH \
    --per_device_batch_size $PER_DEVICE_BATCH_SIZE \
    --gradient_accumulation_steps $GRAD_ACC_STEPS \
    --total_batch_size $TBS \
    --data_path $DATA_NAME \
    --tokenizer_path $TOKENIZER_PATH \
    --learning_rate $LR \
    --weight_decay 5.0 \
    --n_layer $L \
    --n_head $H \
    --n_embd $D \
    --ttt_layer_type $TTT_LAYER_TYPE \
    --ttt_base_lr $TTT_BASE_LR \
    --mini_batch_size $MINI_BATCH_SIZE \
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
