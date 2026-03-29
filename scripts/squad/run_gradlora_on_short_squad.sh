#!/bin/bash
export WANDB_PROJECT=squad
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}

NP=$(echo $CUDA_VISIBLE_DEVICES | awk -F',' '{print NF}')

cd ../..

LR=${LR:-1e-04}
LR_SCHEDULER_TYPE="constant_with_warmup"
TBS=${TBS:-64}
PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE:-32}
GRAD_ACC_STEPS=$(($TBS/($PER_DEVICE_BATCH_SIZE*$NP)))
USE_GRAD_CKPT=false

PRETRAINED_MODEL=${PRETRAINED_MODEL:-mkairov/gpt2_short_squad}

# GradLoRA parameters
LORA_MODE=${LORA_MODE:-residual}      # "residual" or "ffn"
LAYER_IDX=${LAYER_IDX:-6}
RANK=${RANK:-16}
K=${K:-2}
LAST_K_SECOND_ORDER=${LAST_K_SECOND_ORDER:-$K}
INNER_LR=${INNER_LR:-0.03}
INNER_CLIP_VALUE=None
INNER_CLIP_NORM=None
USE_ADAM=${USE_ADAM:-false}
GRAD_MODE=${GRAD_MODE:-second}

DATA_NAME=${DATA_NAME:-mkairov/short_squad}

# Build run name
if [ "$LORA_MODE" = "ffn" ]; then
    MODE_TAG="ffn"
else
    MODE_TAG="lora"
fi
RUN_NAME=grad${MODE_TAG}_${PRETRAINED_MODEL##*/}_l${LAYER_IDX}_r${RANK}_K${K}_ilr${INNER_LR}
if [ "$LAST_K_SECOND_ORDER" != "$K" ] && [ "$GRAD_MODE" == "second" ]; then
    RUN_NAME=${RUN_NAME}_last_K${LAST_K_SECOND_ORDER}
fi
RUN_NAME=${RUN_NAME}_grad_${GRAD_MODE}
if [ "$USE_ADAM" = true ]; then
    RUN_NAME=${RUN_NAME}_with_adam
fi
if [ "$USE_GRAD_CKPT" = true ]; then
    RUN_NAME=${RUN_NAME}_gc
fi
RUN_NAME=${RUN_NAME}_bs_${TBS}_lr_${LR}

N_VALUES=(${N_VALUES:-6})
for N in "${N_VALUES[@]}"; do
    EXP_PATH="./runs/${DATA_NAME}/${RUN_NAME}/run_$N"

    export WANDB_NAME=gradlora_squad_run_${N}

    accelerate launch \
        --main_process_port $((29500+$TBS+$N+21)) \
        --num_processes $NP \
        --mixed_precision bf16 \
        --config_file accelerate.yaml \
        run_armt_on_squad.py \
        --model_class gradlora \
        --lora_mode $LORA_MODE \
        --layer_idx $LAYER_IDX \
        --rank $RANK \
        --exp_path $EXP_PATH \
        --per_device_batch_size $PER_DEVICE_BATCH_SIZE \
        --gradient_accumulation_steps $GRAD_ACC_STEPS \
        --total_batch_size $TBS \
        --dataset_name $DATA_NAME \
        --learning_rate $LR \
        --lr_scheduler_type $LR_SCHEDULER_TYPE \
        --pretrained_model $PRETRAINED_MODEL \
        --K $K \
        --last_K_second_order $LAST_K_SECOND_ORDER \
        --inner_lr $INNER_LR \
        --use_inner_adam $USE_ADAM \
        --grad_mode $GRAD_MODE \
        $( [ "$INNER_CLIP_VALUE" != "None" ] && echo "--inner_clip_value $INNER_CLIP_VALUE" ) \
        $( [ "$INNER_CLIP_NORM" != "None" ] && echo "--inner_clip_norm $INNER_CLIP_NORM" ) \
        $( [ "$USE_GRAD_CKPT" = true ] && echo "--use_gradient_checkpointing" ) \
        --max_steps 75000 \
        --eval_steps 500 \
        --logging_steps 500 \
        --warmup_steps 10000 \
        --early_stopping_patience 500 \
        --seed $((142+$N))
done

echo "Done"
