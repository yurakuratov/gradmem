#!/bin/bash
set -e
cd ../..
export GRAD_VERBOSE=1

# ── model ──
MODEL=${MODEL:-gpt2}
DTYPE=${DTYPE:-float32}
DEVICE=${DEVICE:-cuda:1}

# ── low-rank injection ──
LORA_MODE=${LORA_MODE:-ffn}   # "residual" or "ffn"
LAYER_IDX=${LAYER_IDX:-6}
RANK=${RANK:-1}
N_STEPS=${N_STEPS:-1000}
LR=${LR:-0.01}
OPTIMIZER=${OPTIMIZER:-adam}
EARLY_STOP_ACC=${EARLY_STOP_ACC:-0.99}
EARLY_STOP_CHECK=${EARLY_STOP_CHECK:-100}

# ── data ──
DATASET=${DATASET:-pg19}
SPLIT=${SPLIT:-test}
N_SAMPLES=${N_SAMPLES:-10}

# ── sweep ──
LENGTHS=${LENGTHS:-"1024"}
THRESHOLD=${THRESHOLD:-0.99}

# ── misc ──
SEED=${SEED:-42}
OUTPUT_DIR=${OUTPUT_DIR:-cramming_results}

echo "RUNNING: model=$MODEL  mode=$LORA_MODE  layer=$LAYER_IDX  rank=$RANK  optimizer=$OPTIMIZER  lr=$LR  steps=$N_STEPS"

python run_cramming.py \
    --model_class gradlora \
    --lora_mode $LORA_MODE \
    --model $MODEL \
    --dtype $DTYPE \
    --device $DEVICE \
    --layer_idx $LAYER_IDX \
    --rank $RANK \
    --n_steps $N_STEPS \
    --lr $LR \
    --optimizer $OPTIMIZER \
    --early_stop_acc $EARLY_STOP_ACC \
    --early_stop_check_every $EARLY_STOP_CHECK \
    --dataset $DATASET \
    --split $SPLIT \
    --n_samples $N_SAMPLES \
    --lengths $LENGTHS \
    --threshold $THRESHOLD \
    --seed $SEED \
    --output_dir $OUTPUT_DIR
