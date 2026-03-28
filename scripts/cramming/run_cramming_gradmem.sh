#!/bin/bash
set -e
cd ../..

# ── model ──
MODEL=${MODEL:-gpt2}
DTYPE=${DTYPE:-float32}
DEVICE=${DEVICE:-cuda}

# ── memory optimisation ──
N_MEM=${N_MEM:-24}
N_STEPS=${N_STEPS:-5000}
LR=${LR:-0.01}
OPTIMIZER=${OPTIMIZER:-adam}
EARLY_STOP_ACC=${EARLY_STOP_ACC:-0.99}
EARLY_STOP_CHECK=${EARLY_STOP_CHECK:-100}

# ── data ──
DATASET=${DATASET:-pg19}
SPLIT=${SPLIT:-test}
N_SAMPLES=${N_SAMPLES:-10}

# ── sweep ──
LENGTHS=${LENGTHS:-"1000"}
THRESHOLD=${THRESHOLD:-0.99}

# ── misc ──
SEED=${SEED:-42}
OUTPUT_DIR=${OUTPUT_DIR:-cramming_results}

echo "RUNNING: model=$MODEL  n_mem=$N_MEM  optimizer=$OPTIMIZER  lr=$LR  steps=$N_STEPS"

python run_cramming.py \
    --model $MODEL \
    --dtype $DTYPE \
    --device $DEVICE \
    --n_mem_tokens $N_MEM \
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
