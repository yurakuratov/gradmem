#!/bin/bash
set -e
cd ../..

# ── shared defaults (override from env) ──
MODEL=${MODEL:-gpt2}
DTYPE=${DTYPE:-float32}
DEVICE=${DEVICE:-cuda:1}

LORA_MODE=${LORA_MODE:-residual}   # "residual" or "ffn"
RANK=${RANK:-2}
N_STEPS=${N_STEPS:-1000}
LR=${LR:-0.01}
OPTIMIZER=${OPTIMIZER:-adam}
EARLY_STOP_ACC=${EARLY_STOP_ACC:-0.99}
EARLY_STOP_CHECK=${EARLY_STOP_CHECK:-100}

DATASET=${DATASET:-pg19}
SPLIT=${SPLIT:-test}
N_SAMPLES=${N_SAMPLES:-10}

LENGTHS=${LENGTHS:-"1024"}
THRESHOLD=${THRESHOLD:-0.99}
SEED=${SEED:-42}

OUTPUT_DIR=${OUTPUT_DIR:-cramming_results}

# ── layer range (GPT-2 has 12 layers: 0..11) ──
LAYER_MIN=${LAYER_MIN:-0}
LAYER_MAX=${LAYER_MAX:-11}

echo "═══ Layer sweep: mode=$LORA_MODE  layers $LAYER_MIN..$LAYER_MAX  rank=$RANK  lr=$LR  steps=$N_STEPS ═══"

if [ "$LORA_MODE" = "ffn" ]; then
    MODE_TAG="ffn"
else
    MODE_TAG="lora"
fi

for L in $(seq $LAYER_MIN $LAYER_MAX); do
    echo ""
    echo "──────────────── layer $L ────────────────"
    python run_cramming.py \
        --model_class gradlora \
        --lora_mode $LORA_MODE \
        --model $MODEL \
        --dtype $DTYPE \
        --device $DEVICE \
        --layer_idx $L \
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
done

echo ""
echo "═══ All layers done. Plotting… ═══"

python scripts/cramming/plot_sweep.py \
    --results_dir $OUTPUT_DIR \
    --pattern "${MODE_TAG}_l*_r${RANK}_${OPTIMIZER}_lr${LR}_steps${N_STEPS}" \
    --x_field config.layer_idx \
    --x_label "Layer index" \
    --title "GradLoRA ($LORA_MODE) layer sweep (rank=$RANK, lr=$LR, steps=$N_STEPS)" \
    --output $OUTPUT_DIR/sweep_layers_${MODE_TAG}_r${RANK}_${OPTIMIZER}_lr${LR}_steps${N_STEPS}.png
