#!/bin/bash
# Reuse the energy launcher, including locking, provenance, and all auxiliaries.
# Any shared-runner CLI flag can be supplied here; trailing flags override the
# preset. For example: --write_objective reconstruction --memory_backend lora
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export TRAIN_SCRIPT=run_self_distillation_on_kv_retrieval.py
READ_CE_WEIGHT=${READ_CE_WEIGHT:-1.0}
DISTILL_LOGITS_QUERY_WEIGHT=${DISTILL_LOGITS_QUERY_WEIGHT:-0.0}
DISTILL_LOGITS_TARGET_WEIGHT=${DISTILL_LOGITS_TARGET_WEIGHT:-1.0}
DISTILL_HIDDEN_QUERY_WEIGHT=${DISTILL_HIDDEN_QUERY_WEIGHT:-0.0}
DISTILL_HIDDEN_TARGET_WEIGHT=${DISTILL_HIDDEN_TARGET_WEIGHT:-0.0}
ARGS=(
  --read_ce_weight "$READ_CE_WEIGHT"
  --distill_logits_query_weight "$DISTILL_LOGITS_QUERY_WEIGHT"
  --distill_logits_target_weight "$DISTILL_LOGITS_TARGET_WEIGHT"
  --distill_hidden_query_weight "$DISTILL_HIDDEN_QUERY_WEIGHT"
  --distill_hidden_target_weight "$DISTILL_HIDDEN_TARGET_WEIGHT"
  "$@"
)
# Include both readable weights and an exact CLI digest: overridden inherited
# settings must not silently collide with a different experiment's directory.
DIGEST=$(printf '%s\0' "${ARGS[@]}" | sha256sum | cut -c1-12)
export RUN_NAME_SUFFIX="${RUN_NAME_SUFFIX:+${RUN_NAME_SUFFIX}_}sd_ce${READ_CE_WEIGHT}_klq${DISTILL_LOGITS_QUERY_WEIGHT}t${DISTILL_LOGITS_TARGET_WEIGHT}_hq${DISTILL_HIDDEN_QUERY_WEIGHT}t${DISTILL_HIDDEN_TARGET_WEIGHT}_${DIGEST}"
exec bash "$SCRIPT_DIR/run_energygradmem_on_kv_retrieval.sh" "${ARGS[@]}"
