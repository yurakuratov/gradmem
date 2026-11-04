"""KV runner retaining every GradMem option, with optional self-distillation."""

from dataclasses import dataclass

from run_gradmemgpt_on_kv_retrieval import ExperimentArgs, main
from self_distillation import (
    DISTILLATION_DEFAULTS, SelfDistillationConfig, SelfDistillationGradMemGPT,
)


@dataclass
class SelfDistillationArgs(ExperimentArgs):
    read_ce_weight: float = 1.0
    distill_logits_query_weight: float = 0.0
    distill_logits_target_weight: float = 0.0
    distill_hidden_query_weight: float = 0.0
    distill_hidden_target_weight: float = 0.0


if __name__ == '__main__':
    # TRAIN_COMPONENT_KEYS is built at import time in the shared runner.
    import run_gradmemgpt_on_kv_retrieval as runner

    keys = ['weighted_target_loss', 'distillation_loss'] + [
        name.removesuffix('_weight') + '_loss'
        for name in DISTILLATION_DEFAULTS if name != 'read_ce_weight'
    ]
    runner.LOSS_COMPONENT_KEYS.extend(keys)
    runner.TRAIN_COMPONENT_KEYS.extend(keys)
    main(SelfDistillationArgs, SelfDistillationConfig, SelfDistillationGradMemGPT,
         tuple(DISTILLATION_DEFAULTS))
