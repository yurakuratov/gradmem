# SQuAD support-question distillation

Standalone entry points (neither invokes another experiment script):

```bash
CUDA_VISIBLE_DEVICES=0 NP=1 conda run -n rmt \
  bash scripts/run_support_distillation_on_squad.sh \
  --n_support_questions 2 --K 3 --inner_lr 0.01
```

Check GPU availability before launching. Batch size counts **contexts**;
each inner forward has `batch_size * n_support_questions` rows. The default
launcher uses one GPU, batch size one, eager attention and bf16. Override
launcher settings with environment variables, or pass **any Python runner
argument** after the shell command. All existing self-distillation runner
arguments are declared locally in the new runner, not imported from it:

```bash
conda run -n rmt python run_support_distillation_on_squad.py --help
```

The launcher does not source another shell script or the repository's Accelerate
configuration. For a small randomly initialized model, set `PRETRAINED_MODEL=''`,
`BASE_MODEL=llama`, `TOKENIZER_PATH=<tokenizer>`, and optionally `L`, `H`, `D`.
Output defaults to a unique directory under `runs/squad/`; CLI arguments are saved
in `config.json`, and full model configuration is saved with checkpoints.

## Protocol

- Group raw SQuAD rows by exact context; deduplicate question text. Skip contexts
  with fewer than `n_support_questions + 1` answered, distinct questions.
- Construct one episode per possible held-out question, selecting support
  questions without replacement. Episode selection is fixed and reproducible
  for the seed; train and validation are grouped separately. Validation uses
  `seed + 1`. Supports do not resample every epoch.
- Match the existing distillation implementation: the teacher is the current
  backbone with stop-gradient, no memory, and WRITE LoRA disabled. It sees context
  plus support question and the teacher-forced first reference answer.
- The student sees shared per-context prefix memory plus each support question
  and its teacher-forced answer. All support questions run in parallel.
  `KL(teacher || student)` uses temperature one. Normalize per question over
  selected tokens, then average over support questions. Sum over independent
  context memories when differentiating, so inner LR is batch-size independent.
- `inner_distill_target_weight=1` and `inner_distill_query_weight=0` by default.
  K differentiable SGD steps update memory only; model parameters are updated
  by the outer optimizer, not by the inner loop.
- Held-out READ sees only memory and the held-out question/teacher-forced answer.
  Its answer CE backpropagates through all K updates. Held-out inputs and labels
  never influence WRITE. Optional **outer** distillation weights retain their
  existing meanings and default to zero.
- Metrics match the existing runner: teacher-forced token accuracy and token
  exact match, **not** autoregressive SQuAD EM/F1. Evaluation still performs WRITE.

This WRITE is support-question-conditioned, not standard query-independent
GradMem. It uses reference support answers as teacher-forcing inputs; it does
not generate pseudo-answers. `max_context_length` truncates teacher contexts
only and is unset by default.

## Arguments and restrictions

The standalone runner retains the full existing CLI, including model/tokenizer,
checkpoint initialization, optimization, backbone freezing, WRITE LoRA, memory
projection, clipping, attention, outer distillation, training, and evaluation
arguments. New arguments:

- `--n_support_questions` (2)
- `--inner_distill_query_weight` (0)
- `--inner_distill_target_weight` (1)
- `--inner_reconstruction_weight` (0): add original context reconstruction CE
  to every inner step, e.g. `--inner_reconstruction_weight 1.0` or launcher
  environment `INNER_RECONSTRUCTION_WEIGHT=1.0`.
- `--dataloader_num_workers` (0)
- `--resume_from_checkpoint` (unset)

Requires prefix memory, `grad_mode=second`, SGD (`use_adam=false`), and all K
steps second order. Leave `write_objective=reconstruction`: this runner replaces
that inner objective with support KL. Energy/shaping losses, K sampling,
separate writers, WRITE heads, memory noise and other auxiliary trajectory
losses are rejected explicitly rather than silently ignored. Eager attention
is the safe default for higher-order gradients; alternative attention backends
must support backward-over-backward. The existing Adam inner optimizer detaches
its updates, so it is deliberately disallowed.

To resume, invoke the Python entry point directly with the original arguments,
`--exp_path <existing-run>` and `--resume_from_checkpoint <checkpoint-directory>`.
The shell launcher deliberately refuses existing output paths.

Tests: `conda run -n rmt python -m pytest -q tests/test_support_distillation.py`.

## Optional inner reconstruction CE

The inner objective is support KL plus `inner_reconstruction_weight * context CE`.
CE is averaged over non-padding context tokens independently per context, then
added to that context's support-averaged KL. It does not directly enter the outer
loss. Its default weight of zero preserves KL-only WRITE.

When enabled, reconstruction rows and **all** support-question rows (including
the first) share one backbone forward call per inner step. They share the same
per-context memory through differentiable copies, but remain separate batch
rows: support KL never sees the context tokens. This avoids a separate CE
forward invocation, not the necessary context-token computation; padding to the
longest row may increase memory/compute. Concatenating context before support
QA would be cheaper in some cases but would leak context into the student.
The teacher is still computed once before the inner loop. Logs distinguish
`inner_support_kl_loss`, `inner_reconstruction_loss`, and their weighted total
`inner_loss`.
