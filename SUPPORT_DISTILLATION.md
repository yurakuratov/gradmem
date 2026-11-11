# SQuAD support-question distillation

Standalone entry points (neither invokes another experiment script):

```bash
N_SUPPORT_QUESTIONS=2 INNER_RECONSTRUCTION_WEIGHT=1.0 conda run -n rmt \
  bash scripts/run_support_distillation_on_squad.sh
```

Check GPU availability before launching. The launcher is based directly on
`scripts/run_energygradmem_on_squad.sh`: it retains every Bash variable, CLI
argument, conditional branch, run-name construction, run loop, locking, logging,
timing, and finalization. Configure the Bash variables in the script, or use
environment overrides where the assignment supports them. It does not use a
generic CLI passthrough as a substitute for explicit Bash parameters.

Inherited settings include `NP=2`, `TBS=64`, `PER_DEVICE_BATCH_SIZE=16`, `K=1`,
`INNER_LR=0.4`, WRITE LoRA, frozen backbone, `flashbob`, and bf16. These settings
are retained, **not** a newly validated hardware configuration. For a single-GPU
run, edit `NP=1`, check batch size/accumulation, select an available GPU using
`CUDA_VISIBLE_DEVICES`, and verify Accelerate configuration before launching.
The necessary algorithm-specific default change is
`WRITE_OBJECTIVE=reconstruction`: support KL replaces the original WRITE loss.
Energy-related variables and forwarding are retained; unsupported enabled
objectives still fail explicitly in model configuration validation.

Batch size counts **contexts**; each inner forward has
`batch_size * n_support_questions` support rows, plus one reconstruction row per
context if CE is enabled. The standalone Python runner also exposes its full CLI:

```bash
conda run -n rmt python run_support_distillation_on_squad.py --help
```

Like the source launcher, it sources `collect_env_state.sh` for shared run
utilities and passes `accelerate.yaml` to Accelerate. It does not invoke or source
another experiment launcher. For a small randomly initialized model, edit
`PRETRAINED_MODEL=''`, `BASE_MODEL=llama`, `TOKENIZER_PATH=<tokenizer>`, and
optionally `L`, `H`, `D`. Ensure the tokenizer matches the selected backbone.
Outputs use `runs/squad/<descriptive-run-name>/run_<N>` and the inherited run-lock
lifecycle. CLI arguments are saved in `config.json`, and full model configuration
is saved with checkpoints.

New Bash controls include `N_SUPPORT_QUESTIONS`, `INNER_DISTILL_QUERY_WEIGHT`,
`INNER_DISTILL_TARGET_WEIGHT`, `INNER_RECONSTRUCTION_WEIGHT`, `READ_CE_WEIGHT`,
all four `DISTILL_*_WEIGHT` controls, and explicit training/evaluation settings.
The support count and all inner/outer distillation weights appear in the run
name. Non-default added training controls get suffixes as well.

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
not generate pseudo-answers. `max_context_length` truncates contexts used by the
teacher and optional reconstruction CE. The launcher inherits a 256-token cap;
the Python entry point defaults to no cap.

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

To resume through the launcher, set `RESUME_FROM_CHECKPOINT` to a checkpoint
directory; the run name includes a resume suffix. Alternatively invoke the Python
entry point with the original arguments, `--exp_path <run>` and
`--resume_from_checkpoint <checkpoint-directory>`. The shell uses the inherited
run-lock/existing-run policy from `collect_env_state.sh`.

Tests: `conda run -n rmt python -m pytest -q tests/test_support_distillation.py`.
Launcher coverage checks (no training dependencies or GPU run):
`conda run -n rmt python tests/test_support_distillation_launcher.py`.

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
