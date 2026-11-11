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
`WRITE_OBJECTIVE` now defaults to `energy`, matching the source launcher.
Support KL augments the selected WRITE signal; use `WRITE_OBJECTIVE=reconstruction`
for KL with optional context CE and no learned energy. All inherited model
compatibility checks still apply. Reconstruction, energy, and mixed objectives
are available for **all three memory formats**, including LoRA and KV cache.

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
  Inner steps update memory only; model parameters are updated by the outer
  optimizer, not by the inner loop. SGD with `grad_mode=second` preserves the
  differentiable WRITE trajectory.
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

Support distillation now uses the original GradMem meta-loop rather than a
replacement loop. It supports all three memory formats: `prefix`, `lora`, and
`kv_cache`. Support questions share one per-context state, expanded recursively
over prefix tensors, LoRA factors, or KV tensors. Backends retain their native
activation contexts, cache construction, and causal logit alignment.

The inherited mechanisms remain operational: K/LR sampling, gradient modes and
partial second-order depth, clipping, memory projection, WRITE heads, separate
writers, backbone freezing / WRITE LoRA, memory noise, inner-loss-to-outer,
intermediate READ, alignment/orthogonal/Ivan losses, energy ranking/trajectory/
anchor/Lipschitz shaping, and memory search. Original validation constraints are
preserved, not bypassed: energy-memory search requires prefix memory, SGD, and
all WRITE steps second order. Energy itself, layer-wise energy,
ranking/trajectory/anchor shaping, and the Lipschitz penalty work with prefix,
LoRA, and KV-cache memory. Non-prefix ranking negatives and gradient norms use
the concatenated fast-factor/cache parameter space. See
`AGENTS.md` for the target-guided training/inference mismatch when
`energy_memory_search_use_best_for_next_step` is enabled.

`grad_mode=second` with SGD is recommended for the requested meta-gradients.
First-order and detached modes remain selectable with their original meanings.
The inherited Adam inner optimizer remains selectable, but it detaches its
updates and does **not** provide full second-order meta-gradients. Eager attention
is a safe choice for CPU tests; other attention implementations must support
backward-over-backward for second-order training.

To resume through the launcher, set `RESUME_FROM_CHECKPOINT` to a checkpoint
directory; the run name includes a resume suffix. Alternatively invoke the Python
entry point with the original arguments, `--exp_path <run>` and
`--resume_from_checkpoint <checkpoint-directory>`. The shell uses the inherited
run-lock/existing-run policy from `collect_env_state.sh`.

Tests: `conda run -n rmt python -m pytest -q tests/test_support_distillation.py`.
Launcher coverage checks (no training dependencies or GPU run):
`conda run -n rmt python tests/test_support_distillation_launcher.py`.
Energy/backend matrix (native and support-distillation, GPT-2/NeoX/Llama):
`conda run -n rmt python -m unittest discover -s tests -p test_energy_memory_backends.py -v`.

## Optional inner reconstruction CE

For `write_objective=reconstruction`, the inner objective is support KL plus
`inner_reconstruction_weight * context CE`. This preserves the existing
support-runner default of KL-only WRITE; `write_reconstruction_weight` retains
its original role in the **energy_with_reconstruction** objective.
For `write_objective=energy`, add the original learned energy (coefficient one).
For `write_objective=energy_with_reconstruction`, add the original weighted sum
`write_energy_weight * energy + write_reconstruction_weight * context CE`.
Thus `inner_reconstruction_weight` always adds optional extra CE, and should
normally remain zero when the weighted mixed objective already supplies CE.
Both support KL weights may be zero for native CE/energy baselines, provided
another inner signal remains enabled.
CE is averaged over non-padding context tokens independently per context, then
added to that context's support-averaged KL. It does not directly enter the outer
loss. Its default weight of zero preserves KL-only WRITE.

For shared-writer prefix memory, context CE/energy and **all** support-question
rows (including the first) share one backbone forward call per inner step.
Layer-wise energy and custom WRITE heads are supported in this fused path.
They share the same
per-context memory through differentiable copies, but remain separate batch
rows: support KL never sees the context tokens. This avoids a separate CE
forward invocation, not the necessary context-token computation; padding to the
longest row may increase memory/compute. Concatenating context before support
QA would be cheaper in some cases but would leak context into the student.
Separate writers and non-prefix memory use their native context WRITE forward
plus one parallel support-question forward per step, rather than combining
different hidden spaces or cache/adapter activation into an invalid batch.
The teacher is still computed once before the inner loop. As in native GradMem,
there is also a final WRITE-objective evaluation for diagnostics and shaping.
Logs distinguish
`inner_support_kl_loss`, `inner_reconstruction_loss`, and their weighted total
`inner_loss`.
