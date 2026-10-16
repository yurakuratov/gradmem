# Curriculum Learning for GradMemGPT

> Implementation: [`run_gradmemgpt_on_kv_retrieval.py`](./run_gradmemgpt_on_kv_retrieval.py) — `CurriculumCallback`, `CurriculumTrainer`, and the stage loop in `main()`.
> Example config (adaptive model): [`configs/gradmemgpt/kv_retrieval/adaptive_curriculum.yaml`](./configs/gradmemgpt/kv_retrieval/adaptive_curriculum.yaml)
> Example config (base + Hopfield): [`configs/gradmemgpt/kv_retrieval/hopfield_curriculum.yaml`](./configs/gradmemgpt/kv_retrieval/hopfield_curriculum.yaml)
> Tests: [`test_curriculum_mechanics.py`](./test_curriculum_mechanics.py)

## What it does

Curriculum learning trains stage-by-stage on KV-retrieval datasets of increasing difficulty (N = number of KV pairs in the context). A stage ends and the next begins as soon as the model crosses an `exact_match` threshold on the current stage. The model's weights (and the meta-learned `self.mem`) **carry across stages** — each stage warm-starts from the previous one's final weights, so the model is gradually stretched onto harder contexts.

```
 stage 0 (N=4)           stage 1 (N=8)          ...   stage k (N=128)
 ┌────────────┐  eval_exact_match >= threshold  ┌────────────┐
 │ train, eval│ ─────────────────────────────▶ │ train, eval│
 │  model_m   │            carry weights         │ model_{m+1}│
 └────────────┘                                 └────────────┘
```

Each stage trains until **either** `eval_exact_match >= curriculum_threshold` (advance) **or** early-stopping / `max_steps` fires. If a stage ends *without* crossing the threshold, the curriculum **stops** (the advancement gate) — harder stages are not attempted because the model is not ready.

## When to use it

- **Long-context KV-retrieval** where training directly at N=128 fails to learn. Staging from N=4 → N=128 lets the WRITE pathway lock in short-context retrieval first, then generalize.
- **The adaptive fork** (`adaptive:` section). Cross-segment memory carry means later stages unroll the gated recurrence over *more* segments — curriculum is the natural way to grow the recurrence length. This is the regime `adaptive_curriculum.yaml` targets.

## How to use it

### 1. Prepare the per-stage datasets

Curriculum loads each stage from `<curriculum_data_dir>/<curriculum_dataset_template formatted with n_kv>`. With the defaults:

| stage | `n_kv` | dataset path |
|---|---|---|
| 0 | 4  | `./data/N4-K2V2-V62_1M` |
| 1 | 8  | `./data/N8-K2V2-V62_1M` |
| 2 | 16 | `./data/N16-K2V2-V62_1M` |
| … | … | `./data/N{n_kv}-K2V2-V62_1M` |

Generate them via `notebooks/dump_dataset.ipynb` or download with `./scripts/download_kv_retrieval.sh`. `n_kv` may also be a non-integer string (then the template value is used verbatim as the dataset name and the label).

### 2. Add a `curriculum:` section to your config

```yaml
curriculum:
  curriculum_enabled: true
  curriculum_threshold: 0.9          # advance when eval_exact_match >= threshold
  curriculum_levels: "4,8,16,32,64,96,128"
  curriculum_data_dir: "./data"
  curriculum_dataset_template: "N{n_kv}-K2V2-V62_1M"
  # Optional per-stage hyperparameter overrides (keys are stage indices):
  # stage_overrides:
  #   3: {inner_lr: 0.02}
  #   5: {inner_lr: 0.01, eval_steps: 1000}
```

| knob | type | default | meaning |
|---|---|---|---|
| `curriculum_enabled` | bool | `false` | turns the stage loop on |
| `curriculum_threshold` | float | `0.95` | `eval_exact_match` value that advances a stage (`>=`) |
| `curriculum_levels` | str | `"4,8,16,32,64,128"` | comma-separated `n_kv` per stage |
| `curriculum_data_dir` | str | `"./data"` | dir the stage datasets live under |
| `curriculum_dataset_template` | str | `"N{n_kv}-K2V2-V62_1M"` | `{n_kv}` is formatted with the stage's level |
| `stage_overrides` | map | — | per-stage hyperparameter overrides (see below) |

> **Key-name note.** `run_from_config.py --dry-run` does **not** echo the curriculum CLI flags (it checks `curriculum.enabled` while configs use `curriculum_enabled`). This is cosmetic — the subprocess receives `--config <path>`, and `run_gradmemgpt_on_kv_retrieval.py:main()` re-reads the YAML and sets `args.curriculum_enabled` directly. Curriculum is active whenever `curriculum_enabled: true` is in the config. The same applies to the existing `hopfield_curriculum.yaml`.

### 3. Run

```bash
# Dry-run (shows the resolved command; curriculum flags won't appear — see note above)
python run_from_config.py --config configs/gradmemgpt/kv_retrieval/adaptive_curriculum.yaml --dry-run

# Train (CPU-safe to drop the accelerate wrapper; GPU uses it)
python run_from_config.py --config configs/gradmemgpt/kv_retrieval/adaptive_curriculum.yaml

# Override a single value without editing the file
python run_from_config.py --config configs/gradmemgpt/kv_retrieval/adaptive_curriculum.yaml curriculum_threshold=0.85
```

## Per-stage overrides

`stage_overrides` lets you change hyperparameters at a specific stage index. Each value is set before that stage's trainer is built. Recognized keys:

- **Model params** (`MODEL_PARAM_MAP`) are set on **both** `model.<attr>` and `model.config.<attr>`: `inner_lr`, `inner_clip_value`, `inner_clip_norm`, `hopfield_*`, `gated_delta_*`.
- **Training params** (`TRAINING_PARAM_MAP`) are set on `args` directly: `learning_rate`, `warmup_steps`, `weight_decay`, `per_device_batch_size`, `max_steps`, `eval_steps`, `logging_steps`, `early_stopping_patience`, `total_batch_size`.
- **Unknown keys** are logged and skipped (not fatal).

A common pattern is annealing the inner learning rate as N grows, since longer contexts produce larger accumulated inner gradients:

```yaml
  stage_overrides:
    3: {inner_lr: 0.02}
    5: {inner_lr: 0.01}
```

## How the mechanics work

Three pieces collaborate (see [`test_curriculum_mechanics.py`](./test_curriculum_mechanics.py) for executable specs):

1. **`CurriculumCallback`** — an `on_evaluate` callback. At every eval it looks up `eval_exact_match`; if `>= curriculum_threshold`, it sets `threshold_reached = True` and flips `control.should_training_stop` (ending the stage's `trainer.train()`). A missing metric key is a silent no-op.

2. **`CurriculumTrainer(PerSegmentForgettingTrainer)`** — stamps every log with `curriculum_stage` / `curriculum_stage_label`, and adds `step_offset` (the cumulative steps from prior stages) to the logged `global_step` so plots are continuous across stages. The real `state.global_step` is left at the within-stage value (the offset is applied only for logging, then restored). It also inherits the per-segment forgetting eval path (see below), so under curriculum the forgetting matrix fires both on cadence and **forced at every stage boundary**. When `per_segment_eval` is off, the forgetting path is a no-op and the trainer behaves like the plain `CustomTrainer`.

3. **The stage loop** (`main()`, behind `if args.curriculum_enabled:`) — for each level: apply `stage_overrides`, load that stage's dataset, build a `CurriculumTrainer` with a fresh `CurriculumCallback`, train, then carry the model (`model = trainer.model`) and accumulate `cumulative_steps += trainer.state.global_step`. The loop **breaks early** if a stage did not reach the threshold (the advancement gate). Final eval + `curriculum_metrics.json` are written for the last completed stage.

## Segmentation × curriculum (adaptive fork)

If you use the `adaptive:` model with a fixed `segment_size`, the **number of segments grows with N**: a stage-N4 context splits into ~1 segment, stage-N128 into many. Each stage therefore unrolls the gated recurrence over more steps, and information from the first segment must survive all the later ones — exactly the forgetting regime the gated update (`memory_update_rule: mamba`) is meant to fight. The per-segment forgetting matrix (`per_segment_eval: true`) makes this visible: as stages progress, older-segment rows degrade less at later probe columns.

### Forgetting matrix fires at every stage boundary

Under curriculum, the forgetting matrix runs on **two** triggers:

1. **Periodic (in-stage)** — every `per_segment_eval_steps`, like the single-stage path, giving the in-stage forgetting trajectory.
2. **Forced (stage-boundary)** — exactly once at the end of each stage (every stage, whether it advanced or halted), via `trainer.force_forgetting_pass(stage_idx, stage_label)` called right before the end-of-stage eval. This bypasses the step-cadence gate so you always get a snapshot of the model's memory at each stage transition, even if the stage ended off-cadence (early-stopping, threshold reached).

Forced passes are written as **stage-labeled artifacts** so they're easy to tell apart from periodic ones:
- `<exp_path>/stage_<idx>_<label>/forgetting_matrix_step{step}_stage{idx}_{label}.json` / `.csv` (periodic ones keep the plain `forgetting_matrix_step{step}` name).
- The JSON gains `forced: true`, plus `stage_idx` and `stage_label` fields.

To instead keep a **fixed** number of segments per stage, set `n_segments` and drop `segment_size`.

## Outputs

Each stage writes to `<exp_path>/stage_<idx>_<label>/` (checkpoints, eval logs). At the end:

- `curriculum_metrics.json` — per-stage final eval metrics (`{"stage_0_N4": {...}, ...}`).
- `trainer_state.json` — final trainer state.
- `all` metrics — saved via `trainer.save_metrics(split='all', ...)`.
- **(adaptive + `per_segment_eval`)** per-stage forgetting matrices under `stage_<idx>_<label>/`: periodic `forgetting_matrix_step{N}.*` plus a forced stage-boundary `forgetting_matrix_step{N}_stage{idx}_{label}.*` (see above).

The carried model and the continuous step count mean a curriculum run looks like one long training trajectory in comet_ml (modulo the per-stage `step_offset`), with `curriculum_stage` / `curriculum_stage_label` available to color the curve.

## Combining with the energy / recon-pretrain schedules

> ⚠️ **Beware.** The energy-recon anneal and the reconstruction pretrain warmup are driven by `current_train_step`, which is stamped from `trainer.state.global_step`. In `CurriculumTrainer` this resets per stage, so **these schedules restart at the beginning of every curriculum stage.** That is usually desirable (each stage re-anneals), but if you intended a single global schedule across the whole curriculum, you must account for the per-stage reset.
