# AGENTS.md - Test Time Gradient Descent

## Environment Setup
```bash
uv sync                      # creates .venv from pyproject.toml + uv.lock
uv run python run_from_config.py --config configs/gradmemgpt/kv_retrieval/default.yaml
```
Dependencies, pins, and the PyTorch `cu121` wheel index live in `pyproject.toml`; `uv.lock` is the fully resolved graph (commit it). Use `uv sync --extra notebooks` for notebook/data-prep deps only.

## Running Training (Two Options)

### Option 1: YAML Config (Recommended)
```bash
# Dry run to see generated command
python run_from_config.py --config configs/gradmemgpt/kv_retrieval/default.yaml --dry-run

# Generate config and show command
python run_from_config.py --config configs/gradmemgpt/kv_retrieval/default.yaml

# With parameter overrides
python run_from_config.py --config configs/gradmemgpt/kv_retrieval/default.yaml --model gradmemgpt inner_lr=0.02 seed=143
```

### Option 2: Shell Scripts (Legacy)
```bash
./scripts/run_gradmemgpt_on_kv_retrieval.sh
./scripts/run_gpt_on_kv_retrieval.sh
```

## Config Structure
```yaml
# configs/gradmemgpt/kv_retrieval/default.yaml
model:
  base_model: gpt2
  n_layer: 4
  n_head: 4
  n_embd: 128

training:
  per_device_batch_size: 64
  total_batch_size: 64
  learning_rate: 1.0e-4
  max_steps: 1000000
  seed: 142

dataset:
  data_name: "N8-K2V2-V62_1M"
  tokenizer_path: "./tokenizers/kv_alphabet_62/"

gradmem:
  n_mem_tokens: 8
  K: 2
  inner_lr: 0.04
  grad_mode: "second"
  use_write_head: true

# Optional top-level output-folder controls:
# runs_dir: "./runs"     # output root prefix (default: ./runs)
# run_name: "my_run"     # override the auto-generated hyperparameter folder component (default: auto)
# run_naming:            # build the auto name from listed keys instead of per-model logic
#   keys: [...]
```

### Output-folder naming & uniqueness
The output dir layout is `<runs_dir>/<data_name>/<run_name>/run_<seed>_<uid8>`:
- `runs_dir` (top-level, default `./runs`) is the root prefix.
- `run_name` (top-level, default auto-generated from hyperparameters) is the middle component. When set, it **fully replaces** the auto name; `run_name_suffix` only appends to the auto name.
- `run_<seed>` is the training seed (`training.seed`).
- `<uid8>` is a unique per-run postfix so identical-config runs no longer overwrite each other. When comet_ml is available, `get_run_uid()` (`generate_run_name.py`) generates the id via `comet_ml.generate_guid()` and exports it as `COMET_EXPERIMENT_KEY` — HF's `CometCallback` then binds the comet experiment to that exact key, so `<uid8>` (its first 8 chars) visually pairs the local folder with the comet URL. `run_from_config.py` generates the uid once and the launched subprocess inherits `COMET_EXPERIMENT_KEY` via the parent env.

### Run name generation & comet
- **Name → comet:** every run script resolves `run_name` (manual override else auto-generated) and passes it to `TrainingArguments(run_name=...)`. HF's `CometCallback` only sets the comet experiment name when `run_name != output_dir`; previously both defaulted to `output_dir` so comet used a random name. Now the comet experiment name matches the local folder's `run_name` component for *all* runs (manual or auto).
- **Auto-name source:** `generate_run_name(cfg)` (`generate_run_name.py`). If a top-level `run_naming:` section with a `keys:` list is present, the name is built generically from those keys (fully replacing the per-model-type logic). Each `keys` entry is a plain `"section.key"` string (prefix = key name) or a `{key, prefix}` map for a short prefix. Missing/None/False values are skipped; `True` renders as the flag name. Otherwise the per-model-type generators (`generate_run_name_gradmemgpt/rmt/gpt2`) run as before.

## Key Scripts
- `run_gpt2_on_kv_retrieval.py` - vanilla causal LM baseline
- `run_gradmemgpt_on_kv_retrieval.py` - model with writable memory (grad_memgpt.py)
- `run_gradmemgpt_on_squad.py` - SQuAD question answering
- `run_rmt_on_kv_retrieval.py` - recurrent memory transformer
- `run_from_config.py` - YAML config entry point (new)
- `run_scheduler.py` - batch many `run_from_config.py` runs from a YAML manifest (see below)
- `generate_run_name.py` - utility for run name generation

## Batch scheduling (`run_scheduler.py`)
Runs a grid of experiments from one manifest YAML with checkpointing, optional
parallelism, auto-retry, and resume-after-interrupt. Each run is launched as
`<command> <base_config> <positional key=value overrides>` — the overrides are
**positional** (no `--`), exactly as `run_from_config.py` accepts them.
```bash
python run_scheduler.py manifests/kv_retrieval_grid.yaml --dry-run   # show commands
python run_scheduler.py manifests/kv_retrieval_grid.yaml             # run for real
```
Manifest schema (`experiments[]` each need `name`, `command`, `base_config`;
optional `overrides`, `grid` (cartesian product), `max_parallel`; top-level
`max_parallel`, `schedule: sequential|interleaved`, `spawn_delay`, `max_retries`):
```yaml
max_parallel: 1
experiments:
  - name: gradmem_K_grid
    command: "python run_from_config.py --config"   # must end with --config
    base_config: "configs/gradmemgpt/kv_retrieval/default.yaml"
    overrides: {gradmem.use_write_head: true}
    grid: {gradmem.K: [1, 2, 4], gradmem.inner_lr: [0.02, 0.04]}
```
- `command` is the launch prefix and `base_config` is appended as a bare token
  right after it. `run_from_config.py` requires `--config <path>` (a bare
  positional is rejected), so `command` **must end with `--config`** — then
  `base_config` becomes its value.
- Flags like `--debug` go in `command` **before** the trailing `--config`:
  `command: "python run_from_config.py --debug --config"`. (Overrides are always
  `key=value`, never flags.)
- Per-run state lives in `<manifest>.checkpoint.json` next to the manifest
  (gitignored). Re-running the same command resumes: `ok` runs skip, `running`
  runs with a live PID are polled, others re-queue. Ctrl-C / SIGTERM prompts
  whether to terminate or leave children running (resumable via stored PID).
- **Failed-run backtraces:** each run's combined stdout/stderr is captured to
  `<manifest>.logs/<exp>__<idx>.log` (gitignored). On a non-zero exit the
  scheduler prints the last `--log-lines` (default 50) lines inline — the
  Python/accelerate traceback is the last thing a crashing run writes, so this
  surfaces it. Success logs are deleted; failure logs are kept (path also
  recorded in the checkpoint). `--no-log` disables capture; `--log-dir=` and
  `--log-lines=` (0 = capture but don't print) tune it. A resumed child left
  running keeps writing to its log; a resumed scheduler reads it back via the
  stored path if that child later fails.
- Useful flags: `--restart-failed`, `--restart-unknown`, `--max-retries=N`
  (auto-respawn on failure), `--on-interrupt=terminate|leave`, `--force`
  (merge when the manifest changed). See `python run_scheduler.py` for usage.

## Config Locations
- `configs/gpt2/kv_retrieval/` - GPT2 baselines
- `configs/gradmemgpt/kv_retrieval/` - GradMemGPT experiments
- `configs/gradmemgpt/squad/` - SQuAD experiments
- `configs/rmt/kv_retrieval/` - RMT experiments

## Datasets
- KV-retrieval: generate via `notebooks/dump_dataset.ipynb` or download via `./scripts/download_kv_retrieval.sh`
- bAbI: `./scripts/download_babi.sh`

## Important Implementation Details
- `attn_double_bwd/` directory contains custom attention kernels (imported as side-effect in grad_memgpt.py)
- grad_mode options: "none" (frozen), "first" (1st-order), "second" (full MAML - high VRAM)
- Config file `accelerate.yaml` uses BF16 precision, single process

## Local Environment (this machine)
Dependencies are managed by uv (see "Environment Setup" above). The `pyenv`-based `gradmem` env previously used here has been superseded by the project's uv-managed `.venv`:
```bash
uv sync                       # torch 2.5.1+cu121, Python >=3.10,<3.13, pinned in pyproject.toml
uv run python <script>.py     # or activate .venv: source .venv/bin/activate
```
- `uv` is at `~/.local/bin/uv`; `uv sync` creates `.venv/` in the repo root and installs the locked graph.
- Smoke-testing without a GPU: a tiny `GradMemGPT` (n_layer=2, n_head=2, n_embd=32, vocab 64) instantiates and does forward+backward in a few seconds — use it to validate edits to `grad_memgpt.py` before launching real runs.

## Tokenizer / model selection (IMPORTANT — easy to get wrong)
The babilong configs carry `tokenizer_path: "./tokenizers/kv_alphabet_62/"`, but **this field is inert for bAbI**. What actually loads is decided by which model key the YAML sets:

- `model.pretrained_model: gpt2` (all `configs/gradmemgpt/babi/*.yaml`) → `run_gradmemgpt_on_kv_retrieval.py:451-453` takes the `else` branch:
  - tokenizer = real **gpt2 (vocab 50257)**
  - model = **full pretrained gpt2** via `from_pretrained('gpt2')` (`n_embd=768`, 12 layers); the `n_layer/n_head/n_embd` YAML knobs are **ignored** on this branch.
- `model.base_model: gpt2` (kv_retrieval configs) → the `if` branch at line 419-449:
  - tokenizer = **kv_alphabet_62** (a 70-token character-level WordLevel tokenizer)
  - model = tiny from-scratch config (the `n_embd=128 / n_layer=4` knobs DO apply here), vocab resized to the tokenizer's.

Consequences:
- On bAbI the effective vocab is **50257**, so chance cross-entropy ≈ ln(50257) ≈ **10.8**. An inner loss of ~4 on noisy segments is genuine reconstruction of English prose, NOT chance.
- kv_alphabet_62 is **incompatible with bAbI** (no space/period in its vocab → everything maps to `[UNK]`). It is only valid for the synthetic KV-retrieval task.
- There is no `resize_token_embeddings` call in `grad_memgpt.py`; the model vocab is fixed by the pretrained weights.
- `generate_run_name.py` labels bAbI runs `L4H4D128` (from the ignored knobs) even though the real model is full gpt2 — cosmetic only.

## Datasets actually present locally
- `data/babilong_qa3_0k` and `data/babilong_qa3_4k` exist (facts-only and 4k-noise qa3 variants).
- babilong data on disk is **raw text strings** (HF `DatasetDict`, schema `{context, query, target}`), NOT token ids. Generation (`notebooks/create_tasks.py`) uses gpt2 only transiently for length budgeting, then decodes back to text. Re-tokenization happens at train time in `collate_fn`.
- bAbI noise = PG19 sentences inserted at random gaps between facts (`notebooks/babilong/babilong_utils.py:NoiseInjectionDataset`).

## bAbI / bAbIlong run path
- There is **no separate `run_gradmemgpt_on_babi.py`**. bAbI uses `run_gradmemgpt_on_kv_retrieval.py` (aliased in `run_from_config.py:42` as `gradmemgpt_babi`). Only this script has the energy/reconstruction features; `run_gradmemgpt_on_squad.py` only supports `add_inner_loss_to_outer`.
- Run from config: `python run_from_config.py --config configs/gradmemgpt/babi/<name>.yaml`.

## GradMemGPT model cheatsheet (`grad_memgpt.py`)
Two-loop design (docstring lines 346-383):
- **Inner loop (WRITE):** layout `[write_st][mem][write_end][context]`. For each context *segment*, run K gradient-descent steps on `mem_batch [B,M,d]` (initialized from meta-learned `self.mem [M,d]`) to reconstruct the segment. This is the "test-time gradient descent."
- **Outer loop (READ):** layout `[read_st][mem][read_end][query][target]`. Predict query answer from the final memory.

Key facts:
- **Plain memory is RESET per segment** (`grad_memgpt.py` ~line 1145: `mem_batch = self.mem...clone()`). Only the **last** segment's `mem_batch` reaches READ. Cross-segment continuity requires Hopfield (`use_hopfield_memory`) or Gated DeltaNet (`use_gated_delta_memory`) associative stores.
- **`retain_graph` only keeps the last segment's inner-loss graph** for the outer optimizer (line ~1264). With ~125 segments in second-order MAML, retaining all graphs is infeasible — any new outer-loss term that needs per-segment graphs must respect this.
- **Loss assembly** (`grad_memgpt.py` ~line 1877-1900): three mutually-exclusive branches gated by `add_inner_loss_to_outer`, `use_reconstruction_loss`, or the energy family. Order matters.
- **`grad_mode`:** `"none"` detaches (no meta-gradient to `self.mem`), `"first"` keeps graph but drops Hessian, `"second"` uses `create_graph=True` for full MAML. `last_K_second_order` (default = K) controls how many of the K steps are second-order.

## How losses/stats flow to logs
- Model `forward` returns `output = {'predictions': logits_q, 'inner_loop_stats': <dict>, 'loss': combined_loss}`.
- `inner_loop_stats` accumulates: `inner_loss`, `target_loss`, `rec_loss` (RMT only), `energy`/`energy_recon` (energy configs), `inner_grad_norm_*`, `mem_norm_*`, `delta_mem_norm_*`, `seg_nonempty_*`, `hopfield_*` or `gd_*` diagnostics, `gate_*` (gated rules), `mem_prior_*`, the train-only `star_*`/`replay_*` (see "Recent additions" #5), plus the per-segment diagnostic `inner_loss_seg_min/max` and pretrain schedule weights.
- **Two independent surfacing paths, both generic — adding a new metric needs NO run-script edit:**
  - **Eval side:** `preprocess_logits_for_metrics` passes `inner_loop_stats` through to `compute_metrics_fn`; after the explicit/hand-derived metrics (exact_match, token_accuracy, …) a generic catch-all forwards EVERY remaining tensor-valued key as `eval_<key>` (mean over the eval set; non-scalar/non-numeric values are skipped via try/except — give those explicit handling above the loop).
  - **Train side:** `CustomTrainer.compute_loss` captures every tensor-valued `inner_loop_stats` key of each training step, and `CustomTrainer.log` injects them into the HF training logs as `train_<key>` (guarded to training logs only: `'loss' in logs and 'eval_loss' not in logs`) → comet_ml. This is the ONLY way train-only stats (STAR/replay) are visible; it also gives train-time curves of `train_inner_loss`, `train_target_loss`, etc.
- **Recipe — to add a metric:** write `output['inner_loop_stats']['my_metric'] = <detached scalar tensor>` in the model forward. If it is computed in eval/both modes → it appears as `eval_my_metric`; if in training mode → as `train_my_metric`; if in both → both. Keep values scalar (0-dim or per-batch stackable); anything needing max/min variants or derived computation goes into the explicit block of `compute_metrics_fn` instead.
- **Idle-vs-off semantics:** stats that exist only when a feature fires should be emitted as explicit zeros when the feature is enabled but idle (e.g. `star_kl=0, star_n_probes=0` when the x\* mask is empty), so comet can distinguish "off" from "idle" — follow that convention for new regularisers.
- **The step schedule** (`energy_recon` anneal, recon-pretrain, `star/replay_anneal_steps`) is driven by `current_train_step`, stamped via `CustomTrainer.compute_loss` → `model.set_train_step(self.state.global_step)`. In `CurriculumTrainer`, `global_step` resets per stage, so schedules restart each curriculum stage — beware if combining schedules with curriculum.

## Config→CLI forwarding
`run_from_config.py:181-192` generically flattens every `gradmem:` key into a `--key` (underscore→hyphen) CLI arg. **Adding a new `gradmem:` YAML key requires only:** (1) a field on `GradMemGPTConfig` (`grad_memgpt.py`), (2) `getattr(config, ...)` in `GradMemGPT.__init__`, (3) a field on `ExperimentArgs` + passing it into the `GradMemGPTConfig(...)` call in `run_gradmemgpt_on_kv_retrieval.py`. No `run_from_config.py` edit needed. Booleans: `True` emits `--flag`, `False` emits `--flag=False`.
- **Caveat:** the `training:` section is *NOT* generically flattened — it's hand-coded field-by-field in `run_from_config.py` (~lines 138-155). New `training:` keys (e.g. `no_context_eval`) need an explicit forwarding stanza there.

## PG19 associative reconstruction task
A variant of associative retrieval on natural English text. Given a PG19 excerpt as `context` (written to memory via the inner-loop WRITE phase), reconstruct a held-out continuation `target` given its prefix `query`. The headline metric is **how much writing the context to memory helps**: `eval_delta_token_accuracy = eval_token_accuracy − eval_no_mem_token_accuracy`.

- **Dataset prep:** `prepare_pg19_reconstruction_dataset.py` consumes the chunk dataset from `prepare_pg19_chunks.py` (schema `{context, word_count, ...}`) and emits a `{context, query, target}` `DatasetDict` with splits `train`, `valid`, and `valid_no_context` (same rows as `valid` but `context=""`). For each example a contiguous substring (`query`+`target`) is held out from the end of `context`; `query+target` is always a contiguous substring of the original passage. Wrapper: `./scripts/prepare_pg19_reconstruction_dataset.sh`.
- **Run path:** reuses `run_gradmemgpt_on_kv_retrieval.py` (no separate script). Config: `configs/gradmemgpt/pg19_reconstruction/default.yaml` — uses `model.pretrained_model: gpt2` (real GPT-2, like bAbI).
- **Dual eval ("without memory"):** enabled by `training.no_context_eval: true`. `DualEvalTrainer.evaluate()` runs the normal eval, then a second pass against the `valid_no_context` split with `collate_fn_no_context` (which feeds an all-pad `context_input_ids` → the WRITE branch is skipped at `grad_memgpt.py:1209` because `context_input_ids.ne(pad_id).any()` is False → READ runs against the initial unwritten memory). `compute_metrics_fn_no_context` guards every `inner_loop_stats` key access (when WRITE is skipped the stats dict lacks `inner_loss` and the grad/mem norms). The deltas (`eval_delta_token_accuracy`, `eval_delta_exact_match`) are injected into the primary metrics dict → comet_ml, and are usable as `metric_for_best_model`.
- **Not a model change:** the all-pad-context trick avoids editing `grad_memgpt.py`. "Without memory" here means initial (meta-learned, unwritten) memory, not zero memory tokens.

## Recent additions (this session)
1. **Per-segment inner-loss diagnostic** (`inner_loss_seg_min`, `inner_loss_seg_max` in `inner_loop_stats`; surfaced as eval metrics). Answers whether fact-segments reconstruct well under noise (min ≈ facts-only loss) or whether everything fails (min ≈ mean). Decision tool for whether a reconstruction pretrain is the right fix.
2. **Reconstruction-only pretrain warmup** — config params `recon_pretrain_steps`, `recon_pretrain_target_weight`, `recon_pretrain_recon_weight` (default-off). Linear schedule over `recon_pretrain_steps`: target-loss weight `target_weight → 1.0`, extra inner-recon weight `recon_weight → 0.0`. Reuses the existing last-segment graph (memory-safe). See `_recon_pretrain_target_weight_now()` / `_recon_pretrain_recon_weight_now()` in `grad_memgpt.py`.
3. New configs: `configs/gradmemgpt/babi/hopfield_reconpretrain.yaml` (recon-pretrain on noisy qa3_4k) and `hopfield_reconpretrain_0k.yaml` (recon-pretrain on clean qa3_0k, then fine-tune via `--init_checkpoint`).
4. **VAE-style memory regularisation** (adaptive fork `grad_memgpt_adaptive.py` only; default-off). Two knobs under the `adaptive:` config section, both `0` by default so existing configs are unchanged:
   - `mem_noise_std` (σ): after the WRITE inner loop the deterministic memory `m` is perturbed as `m_used = m + σ·ε`, `ε ~ N(0,I)`, and READ consumes `m_used`. Reparameterised → grads flow to the mean `m` (and through it to the inner-loop params); `ε` carries no grad. **Train-only** (`self.training` gate) so eval READ stays deterministic and `token_accuracy` is stable. Forcing-Robustness bottleneck; with σ fixed, MSE (not full KL) is the correct prior form.
   - `mem_prior_weight` (λ) + `mem_prior_anneal_steps`: adds `λ·MSE(m_0, m_final)` to the **outer** loss only, where `m_0` are the initial (meta-learned) memory tokens. `m_0` is **detached** in the term → the reg pulls `m → m_0` but never `m_0 → m` (avoids the "prior drifts to satisfy the reg" collapse). Computed on the deterministic endpoint (pre-sampling), not the noisy sample. λ warms `0 → mem_prior_weight` over `mem_prior_anneal_steps` then constant. Driven by `set_train_step` / `_mem_prior_weight_now` — note the adaptive model extends `PreTrainedModel` directly (NOT the base `grad_memgpt.GradMemGPT`), so those accessors are defined on the adaptive class itself; without them `CustomTrainer.compute_loss`'s `hasattr(m,"set_train_step")` call silently no-ops and λ never ramps.
   - New eval metrics: `mem_prior_loss`, `mem_prior_weight_now`, `mem_sampled_delta_norm_mean`. Placement note: the sampling block sits between the `mem_norm`/`delta_mem_norm` stats (computed pre-sampling, so they keep measuring genuine inner-loop drift) and the READ phase; the per-segment forgetting probes snapshot even earlier (inside the inner loop), so they are unaffected regardless.
   - Config: commented default-off keys in `configs/gradmemgpt/kv_retrieval/adaptive_mamba.yaml`. `adaptive_curriculum.yaml` left untouched.
   - **Deferred next steps** (discussed but not implemented): learned per-dim / predicted σ (would need full KL to avoid σ-collapse); inner-loop prior reg (folds `λ·MSE(m_0,m_t)` into `seg_inner_loss` — changes test-time dynamics, riskier with second-order); per-segment/per-step prior (conflicts with the last-segment-only graph budget).
5. **STAR worst-case stability regulariser** (adaptive fork `grad_memgpt_adaptive.py` only; default-off; no new parameters). **Full guide: [`STAR.md`](./STAR.md)** — method, equations, config reference, run protocol, diagnostics, footguns. Summary:
   `L_STAR(m_s) = KL(READ(m_s; p*) ‖ READ(m_s+δ; p*))`, `δ = δ0 + γ·(‖m_j‖/‖g_j‖)·g_j` per memory token j (one noise-initialised gradient-ascent step; `g = ∇KL` at the perturbed point). p* = teacher-forced `?!K:V!|` probes of pairs already written, restricted to currently-retrievable ones (exact match under m_s) — STAR's x*. The perturbation stands in for the memory movement that LATER segments' inner-loop updates cause; minimising the worst-case KL makes retrieval robust in every direction. Only the perturbed branch carries meta-gradient (δ and the reference branch are detached, the paper's first-order approximation) → never exceeds the existing second-order autograd depth. Train-only (`self.training` gate): the test-time WRITE/READ procedure and eval forwards are bitwise unchanged; `star_weight=0` is a bitwise no-op (locked by `test_grad_memgpt_adaptive_star.py`).
   - Knobs under `adaptive:`: `star_weight` (λ, 0=off), `star_gamma` (relative perturbation size, paper grid 0.001–0.05), `star_epsilon` (δ0 noise scale), `star_ascent_steps` (1; ≥5 degrades), `star_perturbation` `"grad"|"random"` (random = matched-magnitude noise, the paper's Table-3 ablation — and the adversarial analogue of `mem_noise_std`), `star_probe_scope` `"all"|"past"`, `star_correct_only`, `star_boundaries` `"all"|"last"`, `star_anneal_steps` (λ warmup via `set_train_step`, like the mem-prior).
   - **Probe plumbing:** enabling `star_weight>0` makes the train collator attach `kv_queries` (built by `build_kv_probe_queries`, the SAME teacher-forced probe construction + KV→segment attribution as the per-segment forgetting eval) as a **top-level batch key**; the model takes it via a matching `kv_queries` forward arg. Top-level, NOT inside `input_ids`: the Trainer's eval input-decoding pads `input_ids` leaves and raises on non-tensor types (a list of ignore-token ids inside `input_ids` crashes `pad_across_processes`).
   - New TRAIN-time metrics: `train_star_kl` / `train_star_delta_ratio` / `train_star_n_probes` / `train_star_weight_now` (+ `train_replay_ce` / `train_replay_n_probes` / `train_replay_em_rate` / `train_replay_weight_now`), injected into the HF TRAINING logs by `CustomTrainer.compute_loss`/`.log` — the standard eval-only `inner_loop_stats`→comet path never sees train-only terms. Enabled-but-idle terms (x* mask empty, e.g. early training where nothing is retrievable yet) log ZEROS rather than going silent, so "off" vs "idle" is distinguishable on comet. `star_delta_ratio` must sit near `sqrt(star_gamma² + star_epsilon²)`: δ0 is VECTOR-normalised per memory token (a bare `eps·‖m‖·randn` has norm `eps·sqrt(d)·‖m‖`, ~11× too large at d=128 and drowns the γ-scaled ascent direction; a test locks the calibration). The WRITE trajectory itself is UNTOUCHED by STAR (outer-loss-only term; predictions/mem bitwise identical with STAR on vs off — also locked by tests).
   - Config: `configs/gradmemgpt/kv_retrieval/adaptive_star.yaml` (kv4, `n_segments: 2` — boundary at token 16 → pairs {1,2} in seg 0, pair 3 straddles into seg 1, pair 4 in seg 1; NOTE `segment_size: 14` would give ceil(32/14)=3 segments on the 32-token padded kv4 context). Tests: `test_grad_memgpt_adaptive_star.py` (bitwise-off, write-path invariance, meta-grad flow incl. `grad_mode="none"`, empty-scope guard, random-δ calibration).
   - **Stage-0 controls** (prerequisite, STAR off): `manifests/stage0_capacity_vs_interference.yaml` — m∈{3,4} × n_segments∈{1,2} + a same-seed repeat, deciding whether the 2-segment drop is capacity (→ pilot m=4) or interference (→ pilot m=3, STAR's target). Acceptance metric for the STAR pilot: forgetting gap `G = EM_1seg − EM_2seg` at fixed m should shrink toward 0, with ladder ordering baseline ≤ random-δ ≤ grad-δ on retention.
   - **Vanilla-replay ablation (outer loss, default-off):** `replay_weight` (ρ) + `replay_boundaries` / `replay_probe_scope` / `replay_correct_only` / `replay_anneal_steps` — CE of `READ(m_s; probe)` against the probe's answer tokens at the SAME boundaries/probes/EM-mask as STAR (`_replay_boundary_loss`). Corrective where STAR is preventive (non-zero gradient whenever an old pair is answered imperfectly; can recover pairs STAR's EM-mask ratchets away). One probe forward per boundary, cheaper than STAR. The {replay}×{STAR} 2×2 on identical data isolates refitting vs worst-case consistency; new metric `replay_em_rate` is a live forgetting measurement during training. The train-collator probe gate now fires for `star_weight>0` OR `replay_weight>0` (`use_probes`).
   - **Local no-network runs:** this machine cannot reach comet.com; launch runs with `COMET_START_ONLINE=0` (offline comet zips under `.cometml-runs/`, all trainer metrics/forgetting matrices still land locally) and `CUDA_VISIBLE_DEVICES=0` (`.envrc` points at GPU 1, which does not exist locally).
6. **Per-segment forgetting eval for the main model** (`grad_memgpt.py`; previously adaptive-only). `training.per_segment_eval: true` now also works without an `adaptive:` section — the trainer's `PerSegmentForgettingTrainer` path is unchanged, the model grew the adaptive fork's `forward_per_segment_eval` + `forward(collect_segment_mems=True)`. Key differences vs the adaptive fork, all forced by the main model's architecture:
   - **Segmentation source:** the probe collator attributes KVs to segments using `hopfield_n_segments` / `hopfield_segment_size` (NOT the adaptive `n_segments`/`segment_size`), and `build_kv_probe_queries` gained `seg_pad_side='left'` because `grad_memgpt.py` LEFT-pads the context to a segment-aligned length before chunking (the adaptive fork right-pads; unshifted attribution would land one segment off near boundaries). Plain runs ignore `hopfield_segment_size` (forward honors it only with hopfield/gated-delta on) — the runner mirrors that condition. Under curriculum, `hopfield_n_segments` is stage-overridable but the per-segment collator is built once: overriding it desyncs the probe attribution.
   - **Boundary memory semantics** (`_probe_boundary_memory`): plain modes RESET mem per segment, so cell (s,k), s<k, measures reset-forgetting (architecture property — only the last segment's mem ever reaches READ); Hopfield / Gated Delta ACCUMULATE their store across segments, and since their retrieval is QUERY-dependent it is re-run per probe against the store as of boundary k (all retrieval modes/projections/unwritten fallbacks mirrored from forward's RETRIEVE phases) — directly comparable to the adaptive fork's interference-forgetting matrix.
   - **Snapshots are O(1) per boundary:** plain = detached `mem`; Hopfield = the store COUNT only (stored keys/values/masks are append-only within one forward, so `final_list[:n_stored]` reconstructs the boundary state); Gated Delta = detached `S` + per-sample `written` flag (the one O(B·d²)/boundary case). Purity is test-locked: eval predictions / train loss are bitwise identical with `collect_segment_mems` on vs off.
   - Tests: `test_grad_memgpt_per_segment_eval.py` (purity, snapshot structure incl. empty-segment None alignment, triple invariants for plain/hopfield/gated-delta/ctrl-token configs, guards, collator pad-side attribution).

## Notes
- No lint/typecheck commands - Python-only research codebase
- Uses transformers + torch with custom meta-learning logic
- Dependencies live in `pyproject.toml`; `uv.lock` pins the resolved graph. PyTorch CUDA wheels come from the pinned `cu121` index (`[tool.uv.sources]` in `pyproject.toml`) — adjust the index URL there to switch CUDA versions. `cu121` + torch 2.5.1 targets the remote GPU machine (CUDA 12.3 driver, Python 3.10); cu121 wheels run on any 12.x driver, so the local machine is unaffected.