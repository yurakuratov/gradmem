# AGENTS.md - Test Time Gradient Descent

## Environment Setup
```bash
conda env create -f conda_env.yaml
conda activate /home/jovyan/kuratov/envs/py311_pt2.6_cu12.4  # adjust path as needed
```

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
- `generate_run_name.py` - utility for run name generation

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
The `conda` paths in the block above are from a different host. On **this** machine:
```bash
export PATH="$HOME/.pyenv/bin:$PATH"
eval "$(pyenv init -)"
pyenv activate gradmem          # Python 3.11.2, torch 2.11.0+cu130
```
- `pyenv` lives at `~/.pyenv/bin/pyenv`; plain `pyenv activate` fails unless `~/.pyenv/bin` is on PATH first.
- Other envs available: `dhtm`, `dhtm310`, `belief`, `chicken`, etc. `gradmem` is the one for this repo.
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
- `inner_loop_stats` accumulates: `inner_loss`, `target_loss`, `rec_loss` (RMT only), `energy`/`energy_recon` (energy configs), `inner_grad_norm_*`, `mem_norm_*`, `delta_mem_norm_*`, `seg_nonempty_*`, `hopfield_*` or `gd_*` diagnostics, plus the per-segment diagnostic `inner_loss_seg_min/max` and pretrain schedule weights (see "Recent additions" below).
- `preprocess_logits_for_metrics` passes `inner_loop_stats` through to `compute_metrics_fn`, which adds each stat as an eval metric (auto-prefixed `eval_` by HF Trainer) → reported to comet_ml.
- **The step schedule** (`energy_recon` anneal, recon-pretrain) is driven by `current_train_step`, stamped via `CustomTrainer.compute_loss` → `model.set_train_step(self.state.global_step)` (run script lines 239-247). In `CurriculumTrainer`, `global_step` resets per stage, so schedules restart each curriculum stage — beware if combining schedules with curriculum.

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

## Notes
- No lint/typecheck commands - Python-only research codebase
- Uses transformers + torch with custom meta-learning logic
- conda_env.yaml contains hardcoded path prefix - adjust for local environment