# GradMem: Learning to Write Context into Memory with Test-Time Gradient Descent

This repository contains code for **GradMem**, a memory mechanism where the model compresses a context into a small writable memory state using **test-time gradient descent**.

The key idea is not just to optimize memory at inference, but to **meta-learn the model so that a few (<=5) test-time updates are effective**.

arXiv: https://arxiv.org/abs/2603.13875

<p align="center">
  <img src="./imgs/GradMem_overview_short.png" width="80%" />
</p>

## Concept

Each example is split into:
- `context C` (information to store),
- `query Q` (what to ask),
- `target Y` (expected output).

GradMem runs in two phases:

1. **WRITE (inner loop, K steps)**
   - Start from learned memory initialization `M0`.
   - Optimize memory tokens `M` on a self-supervised reconstruction loss `L_write(M, C)`.
   - Update only memory state; model weights stay fixed during this phase.

2. **READ (outer objective)**
   - Predict `Y` from `[M; Q]` after WRITE.
   - Train model parameters and `M0` by backpropagating task loss through the WRITE updates (meta-learning).

In short: GradMem learns to use gradient descent itself as a writing operation.

## Experimental self-distillation

`run_self_distillation_on_kv_retrieval.py` inherits the complete KV GradMem CLI
(memory backends, WRITE objectives, energy parameters, and auxiliary losses).
`self_distillation.py` adds a teacher forward on `[context; query; target]`
before ordinary query-independent WRITE and context-free READ. The teacher is
the current READ backbone without fast memory or WRITE-only adapters; its outputs
are detached, and its forward runs under `no_grad`. It follows the model's current
train/eval mode, including dropout. There is no separate or EMA teacher.

The final outer loss is weighted READ CE + weighted distillation + all existing
auxiliary losses. These nonnegative CLI weights can be combined:

| Flag | Python default | Loss |
| --- | --- | --- |
| `--read_ce_weight` | 1 | Final READ target CE |
| `--distill_logits_query_weight` | 0 | Query-token KL(teacher ‖ student) |
| `--distill_logits_target_weight` | 0 | Target-token KL(teacher ‖ student) |
| `--distill_hidden_query_weight` | 0 | Query-token hidden MSE |
| `--distill_hidden_target_weight` | 0 | Target-token hidden MSE |

KL uses temperature 1 and the logits **preceding** each selected token. Hidden
MSE uses states **at** each token, averaging over hidden dimensions, non-padding
selected tokens, and equally over all transformer layers (no embedding layer).
These are Hugging Face hidden-state outputs, including final normalization on
the last layer. Target tokens have labels other than `-100`; other non-padding
tokens are query tokens. Empty selections contribute zero. LoRA/KV memory omit
the first token from KL because they have no preceding input position. Only
causal READ alignment is supported. Existing intermediate-READ and auxiliary
loss definitions/weights are unchanged; `read_ce_weight` weights final CE only.

The shell preset reuses the energy launcher and enables target KL with weight 1
alongside CE with weight 1. Its other weights default to zero. Uppercase weight
environment variables or CLI flags override these settings; **all** inherited
runner flags can also be passed, with CLI overrides taking precedence:

```bash
# Example configuration; select an available GPU before launching.
NP=1 bash scripts/run_self_distillation_on_kv_retrieval.sh \
  --write_objective reconstruction \
  --distill_hidden_query_weight 0.1 \
  --distill_hidden_target_weight 0.1
```

Existing backend restrictions still apply (energy WRITE requires prefix memory).
Run names include the distillation weights and a CLI digest to distinguish
overridden inherited settings. Configuration/checkpoint serialization and resume
use the shared runner. With all distillation weights zero, the teacher is skipped;
without labels, inference always skips it. Full-context teacher evaluation with
labels is more expensive than deployment READ and requires room for the full
context + query + target sequence.

## Repository layout

- `grad_memgpt.py` - GradMem model and inner-loop memory optimization logic.
- `rmt.py` - recurrent memory transformer baseline with forward-only memory updates.
- `run_gradmemgpt_on_*.py` - GradMem training/eval entry points.
- `run_rmt_on_*.py` - RMT baseline entry points.
- `run_gpt2_on_*.py` - non-compressive causal LM baselines.
- `kv_dataset_utils.py` - synthetic key-value retrieval data generation and tokenizer helpers.
- `squad_utils.py`, `phonebook_utils.py` - NLP dataset preprocessing.
- `prepare_pg19_chunks.py` - PG19 chunking for language modeling/compression experiments.
- `attn_double_bwd/` - custom attention double-backward implementations for higher-order GradMem training.
- `scripts/` - runnable experiment presets and dataset download scripts.

## Setup

Requirements:
- Python 3.11
- [conda](https://docs.conda.io/en/latest/)

Create environment:

```bash
conda env create -f conda_env.yaml
conda activate /home/jovyan/kuratov/envs/py311_pt2.6_cu12.4
```

`accelerate.yaml` contains a default single-process setup.

### FlashBoB (second-order attention)

GradMem and EnergyGradMem share a **workspace-local** FlashBoB implementation in
`attn_double_bwd/flashbob/` and its Transformers adapter in
`attn_double_bwd/flashbob_attention.py`. No sibling checkout, editable package,
or separately installed `bob` module is required. Use a compatible environment
(PyTorch 2.8/Triton 3.4 or PyTorch 2.9/Triton 3.5):

```bash
conda env create -f environment-flashbob.yaml
```

Use `attn_implementation="flashbob"` in `GradMemGPTConfig`, or pass
`--attn_implementation flashbob` to any `run_gradmemgpt_on_*.py` entry point
(KV retrieval/bAbI, MQAR, SQuAD variants, and text compression). `bob` is an alias.
Both separate READ/WRITE models use the selected backend, which is checkpointed.
Existing energy objectives and shaping use this same shared model path.

All GradMem/EnergyGradMem bash presets accept an environment override and encode
non-eager attention in the run name, including the shaping wrapper:

```bash
CUDA_VISIBLE_DEVICES=0 NP=1 ATTN_IMPL=flashbob conda run -n gradmem-flashbob \
  bash scripts/run_energygradmem_shaping_on_kv_retrieval.sh
```

CUDA acceleration supports MHA FP16/BF16 head dimensions 32, 48, 64, 128;
square GQA/MQA supports BF16 dimensions 32, 64, 128 (group size <=32).
CPU MHA uses FlashBoB's reference implementation. Explicit masks (including
padding and cached causal masks), attention dropout, unsupported dtypes/shapes,
and requested attention weights use a **warned differentiable math fallback**.
FP32 runs remain valid but are not accelerated on CUDA. No mask or dropout is
silently discarded. Q/K/V follow the explicitly selected autocast dtype, just
as with PyTorch SDPA; outside autocast, model dtypes are unchanged. For acceleration,
use BF16/FP16 and zero attention dropout; pretrained GPT-2 commonly has nonzero
attention dropout during training. FlashBoB supports reverse-mode derivatives
through second order, not arbitrary higher-order/forward-mode derivatives.

On hosts exposing only `libcuda.so.1`, Triton may also need a `libcuda.so` linker
symlink and `TRITON_LIBCUDA_PATH` pointing to its directory. Keep that setup local
to the Conda environment rather than modifying system libraries.

Integration checks (CUDA tests run on the explicitly selected GPU):

```bash
CUDA_VISIBLE_DEVICES=0 conda run -n gradmem-flashbob \
  python -m pytest -q tests/test_flashbob_attention.py
```

Other attention variants live alongside it in `attn_double_bwd/`:

- `hvp_manual`: explicit forward, first backward, and second backward, including
  consistent dropout-mask reuse.
- `hvp_semi_manual`: PyTorch ATen FlashAttention forward/first backward for
  eligible CUDA FP16/BF16 calls, with explicit second backward. Other inputs
  use a warned manual fallback. Sequence lengths need not be multiples of 32.
- `flashbob` (alias `bob`): fused forward/first backward and Triton second
  backward, subject to the support limits above.

The obsolete `jvp_flash` implementation has been removed. Use one of the
reverse-mode implementations above for second-order GradMem training.

## Data

### KV retrieval

Synthetic samples contain `!key:value!` pairs in context and a query like `?!K:`; target is the corresponding value.

Download prepared datasets from Hugging Face:

```bash
./scripts/download_kv_retrieval.sh
```

You can also generate data with `notebooks/dump_dataset.ipynb` (uses `kv_dataset_utils.generate_sequence`).

The framed MQAR noise datasets can be downloaded with:

```bash
./scripts/download_mqar_noise_datasets.sh
```

### bAbI

```bash
./scripts/download_babi.sh
```

### PG19 chunks (for text compression / LM experiments)

```bash
./scripts/prepare_pg19_chunks.sh
```

## How to run

### 1) Baseline causal LM (no writable memory)

```bash
accelerate launch --config_file accelerate.yaml \
  run_gpt2_on_kv_retrieval.py \
  --exp_path ./runs/gpt2_example \
  --per_device_batch_size 64 \
  --data_path ./data/N16-K2V2-V62_1M \
  --tokenizer_path ./tokenizers/kv_alphabet_62/ \
  --base_model llama
```

### 2) GradMem on KV retrieval

```bash
accelerate launch --config_file accelerate.yaml \
  run_gradmemgpt_on_kv_retrieval.py \
  --exp_path ./runs/gradmem_example \
  --per_device_batch_size 64 \
  --data_path ./data/N16-K2V2-V62_1M \
  --tokenizer_path ./tokenizers/kv_alphabet_62/ \
  --base_model llama \
  --n_mem_tokens 8 \
  --K 2 \
  --inner_lr 0.04 \
  --grad_mode second
```

For full experiment configurations, use scripts in `scripts/`:
- `scripts/run_gradmemgpt_on_kv_retrieval.sh`
- `scripts/run_energygradmem_on_kv_retrieval.sh`
- `scripts/run_rmt_on_kv_retrieval.sh`
- `scripts/run_gpt_on_kv_retrieval.sh`

Enable the training-only EnergyGradMem memory search with environment overrides:

```bash
ENERGY_MEMORY_SEARCH_WEIGHT=0.1 \
ENERGY_MEMORY_SEARCH_NUM_SAMPLES=4 \
ENERGY_MEMORY_SEARCH_RADIUS_SCALE=0.25 \
ENERGY_MEMORY_SEARCH_USE_GAIN_WEIGHTING=true \
ENERGY_MEMORY_SEARCH_GAIN_EMA_DECAY=0.99 \
ENERGY_MEMORY_SEARCH_USE_BEST_FOR_NEXT_STEP=true \
bash scripts/run_energygradmem_on_kv_retrieval.sh
```

The search runs around `M_1...M_K`, preserves each memory's global L2 norm,
and uses a radius equal to `ENERGY_MEMORY_SEARCH_RADIUS_SCALE` times the
preceding WRITE-step displacement. It requires full second-order SGD WRITE.
Gain weighting uses each selected candidate's relative READ-loss reduction and
normalizes positive gains with a persistent EMA updated once per microbatch.
The optional best-for-next-step mode performs search inline and starts the next
training WRITE step from a straight-through copy of the selected memory. After
the final SGD step, READ uses the selected perturbation as its forward memory;
the search loss still attracts the SGD-derived state toward that detached target.
The full training rollout is target-guided, while inference follows the ordinary
query-independent SGD trajectory.

### Analysis And Reports

Use the unified JSON CLI for MQAR landscapes, K extrapolation, and beneficial
memory-search reports:

```bash
conda run -n rmt python -m analysis_tools experiment.json
```

See [`analysis_tools/README.md`](analysis_tools/README.md) and
[`analysis_tools/examples/`](analysis_tools/examples/) for configuration examples.
All analysis sources are under `analysis_tools/`; generated artifacts are under
`reports/`. Historical `analysis/` results were moved intact; the mapping is in
`reports/migration.md`. Diffusion sweeps and error-placement tools are retained
under `analysis_tools/diffusion/`, and specialist KV/mixed-objective tools under
`analysis_tools/legacy/`.

### Cached checkpoint evaluation

`analysis_tools/legacy/evaluate_energy_shaping.py` stores expensive task and landscape measurements in
`checkpoint-*/energy_shaping_metrics/`. Run it once with `--cache-only` to
precompute checkpoint data, then run later comparisons with the same evaluation
settings and a new `--output-dir`; model and dataset loading are skipped when all
requested caches exist. Use `--recompute-metrics task radial` (or `all`) to
selectively refresh metric families.

Metric families have independent versions in `METRIC_FAMILY_VERSIONS`. When a
stored metric changes, bump only that family's version; adding another family
does not invalidate existing task or landscape measurements.

Precompute every supported checkpoint under `runs/` with:

```bash
conda run -n rmt python scripts/cache_all_energy_shaping_checkpoints.py \
  --runs-dir runs -- --device cpu
```

Arguments after `--` are forwarded to `analysis_tools.legacy.evaluate_energy_shaping`. Use
`--dry-run` before a large sweep, and pass the same evaluation settings that
will be used for later comparisons.
- `scripts/run_gradmemgpt_on_babi.sh`
- `scripts/run_gradmemgpt_on_squad.sh`


For text-compression experiments, prepare PG19 chunks with
`scripts/prepare_pg19_chunks.sh` and run `run_gradmemgpt_on_text_compression.py` directly with `accelerate`.

## GradMem options

`grad_mode` controls gradient flow through WRITE updates:

- `none` - no meta-gradient through inner updates.
- `first` - first-order approximation.
- `second` - full second-order differentiation through inner loop (default for strongest results).

Second-order mode is more expensive in memory/compute, but it is what we found that actually makes GradMem to learn; `attn_double_bwd/` includes optimized double-backward implementations for attention for second-order optimization.

## Outputs

All runs write checkpoints, metrics, and trainer state under `--exp_path` (typically in `./runs/...`).


## Tests

Pytest smoke tests for `GradMemGPT` live in `tests/test_gradmemgpt.py` and cover:

- Model families: `gpt2`, `gpt_neox`, `llama`
- Memory backends: `prefix`, `lora`

Markers:

- `forward`: fast forward-only checks (`test_forward`)
- `one_batch_train`: one optimizer step on a single batch (`test_single_batch_train`)
- `all`: union of `forward` and `one_batch_train`

Examples:

```bash
# Run all smoke tests (forward + one-batch-train)
python -m pytest -q -m all

# Run only forward tests
python -m pytest -q -m forward

# Run only one-batch training tests
python -m pytest -q -m one_batch_train

# Run tests for selected backend(s) only
python -m pytest -q -m all --backend prefix
python -m pytest -q -m all --backend prefix,lora

# Run tests for selected model family/families only
python -m pytest -q -m all --model-family llama
python -m pytest -q -m all --model-family gpt2,gpt_neox

# Combine backend + model-family filters
python -m pytest -q -m all --backend prefix --model-family llama
```

## Citation
```
@misc{kuratov2026gradmem,
      title={GradMem: Learning to Write Context into Memory with Test-Time Gradient Descent}, 
      author={Yuri Kuratov and Matvey Kairov and Aydar Bulatov and Ivan Rodkin and Mikhail Burtsev},
      year={2026},
      eprint={2603.13875},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2603.13875}, 
}
```
