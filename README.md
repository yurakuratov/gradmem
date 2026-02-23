# Test Time Gradient Descend for Memory Update

This repository contains small experiments around "test time" gradient updates for key--value retrieval tasks. The main goal is to train compact GPT style models (both vanilla and models with an adaptive memory) to recover values that appear in the context.


## Intro
Large-context transformers pay a **quadratic cost** every time they reread long prompts.

Our goal is to compress those prompts into a **small, writable parameter block `[mem]`** that we update with a few gradient steps at test time, then drop the original text entirely.

### How it works

| Phase | What happens | N_iters | Input size |
|-------|--------------|--------------|------------|
| **Write (inner loop, *K* steps)** | Show the context, compute an LM loss **L<sub>inner</sub>**, update **`[mem]` only** | *K* | `[mem]` + `context` |
| **Read (outer loop)** | Discard the context; answer the query with the **updated `[mem]`** and compute **L<sub>outer</sub>** | 1 |  `query` |

*Back-propagating L<sub>outer</sub> meta-trains both the Transformer weights θ and the **initial memory `[mem]_0`**, so the model learns how to “write quickly.”*

### Gradient-flow modes

| Flag | What gradients reach `[mem]_0`? | Extra VRAM cost | Typical use-case |
|------|---------------------------------|-----------------|------------------|
| `none` (“Frozen”) | **None** (detach) | None | Baseline sanity check |
| `first` (“1st-order”) | Straight-through, no Hessian term | None | Fast runs, XX% of full accuracy |
| `second` (“2nd-order”) | Full MAML (keeps full graph through the *K* inner steps) | **≈ K × activation-memory** (parameters are shared; what multiplies is the *activations* for each inner forward/backward) | Highest accuracy when GPU RAM is sufficient |


## Prerequisites

* Python 3.11
* [conda](https://docs.conda.io/en/latest/) for environment management

Create an environment using the provided YAML file:

```bash
conda env create -f conda_env.yaml
conda activate /home/jovyan/kuratov/envs/py311_pt2.6_cu12.4  # or the path printed by conda
```

Accelerate is configured via `accelerate.yaml`. The default configuration uses BF16 precision and a single process.

## Datasets

### KV-retrieval
Datasets consist of sequences containing random text segments with embedded `!key:value!` pairs. The last segment queries one of the previous keys (e.g. `?!K:`) and the model must output the corresponding value.

To generate a dataset run the notebook `notebooks/dump_dataset.ipynb`. It relies on `kv_dataset_utils.generate_sequence` to create individual samples and dumps them using Hugging Face `datasets`. The resulting directory will be saved under `./data/<DATASET_NAME>` where `DATASET_NAME` encodes generation parameters, for example `N8_K2V2_1M`.

Download KV-retrieval datasets from HF:
```bash
./scripts/download_kv_retrieval.sh
```

### bAbI
Download bAbI datasets from HF:
```bash
./scripts/download_babi.sh
```


## Training

Two entry points are provided:

* `run_gpt2_on_kv_retrieval.py` &ndash; trains a standard causal LM.
* `run_gradmemgpt_on_kv_retrieval.py` &ndash; trains a small LM with writable memory (see `grad_memgpt.py`).

Both scripts accept the same arguments (batch size, number of layers, dataset path, etc.). They should be launched through `accelerate`:

```bash
accelerate launch --config_file accelerate.yaml \
  run_gpt2_on_kv_retrieval.py \
  --exp_path ./runs/gpt2_example \
  --per_device_batch_size 64 \
  --data_path ./data/N10-K4V4-S4(32-64)_1M
```

```bash
accelerate launch --config_file accelerate.yaml \
  run_gradmemgpt_on_kv_retrieval.py \
  --exp_path ./runs/gradmem_example \
  --per_device_batch_size 64 \
  --data_path ./data/N10-K4V4-S4(32-64)_1M
```

The scripts log metrics and save checkpoints to the directory specified via `--exp_path`.

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
