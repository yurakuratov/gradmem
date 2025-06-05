# Test Time Gradient Descend for Memory Update

This repository contains small experiments around "test time" gradient updates for key--value retrieval tasks. The main goal is to train compact GPT style models (both vanilla and models with an adaptive memory) to recover values that appear in the context.

## Prerequisites

* Python 3.11
* [conda](https://docs.conda.io/en/latest/) for environment management

Create an environment using the provided YAML file:

```bash
conda env create -f conda_env.yaml
conda activate /home/jovyan/kuratov/envs/py311_pt2.6_cu12.4  # or the path printed by conda
```

Accelerate is configured via `accelerate.yaml`. The default configuration uses BF16 precision and a single process.

## Dataset generation

Datasets consist of sequences containing random text segments with embedded `!key:value!` pairs. The last segment queries one of the previous keys (e.g. `?!K:`) and the model must output the corresponding value.

To generate a dataset run the notebook `notebooks/dump_dataset.ipynb`. It relies on `kv_dataset_utils.generate_sequence` to create individual samples and dumps them using Hugging Face `datasets`. The resulting directory will be saved under `./data/<DATASET_NAME>` where `DATASET_NAME` encodes generation parameters, for example `N10-K4V4-S4(32-64)_1M`.

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

