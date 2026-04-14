"""Utilities for generating experiment run names from configuration."""

import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


def generate_run_name_gpt2(cfg: Dict[str, Any]) -> str:
    """Generate run name for GPT2 model."""
    model = cfg.get('model', {})
    training = cfg.get('training', {})

    base_model = model.get('base_model', 'gpt2')
    n_layer = model.get('n_layer', 4)
    n_head = model.get('n_head', 4)
    n_embd = model.get('n_embd', 128)
    max_pos = model.get('max_position_embeddings')

    tbs = training.get('total_batch_size', 64)
    lr = training.get('learning_rate', 1e-4)
    adam_beta2 = training.get('adam_beta2')

    if base_model == 'mamba':
        run_name = f"{base_model}_L{n_layer}D{n_embd}"
    else:
        run_name = f"{base_model}_L{n_layer}H{n_head}D{n_embd}"
        if max_pos:
            run_name += f"_L{max_pos}"

    run_name += f"_bs_{tbs}_lr_{lr}"

    if adam_beta2:
        run_name += f"_b2_{adam_beta2}"

    suffix = cfg.get('run_name_suffix')
    if suffix:
        run_name += f"_{suffix}"

    return run_name


def generate_run_name_gradmemgpt(cfg: Dict[str, Any]) -> str:
    """Generate run name for GradMemGPT model."""
    model = cfg.get('model', {})
    training = cfg.get('training', {})
    gradmem = cfg.get('gradmem', {})
    dataset = cfg.get('dataset', {})

    base_model = model.get('base_model') or model.get('pretrained_model', 'gpt2')
    n_layer = model.get('n_layer', 4)
    n_head = model.get('n_head', 4)
    n_embd = model.get('n_embd', 128)

    n_mem_tokens = gradmem.get('n_mem_tokens', 8)
    n_ctrl_tokens = gradmem.get('n_ctrl_tokens', 0)
    K = gradmem.get('K', 2)
    inner_lr = gradmem.get('inner_lr', 0.04)
    grad_mode = gradmem.get('grad_mode', 'second')
    use_mem_proj = gradmem.get('use_mem_proj', False)
    mem_proj_mode = gradmem.get('mem_proj_mode', 'none')
    use_write_head = gradmem.get('use_write_head', False)
    use_adam = gradmem.get('use_adam', False)
    inner_clip_value = gradmem.get('inner_clip_value')
    inner_clip_norm = gradmem.get('inner_clip_norm')
    add_inner_loss = gradmem.get('add_inner_loss_to_outer', False)
    inner_loss_weight = gradmem.get('inner_loss_weight')
    last_K_second_order = gradmem.get('last_K_second_order', K)

    tbs = training.get('total_batch_size', 64)
    lr = training.get('learning_rate', 1e-4)

    run_name = f"gradmem_{base_model}_L{n_layer}H{n_head}D{n_embd}_mem{n_mem_tokens}"
    if n_ctrl_tokens > 0:
        run_name += f"_c{n_ctrl_tokens}"

    run_name += f"_K{K}_ilr{inner_lr}"

    if grad_mode == 'second' and last_K_second_order != K:
        run_name += f"_last_K{last_K_second_order}"

    if inner_clip_value is not None:
        run_name += f"_icv{inner_clip_value}"

    if inner_clip_norm is not None:
        run_name += f"_icn{inner_clip_norm}"

    if use_mem_proj:
        run_name += "_mem_proj"
        if mem_proj_mode == 'per_sample':
            run_name += "_ps"

    if use_write_head:
        run_name += "_whead"

    run_name += f"_grad_{grad_mode}"

    if add_inner_loss:
        run_name += "_add_inner"
        if inner_loss_weight is not None:
            run_name += f"_w{inner_loss_weight}"

    if use_adam:
        run_name += "_with_adam"

    run_name += f"_bs_{tbs}_lr_{lr}"

    suffix = cfg.get('run_name_suffix')
    if suffix:
        run_name += f"_{suffix}"

    return run_name


def generate_run_name_rmt(cfg: Dict[str, Any]) -> str:
    """Generate run name for RMT model."""
    model = cfg.get('model', {})
    training = cfg.get('training', {})
    rmt = cfg.get('rmt', {})

    base_model = model.get('base_model', 'gpt2')
    n_layer = model.get('n_layer', 4)
    n_head = model.get('n_head', 4)
    n_embd = model.get('n_embd', 128)

    n_mem_tokens = rmt.get('n_mem_tokens', 8)
    n_ctrl_tokens = rmt.get('n_controller_tokens', 0)

    tbs = training.get('total_batch_size', 64)
    lr = training.get('learning_rate', 1e-4)

    run_name = f"rmt_{base_model}_L{n_layer}H{n_head}D{n_embd}_mem{n_mem_tokens}"
    if n_ctrl_tokens > 0:
        run_name += f"_c{n_ctrl_tokens}"

    run_name += f"_bs_{tbs}_lr_{lr}"

    suffix = cfg.get('run_name_suffix')
    if suffix:
        run_name += f"_{suffix}"

    return run_name


def get_model_type(cfg: Dict[str, Any]) -> str:
    """Detect model type from config."""
    if 'gradmem' in cfg:
        return 'gradmemgpt'
    if 'rmt' in cfg:
        return 'rmt'
    if cfg.get('model', {}).get('pretrained_model'):
        return 'gpt2_pretrained'
    return 'gpt2'


def generate_run_name(cfg: Dict[str, Any]) -> str:
    """Generate run name based on model type."""
    model_type = get_model_type(cfg)

    if model_type == 'gradmemgpt':
        return generate_run_name_gradmemgpt(cfg)
    elif model_type == 'rmt':
        return generate_run_name_rmt(cfg)
    else:
        return generate_run_name_gpt2(cfg)


def get_exp_path(cfg: Dict[str, Any], seed: Optional[int] = None) -> Path:
    """Generate experiment path from config."""
    dataset = cfg.get('dataset', {})
    training = cfg.get('training', {})

    data_name = dataset.get('data_name', 'default')
    run_name = generate_run_name(cfg)

    s = seed if seed is not None else training.get('seed', 1)

    exp_path = Path('./runs') / data_name / run_name / f'run_{s}'
    return exp_path


def get_data_path(cfg: Dict[str, Any]) -> str:
    """Get data path from config."""
    dataset = cfg.get('dataset', {})

    if 'data_path' in dataset:
        return dataset['data_path']

    data_name = dataset.get('data_name')
    if data_name:
        cache_dir = os.path.expanduser('~/.cache/test-time-gd-cache/data')
        return f"{cache_dir}/{data_name}"

    raise ValueError("Either data_path or data_name must be specified in dataset config")


def load_config(config_path: str | Path) -> Dict[str, Any]:
    """Load YAML config file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def build_args_from_config(
    cfg: Dict[str, Any],
    model_type: Optional[str] = None,
    overrides: Optional[Dict[str, Any]] = None
) -> list[str]:
    """Build command-line arguments from config."""
    if overrides is None:
        overrides = {}
    
    cfg = {**cfg, **overrides}

    args = []
    dataset = cfg.get('dataset', {})
    model = cfg.get('model', {})
    training = cfg.get('training', {})
    gradmem = cfg.get('gradmem', {})
    rmt = cfg.get('rmt', {})

    exp_path = get_exp_path(cfg)
    args.append(f'--exp_path={exp_path}')

    tbs = training.get('total_batch_size', 64)
    per_device = training.get('per_device_batch_size', 64)
    np = 1
    grad_acc = max(1, tbs // (per_device * np))
    
    args.append(f'--per_device_batch_size={per_device}')
    args.append(f'--gradient_accumulation_steps={grad_acc}')
    args.append(f'--total_batch_size={tbs}')
    args.append(f'--learning_rate={training.get("learning_rate", 1e-4)}')
    args.append(f'--max_steps={training.get("max_steps", 100000)}')
    args.append(f'--eval_steps={training.get("eval_steps", 500)}')
    args.append(f'--logging_steps={training.get("logging_steps", 500)}')
    args.append(f'--warmup_steps={training.get("warmup_steps", 1000)}')
    args.append(f'--seed={training.get("seed", 142)}')

    if training.get('adam_beta2'):
        args.append(f'--adam_beta2={training["adam_beta2"]}')
    if training.get('early_stopping_patience'):
        args.append(f'--early_stopping_patience={training["early_stopping_patience"]}')
    if training.get('use_gradient_checkpointing'):
        args.append(f'--use_gradient_checkpointing')

    if 'data_path' in dataset:
        args.append(f'--data_path={dataset["data_path"]}')
    elif 'data_name' in dataset:
        args.append(f'--data_path={get_data_path(cfg)}')

    if 'tokenizer_path' in dataset:
        args.append(f'--tokenizer_path={dataset["tokenizer_path"]}')

    if 'base_model' in model:
        args.append(f'--base_model={model["base_model"]}')
    if 'pretrained_model' in model:
        args.append(f'--pretrained_model={model["pretrained_model"]}')
    if 'n_layer' in model:
        args.append(f'--n_layer={model["n_layer"]}')
    if 'n_head' in model:
        args.append(f'--n_head={model["n_head"]}')
    if 'n_embd' in model:
        args.append(f'--n_embd={model["n_embd"]}')
    if 'max_position_embeddings' in model:
        args.append(f'--max_position_embeddings={model["max_position_embeddings"]}')

    if 'init_checkpoint' in cfg:
        args.append(f'--init_checkpoint={cfg["init_checkpoint"]}')

    for key, val in gradmem.items():
        if val is not None and val is not False:
            if val is True:
                args.append(f'--{key}')
            else:
                args.append(f'--{key}={val}')

    for key, val in rmt.items():
        if val is not None and val is not False:
            if val is True:
                args.append(f'--{key}')
            else:
                args.append(f'--{key}={val}')

    return args


if __name__ == '__main__':
    import sys

    if len(sys.argv) < 2:
        print("Usage: python generate_run_name.py <config.yaml>")
        sys.exit(1)

    cfg = load_config(sys.argv[1])
    name = generate_run_name(cfg)
    exp_path = get_exp_path(cfg)
    print(f"Run name: {name}")
    print(f"Exp path: {exp_path}")