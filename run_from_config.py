#!/usr/bin/env python
"""Entry point for running experiments from YAML configuration files.

Usage:
    # Dry run - show command without executing
    python run_from_config.py --config configs/gradmemgpt/kv_retrieval/default.yaml --dry-run

    # Run experiment (generates config and prints command)
    python run_from_config.py --config configs/gradmemgpt/kv_retrieval/default.yaml
"""

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path

import yaml

from generate_run_name import (
    generate_run_name, get_exp_path, get_data_path, load_config, get_model_type
)


logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)


SCRIPT_MAP = {
    'gpt2': 'run_gpt2_on_kv_retrieval.py',
    'gpt2_squad': 'run_gpt2_on_squad.py',
    'gpt2_babi': 'run_gpt2_on_kv_retrieval.py',
    'gpt2_phonebook': 'run_gpt2_on_squad.py',
    'gradmemgpt': 'run_gradmemgpt_on_kv_retrieval.py',
    'gradmemgpt_squad': 'run_gradmemgpt_on_squad.py',
    'gradmemgpt_squad_cq_a': 'run_gradmemgpt_on_squad_cq_a.py',
    'gradmemgpt_squad_qc_a': 'run_gradmemgpt_on_squad_qc_a.py',
    'gradmemgpt_babi': 'run_gradmemgpt_on_kv_retrieval.py',
    'gradmemgpt_phonebook': 'run_gradmemgpt_on_squad.py',
    'gradmemgpt_hopfield': 'run_hopfield_kv_eval.py',
    'rmt': 'run_rmt_on_kv_retrieval.py',
    'rmt_squad': 'run_rmt_on_squad.py',
    'rmt_phonebook': 'run_rmt_on_squad.py',
}


def detect_dataset_type(cfg, config_path=None) -> str:
    """Detect dataset type from config."""
    if cfg.get('hopfield_eval'):
        return 'hopfield'

    if config_path is not None:
        path_str = str(config_path).lower()
        if 'squad_cq_a' in path_str:
            return 'squad_cq_a'
        if 'squad_qc_a' in path_str:
            return 'squad_qc_a'
        if 'squad' in path_str:
            return 'squad'
        if 'babi' in path_str:
            return 'babi'
        if 'phonebook' in path_str:
            return 'phonebook'

    path_str = str(config_path or '').lower()
    if 'hopfield' in path_str and 'kv_retrieval' in path_str:
        return 'hopfield'

    dataset = cfg.get('dataset', {})
    data_name = dataset.get('data_name', '').lower()
    data_path = dataset.get('data_path', '').lower()

    if 'squad_cq_a' in data_name or 'squad_cq_a' in data_path:
        return 'squad_cq_a'
    if 'squad_qc_a' in data_name or 'squad_qc_a' in data_path:
        return 'squad_qc_a'
    if 'squad' in data_name or 'squad' in data_path:
        return 'squad'
    if 'babi' in data_name or 'babi' in data_path:
        return 'babi'
    if 'phonebook' in data_name or 'phonebook' in data_path:
        return 'phonebook'
    return 'kv_retrieval'


def get_script_for_model(model_type: str, dataset_type: str = 'kv_retrieval') -> str:
    """Get the appropriate script for the model type."""
    if dataset_type == 'hopfield' and model_type == 'gradmemgpt':
        return SCRIPT_MAP['gradmemgpt_hopfield']
    key = f"{model_type}_{dataset_type}" if dataset_type != 'kv_retrieval' else model_type
    return SCRIPT_MAP.get(key, SCRIPT_MAP.get(model_type, 'run_gpt2_on_kv_retrieval.py'))


def apply_overrides(cfg: dict, overrides: dict) -> dict:
    """Apply command-line overrides to config."""
    if not overrides:
        return cfg

    result = {**cfg}

    for key, value in overrides.items():
        applied = False
        for section in ['model', 'training', 'gradmem', 'rmt', 'hopfield', 'hopfield_eval', 'dataset']:
            if section in result and key in result[section]:
                result[section][key] = value
                applied = True
                break
        if not applied:
            if '.' in key:
                section, subkey = key.split('.', 1)
                if section not in result:
                    result[section] = {}
                result[section][subkey] = value
            else:
                result[key] = value

    return result


def build_cli_args(cfg: dict, overrides: dict = None) -> list[str]:
    """Build command-line arguments from config."""
    cfg = apply_overrides(cfg, overrides)

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
    grad_acc = max(1, tbs // (per_device * 1))

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
        args.append('--use_gradient_checkpointing')

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
    if 'run_name_suffix' in cfg:
        args.append(f'--run_name_suffix={cfg["run_name_suffix"]}')

    for key, val in gradmem.items():
        if val is None or val is False:
            continue
        cli_key = key.replace('_', '-')
        if val is True:
            args.append(f'--{cli_key}')
        else:
            args.append(f'--{cli_key}={val}')

    for key, val in rmt.items():
        if val is None or val is False:
            continue
        cli_key = key.replace('_', '-')
        if val is True:
            args.append(f'--{cli_key}')
        else:
            args.append(f'--{cli_key}={val}')

    hopfield = cfg.get('hopfield', {})
    for key, val in hopfield.items():
        if val is None or val is False:
            continue
        cli_key = key.replace('_', '-')
        if val is True:
            args.append(f'--{cli_key}')
        else:
            args.append(f'--{cli_key}={val}')

    hopfield_eval = cfg.get('hopfield_eval', {})
    for key, val in hopfield_eval.items():
        if val is None or val is False:
            continue
        cli_key = f"hopfield-eval-{key.replace('_', '-')}"
        if val is True:
            args.append(f'--{cli_key}')
        else:
            args.append(f'--{cli_key}={val}')

    return args


def main():
    parser = argparse.ArgumentParser(description='Run experiments from YAML configuration files')
    parser.add_argument('--config', type=Path, required=True, help='Path to YAML config')
    parser.add_argument('--model', type=str, choices=['gpt2', 'gradmemgpt', 'rmt'],
                      help='Override model type')
    parser.add_argument('--dataset', type=str,
                      choices=['kv_retrieval', 'squad', 'squad_cq_a', 'squad_qc_a'],
                      help='Override dataset type')
    parser.add_argument('--dry-run', action='store_true', help='Print command without executing')
    parser.add_argument('--debug', action='store_true', help='Run without accelerate wrapper (for debugging)')
    parser.add_argument('overrides', nargs='*', help='Additional overrides in key=value format')

    args = parser.parse_args()

    cfg = load_config(args.config)

    parse_overrides = {}
    for override in args.overrides:
        if '=' in override:
            key, val = override.split('=', 1)
            try:
                val = eval(val)
            except Exception:
                pass
            parse_overrides[key] = val

    dataset_type = args.dataset or detect_dataset_type(cfg, args.config)
    model_type = get_model_type(cfg)
    if args.model is not None:
        model_type = args.model

    # Compute run_name and exp_path after applying overrides
    if args.model is not None or parse_overrides:
        run_name = generate_run_name(apply_overrides(cfg, parse_overrides))
        exp_path = get_exp_path(apply_overrides(cfg, parse_overrides))
    else:
        run_name = generate_run_name(cfg)
        exp_path = get_exp_path(cfg)

    logger.info(f"Config: {args.config}")
    logger.info(f"Run name: {run_name}")
    logger.info(f"Exp path: {exp_path}")

    script = get_script_for_model(model_type, dataset_type)

    if args.model is not None or parse_overrides:
        run_name = generate_run_name(apply_overrides(cfg, parse_overrides))

    # Prepare config path for debug mode
    config_path = None
    if parse_overrides:
        import tempfile
        merged_cfg = apply_overrides(cfg, parse_overrides)
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as tmp:
            yaml.dump(merged_cfg, tmp)
            config_path = tmp.name

    if args.dry_run:
        if args.debug:
            if config_path:
                cmd = f"import {script.replace('.py', '')}; {script.replace('.py', '')}.main('{config_path}')"
            else:
                cmd = f"import {script.replace('.py', '')}; {script.replace('.py', '')}.main('{args.config}')"
        else:
            cmd = f"accelerate launch --config_file accelerate.yaml {script} --config {args.config}"
        print(f"\n# Run name: {run_name}")
        print(f"# Exp path: {exp_path}")
        print(f"# Command:\n{cmd}")
        return

    exp_path.mkdir(parents=True, exist_ok=True)
    cli_args = build_cli_args(cfg, parse_overrides)
    config = {
        'config_file': str(args.config),
        'run_name': run_name,
        'model_type': model_type,
        'dataset_type': dataset_type,
        'script': script,
        'cli_args': cli_args,
        'overrides': parse_overrides,
    }
    with open(exp_path / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)

    logger.info(f"Saved config to {exp_path / 'config.json'}")

    if args.debug:
        # Import and call main directly for debugging
        import importlib.util
        script_path = Path(script)
        module_name = script_path.stem

        spec = importlib.util.spec_from_file_location(module_name, script_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)

        actual_config_path = config_path if config_path else str(args.config)
        logger.info(f"\nRunning in debug mode: {module_name}.main('{actual_config_path}')")
        module.main(actual_config_path)
    else:
        command = f"accelerate launch --mixed_precision 'no' --config_file accelerate.yaml {script} --config {args.config}"
        logger.info(f"\nRunning:\n{command}")
        result = subprocess.run(command, shell=True)
        sys.exit(result.returncode)


if __name__ == '__main__':
    main()