"""Utilities for generating experiment run names from configuration."""

import os
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


def get_run_uid() -> str:
    """Return a unique run id, binding the local folder to the comet experiment.

    The id is generated once per process and is idempotent on repeated calls:

    - If ``COMET_EXPERIMENT_KEY`` is already set (e.g. by a launching wrapper),
      reuse it so the folder matches the comet experiment key exactly.
    - Else, if ``comet_ml`` is importable, generate a guid with
      ``comet_ml.generate_guid()`` (offline; no experiment is created) and export
      it as ``COMET_EXPERIMENT_KEY``. HF's ``CometCallback`` later calls
      ``comet_ml.start()``, which honours this env var in every mode
      (online/get_or_create/create) — verified empirically — so the comet
      experiment binds to the same key the folder postfix was derived from.
    - Else (no comet_ml installed), fall back to a random ``uuid4`` hex.
    """
    existing = os.getenv('COMET_EXPERIMENT_KEY')
    if existing:
        return existing
    try:
        import comet_ml
        uid = comet_ml.generate_guid()
        os.environ['COMET_EXPERIMENT_KEY'] = uid
        return uid
    except ImportError:
        return uuid.uuid4().hex


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

    if grad_mode == 'second' and last_K_second_order is not None and last_K_second_order != K:
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

    gated_delta = cfg.get('gated_delta', {})
    if gated_delta.get('use_gated_delta_memory'):
        d = gated_delta.get('gated_delta_state_dim', 128)
        run_name += f"_gd{d}"

    # Adaptive (segmented, gated-recurrence) fork. Encodes the cross-segment
    # carry + the gated memory update rule so runs are distinguishable:
    #   _seg<n>            n_segments (or seg<sz> if segment_size is used)
    #   _pa<N>             pair-aware segmentation (N = pairs_per_segment if set)
    #   _<rule>            memory_update_rule (omitted for "sgd" -> identical to base)
    #   _<gate_features>/<granularity>/<retention>  gate config (only for gated rules)
    #   _bptt<n>           seg_bptt truncated-BPTT window (only if set)
    # Segmentation knobs live in the unified `segmentation:` section (with the
    # old `adaptive:` placement as fallback for un-migrated configs).
    adaptive = cfg.get('adaptive', {})
    segmentation = cfg.get('segmentation', {})
    if adaptive or segmentation:
        seg_size = segmentation.get('segment_size', adaptive.get('segment_size'))
        n_seg = segmentation.get('n_segments', adaptive.get('n_segments', 1))
        rule = adaptive.get('memory_update_rule', 'sgd')
        run_name += f"_seg{seg_size}" if seg_size else (f"_seg{n_seg}" if n_seg and n_seg != 1 else "")
        if segmentation.get('pair_aware_segmentation') or adaptive.get('pair_aware_segmentation'):
            pp = segmentation.get('pairs_per_segment', adaptive.get('pairs_per_segment'))
            run_name += f"_pa{pp}" if pp else "_pa"
        if rule and rule != 'sgd':
            run_name += f"_{rule}"
            feats = adaptive.get('gate_features', 'grad_state')
            gran = adaptive.get('gate_granularity', 'per_dim')
            retain = adaptive.get('gate_retention', 'exp')
            # only append non-defaults to keep the name readable
            if feats != 'grad_state':
                run_name += f"_f{feats}"
            if gran != 'per_dim':
                run_name += f"_{gran}"
            if rule == 'mamba' and retain != 'exp':
                run_name += f"_r{retain}"
        bptt = adaptive.get('seg_bptt')
        if bptt is not None:
            run_name += f"_bptt{bptt}"

    curriculum = cfg.get('curriculum', {})
    if curriculum.get('enabled'):
        threshold = curriculum.get('threshold', 0.95)
        levels = curriculum.get('levels', '4,8,16,32,64,128')
        run_name += f"_curr_t{threshold}_N{levels}"

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


# Sections searched when a bare (un-dotted) key is given to the generic builder.
_KNOWN_SECTIONS = ['model', 'training', 'dataset', 'gradmem', 'rmt', 'hopfield',
                   'gated_delta', 'curriculum', 'adaptive', 'segmentation']


def _resolve_key(cfg: Dict[str, Any], key: str):
    """Resolve a config key (dotted or bare) to its value, or None if absent.

    Dotted keys (e.g. ``training.learning_rate``) walk the nested config. Bare
    keys are looked up top-level first, then under each known section.
    """
    if '.' in key:
        cur: Any = cfg
        for part in key.split('.'):
            if not isinstance(cur, dict) or part not in cur:
                return None
            cur = cur[part]
        return cur
    if key in cfg:
        return cfg[key]
    for section in _KNOWN_SECTIONS:
        sec = cfg.get(section)
        if isinstance(sec, dict) and key in sec:
            return sec[key]
    return None


def _format_value(key: str, value: Any, prefix: Optional[str]) -> Optional[str]:
    """Render a single key/value pair as ``<prefix><value>``, or None to skip.

    - ``None`` -> skip.
    - ``True`` -> the prefix alone (a flag), mirroring the existing ``_whead``
      / ``_mem_proj`` style.
    - ``False`` -> skip (negative flags don't add information to a name).
    - otherwise -> ``str(value)``.
    """
    if value is None or value is False:
        return None
    label = prefix if prefix is not None else key
    if value is True:
        return label
    return f"{label}{value}"


def generate_run_name_from_keys(cfg: Dict[str, Any], keys: list,
                                separator: str = '_') -> str:
    """Build a run name from a list of tracked config keys.

    Each entry in ``keys`` is either a plain string (``"training.learning_rate"``,
    in which case the prefix defaults to the key itself) or a dict with ``key``
    and an optional ``prefix`` (e.g. ``{key: training.learning_rate, prefix: lr}``
    renders as ``lr1e-4``). Missing/None/False values are skipped. A top-level
    ``run_name_suffix`` (if set) is appended for parity with the other builders.
    """
    parts = []
    for entry in keys:
        if isinstance(entry, dict):
            key = entry['key']
            prefix = entry.get('prefix')
        else:
            key = entry
            prefix = None
        value = _resolve_key(cfg, key)
        rendered = _format_value(key, value, prefix if prefix is not None else key)
        if rendered is not None:
            parts.append(rendered)

    run_name = separator.join(parts)

    suffix = cfg.get('run_name_suffix')
    if suffix:
        run_name = f"{run_name}{separator}{suffix}" if run_name else str(suffix)

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
    """Generate run name based on config.

    If a ``run_naming`` section with a ``keys`` list is present, the name is
    built generically from those keys (fully replacing the per-model logic).
    Otherwise fall back to the per-model-type generators.
    """
    naming = cfg.get('run_naming', {})
    if naming.get('keys'):
        return generate_run_name_from_keys(cfg, naming['keys'],
                                           naming.get('separator', '_'))

    model_type = get_model_type(cfg)

    if model_type == 'gradmemgpt':
        return generate_run_name_gradmemgpt(cfg)
    elif model_type == 'rmt':
        return generate_run_name_rmt(cfg)
    else:
        return generate_run_name_gpt2(cfg)


def get_exp_path(cfg: Dict[str, Any], seed: Optional[int] = None) -> Path:
    """Generate experiment path from config.

    Layout: ``<runs_dir>/<data_name>/<run_name>/run_<seed>_<uid8>``

    - ``runs_dir`` (top-level config key, default ``./runs``) is the output root.
    - ``run_name`` (top-level config key), if set, fully replaces the auto-generated
      hyperparameter name; otherwise it is auto-generated by ``generate_run_name``.
    - ``run_<seed>`` keeps the existing seed component (``training.seed`` or ``seed``).
    - ``<uid8>`` is the first 8 chars of a process-unique id (see ``get_run_uid``)
      so two runs with identical config+seed no longer overwrite each other. When
      comet is used, the uid is the comet experiment key, tying the folder to the
      comet run.
    """
    dataset = cfg.get('dataset', {})
    training = cfg.get('training', {})
    curriculum = cfg.get('curriculum', {})

    if curriculum.get('enabled'):
        levels = curriculum.get('levels', '4,8,16,32,64,128')
        data_name = f"curriculum_N{levels}"
    else:
        data_name = dataset.get('data_name', 'default')

    # Manual run_name override, else auto-generate from hyperparameters.
    run_name = cfg.get('run_name') or generate_run_name(cfg)

    s = seed if seed is not None else training.get('seed', 1)

    runs_dir = cfg.get('runs_dir', './runs')
    uid8 = get_run_uid()[:8]

    exp_path = Path(runs_dir) / data_name / run_name / f'run_{s}_{uid8}'
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
    if training.get('stop_on_em_threshold') is not None:
        args.append(f'--stop_on_em_threshold={training["stop_on_em_threshold"]}')
    if training.get('use_gradient_checkpointing'):
        args.append(f'--use_gradient_checkpointing')
    if training.get('auto_find_batch_size'):
        args.append('--auto_find_batch_size')

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