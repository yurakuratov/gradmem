"""Utilities for resuming experiments without changing their configuration."""

import json
from pathlib import Path
from typing import Any


def restore_resume_args(args: Any, logger: Any = None, *, preserve_args: tuple[str, ...] = ()) -> Any:
    """Restore runner arguments saved in the experiment directory.

    A model checkpoint contains model, optimizer, scheduler, RNG, and trainer
    state, but it does not contain the task and GradMem runner arguments.  The
    latter must be restored before constructing the dataset and model.
    Names in preserve_args retain their current CLI values instead.
    """
    resume_value = getattr(args, "resume_from_checkpoint", None)
    if resume_value is None:
        return args

    resume_path = Path(resume_value).expanduser().resolve()
    output_path = Path(args.exp_path).expanduser().resolve()
    if not resume_path.is_dir():
        raise ValueError(f"Resume checkpoint directory does not exist: {resume_path}")
    if resume_path.parent != output_path:
        raise ValueError(
            "--resume_from_checkpoint must be a checkpoint directly inside --exp_path so logs continue "
            f"in the same run: checkpoint={resume_path}, exp_path={output_path}"
        )

    import sys

    import torch
    from packaging.version import Version

    if Version(torch.__version__).release < (2, 6):
        raise ValueError(
            f"Resuming training requires PyTorch >= 2.6; found {torch.__version__} "
            f"in {sys.executable}. Trainer restores optimizer and scheduler .pt files "
            "even when model weights use safetensors. Use the rmt environment "
            "(conda run --no-capture-output -n rmt bash scripts/run_energygradmem_on_mqar.sh "
            "--resume_from_checkpoint <checkpoint-directory>), or upgrade PyTorch "
            "in your project environment. Do not bypass the torch.load security check."
        )

    config_path = output_path / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(
            f"Cannot resume without the original experiment configuration: {config_path}"
        )
    with config_path.open() as config_file:
        experiment_config = json.load(config_file)
    saved_args = experiment_config.get("cli_args")
    if not isinstance(saved_args, dict):
        raise ValueError(f"Experiment configuration has no valid cli_args: {config_path}")

    current_args = vars(args).copy()
    restored_keys = []
    for name, value in saved_args.items():
        if (
            name not in current_args
            or name in {"exp_path", "resume_from_checkpoint", "init_checkpoint"}
            or name in preserve_args
        ):
            continue
        if current_args[name] != value:
            restored_keys.append(name)
        setattr(args, name, value)

    # Initialization checkpoints are only used before the first training step.
    # Loading one during resume is both unnecessary and incompatible with the
    # normal mutual-exclusion check below.
    args.init_checkpoint = None
    args.exp_path = str(output_path)
    args.resume_from_checkpoint = str(resume_path)

    if logger is not None:
        logger.info(f"Restored experiment arguments from {config_path}")
        if restored_keys:
            logger.warning(
                "Ignoring current CLI values while resuming; restored: "
                + ", ".join(sorted(restored_keys))
            )
    return args
