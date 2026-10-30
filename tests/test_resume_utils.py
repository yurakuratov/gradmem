import json
from argparse import Namespace

import pytest
import torch

from resume_utils import restore_resume_args


@pytest.mark.parametrize("preserve_args, expected_max_steps", [
    ((), 100),
    (("max_steps",), 1000),
    (("max_steps", "save_last_checkpoint"), 1000),
])
def test_restore_resume_args_uses_saved_configuration(tmp_path, preserve_args, expected_max_steps):
    run_path = tmp_path / "run_1"
    checkpoint_path = run_path / "checkpoint-10"
    checkpoint_path.mkdir(parents=True)
    (run_path / "config.json").write_text(json.dumps({
        "cli_args": {
            "exp_path": str(run_path),
            "per_device_batch_size": 8,
            "inner_lr": 0.4,
            "max_steps": 100,
            "save_last_checkpoint": False,
            "init_checkpoint": "/old/init.safetensors",
        }
    }))

    args = Namespace(
        exp_path=str(run_path),
        resume_from_checkpoint=str(checkpoint_path),
        per_device_batch_size=64,
        inner_lr=5.0,
        max_steps=1000,
        save_last_checkpoint=True,
        init_checkpoint="/new/init.safetensors",
    )

    restore_resume_args(args, preserve_args=preserve_args)

    assert args.per_device_batch_size == 8
    assert args.inner_lr == 0.4
    assert args.max_steps == expected_max_steps
    assert args.save_last_checkpoint is ("save_last_checkpoint" in preserve_args)
    assert args.init_checkpoint is None
    assert args.resume_from_checkpoint == str(checkpoint_path.resolve())


@pytest.mark.parametrize("version", ["2.4.1", "2.5.1+cu124"])
def test_resume_rejects_unsafe_torch_before_reading_configuration(tmp_path, monkeypatch, version):
    checkpoint_path = tmp_path / "checkpoint-10"
    checkpoint_path.mkdir()
    args = Namespace(exp_path=str(tmp_path), resume_from_checkpoint=str(checkpoint_path))
    monkeypatch.setattr(torch, "__version__", version)

    with pytest.raises(ValueError, match=r"Resuming training requires PyTorch >= 2.6"):
        restore_resume_args(args)


def test_non_resume_does_not_require_new_torch(monkeypatch):
    monkeypatch.setattr(torch, "__version__", "2.4.1")
    args = Namespace(resume_from_checkpoint=None)
    assert restore_resume_args(args) is args
