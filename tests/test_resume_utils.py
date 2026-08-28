import json
from argparse import Namespace

from resume_utils import restore_resume_args


def test_restore_resume_args_uses_saved_configuration(tmp_path):
    run_path = tmp_path / "run_1"
    checkpoint_path = run_path / "checkpoint-10"
    checkpoint_path.mkdir(parents=True)
    (run_path / "config.json").write_text(json.dumps({
        "cli_args": {
            "exp_path": str(run_path),
            "per_device_batch_size": 8,
            "inner_lr": 0.4,
            "max_steps": 100,
            "init_checkpoint": "/old/init.safetensors",
        }
    }))

    args = Namespace(
        exp_path=str(run_path),
        resume_from_checkpoint=str(checkpoint_path),
        per_device_batch_size=64,
        inner_lr=5.0,
        max_steps=1000,
        init_checkpoint="/new/init.safetensors",
    )

    restore_resume_args(args)

    assert args.per_device_batch_size == 8
    assert args.inner_lr == 0.4
    assert args.max_steps == 100
    assert args.init_checkpoint is None
    assert args.resume_from_checkpoint == str(checkpoint_path.resolve())
