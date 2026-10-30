import os
from pathlib import Path
import subprocess

import pytest


@pytest.mark.parametrize("active_env", [None, "base"])
def test_launcher_reexecutes_in_rmt_and_preserves_resume_argument(tmp_path, active_env):
    conda = tmp_path / "conda"
    conda.write_text('#!/bin/bash\nprintf "%s\\n" "$@"\n')
    conda.chmod(0o755)
    env = os.environ.copy()
    env["PATH"] = f"{tmp_path}:{env['PATH']}"
    if active_env is None:
        env.pop("CONDA_DEFAULT_ENV", None)
    else:
        env["CONDA_DEFAULT_ENV"] = active_env
    launcher = Path(__file__).resolve().parents[1] / "scripts/run_energygradmem_on_mqar.sh"
    result = subprocess.run(
        ["bash", str(launcher), "--resume_from_checkpoint", "/a checkpoint/checkpoint-10"],
        env=env, capture_output=True, text=True, check=True,
    )
    assert result.stdout.splitlines() == [
        "run", "--no-capture-output", "-n", "rmt", "bash", str(launcher),
        "--resume_from_checkpoint", "/a checkpoint/checkpoint-10",
    ]
