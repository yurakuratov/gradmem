import xml.etree.ElementTree as ET
import zipfile

import numpy as np
import torch
from transformers import GPT2Config

from evaluate_kv_checkpoints import (
    best_checkpoint_for_run,
    discover_run_directories,
    local_lipschitz_trajectory_batch,
    summarize_local_lipschitz,
    write_xlsx,
)
from grad_memgpt import GradMemGPT, GradMemGPTConfig


def test_write_xlsx_writes_metrics_sheet(tmp_path):
    output_path = tmp_path / "metrics.xlsx"
    write_xlsx(output_path, [{"checkpoint": "run/checkpoint-1", "token_accuracy": 0.75}])

    with zipfile.ZipFile(output_path) as archive:
        assert "xl/worksheets/sheet1.xml" in archive.namelist()
        sheet = ET.fromstring(archive.read("xl/worksheets/sheet1.xml"))

    namespace = {"xlsx": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    assert len(sheet.findall(".//xlsx:row", namespace)) == 2


def test_run_directory_selects_recorded_best_checkpoint(tmp_path):
    run_path = tmp_path / "run_1"
    run_path.mkdir()
    (run_path / "config.json").write_text('{"cli_args": {"metric_for_best_model": "exact_match"}}')
    for step in (10, 20):
        checkpoint = run_path / f"checkpoint-{step}"
        checkpoint.mkdir()
        (checkpoint / "model.safetensors").touch()
    (run_path / "trainer_state.json").write_text(
        '{"best_model_checkpoint": "checkpoint-20", "best_metric": 0.8}'
    )

    assert discover_run_directories(tmp_path) == [run_path]
    checkpoint, metric_name, metric_value = best_checkpoint_for_run(run_path)
    assert checkpoint == run_path / "checkpoint-20" / "model.safetensors"
    assert metric_name == "exact_match"
    assert metric_value == 0.8


def test_summarize_local_lipschitz_reports_each_trajectory_step():
    metrics = summarize_local_lipschitz(
        [
            [np.asarray([1.0, 3.0]), np.asarray([5.0])],
            [np.asarray([2.0, 4.0, 6.0])],
        ],
        [[np.asarray([0.5, 1.5, 2.5])]],
    )

    assert metrics["lipschitz_grad_k0_mean"] == 3.0
    assert metrics["lipschitz_grad_k0_median"] == 3.0
    assert metrics["lipschitz_grad_k1_max"] == 6.0
    assert metrics["lipschitz_secant_k1_mean"] == 1.5


def test_local_lipschitz_replays_each_write_step():
    torch.manual_seed(0)
    model = GradMemGPT(GradMemGPTConfig(
        base_config=GPT2Config(
            vocab_size=31,
            n_embd=16,
            n_layer=1,
            n_head=2,
            n_positions=16,
            n_ctx=16,
            pad_token_id=0,
            eos_token_id=1,
        ),
        n_mem_tokens=2,
        n_ctrl_tokens=1,
        K=2,
        lr=0.01,
        use_adam=False,
        grad_mode="none",
        mem_proj_mode="none",
        attn_implementation="eager",
    ))
    model.eval()
    input_ids = {
        "context_input_ids": torch.tensor([[2, 3, 4, 0], [5, 6, 7, 8]]),
        "query_input_ids": torch.tensor([[9, 10], [11, 12]]),
    }

    gradient_norms, secants = local_lipschitz_trajectory_batch(model, input_ids)

    assert len(gradient_norms) == model.K + 1
    assert len(secants) == model.K
    assert all(values.shape == (2,) and np.isfinite(values).all() for values in gradient_norms)
    assert all(values.shape == (2,) and np.isfinite(values).all() for values in secants)
