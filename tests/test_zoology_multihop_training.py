import json
import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from transformers import HfArgumentParser

from energy_gradmem import EnergyGradMem, EnergyGradMemConfig
from run_energy_gradmem_on_zoology_multihop import (
    ZoologyExperimentArgs,
    ZoologyTrainer,
    build_zoology_base_config,
    collate_zoology_batch,
    compute_zoology_metrics,
    progression_metrics_for_json,
    query_target_label_positions,
    save_progression_metrics,
    segment_context_input_ids,
    split_dataset,
    validate_zoology_args,
)
from zoology_curriculum_gate import evaluate_curriculum_gate
from zoology_multihop_dataset import (
    BOS_ID,
    CONTEXT_END_ID,
    CONTEXT_START_ID,
    EOS_ID,
    KV_RECORD_WIDTH,
    generate_sample,
    parse_context_records,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    ("n_pairs", "pairs_per_segment", "expected_segments"),
    [
        (8, 8, 1),
        (16, 8, 2),
        (128, 8, 16),
        (8, 32, 1),
        (64, 32, 2),
        (128, 32, 4),
        (8, 8, 1),
        (128, 128, 1),
    ],
)
def test_context_segmentation_counts_kv_pairs_without_splitting_records(
    n_pairs,
    pairs_per_segment,
    expected_segments,
):
    sample = generate_sample(n_pairs, 1, seed=11)
    segments = segment_context_input_ids(
        sample["context_input_ids"],
        n_pairs=n_pairs,
        kv_pairs_per_segment=pairs_per_segment,
    )

    assert len(segments) == expected_segments
    assert segments[0][:2] == [BOS_ID, CONTEXT_START_ID]
    assert segments[-1][-2:] == [CONTEXT_END_ID, EOS_ID]
    record_token_lengths = [
        len(segment)
        - (2 if index == 0 else 0)
        - (2 if index == len(segments) - 1 else 0)
        for index, segment in enumerate(segments)
    ]
    assert all(length % KV_RECORD_WIDTH == 0 for length in record_token_lengths)

    reconstructed = []
    for index, segment in enumerate(segments):
        start = 2 if index == 0 else 0
        stop = -2 if index == len(segments) - 1 else None
        reconstructed.extend(segment[start:stop])
    flat_context = [
        BOS_ID,
        CONTEXT_START_ID,
        *reconstructed,
        CONTEXT_END_ID,
        EOS_ID,
    ]
    assert parse_context_records(flat_context) == parse_context_records(
        sample["context_input_ids"]
    )


def test_all_regime_passes_the_complete_context_as_one_segment():
    sample = generate_sample(32, 1, seed=13)
    segments = segment_context_input_ids(
        sample["context_input_ids"],
        n_pairs=32,
        kv_pairs_per_segment=32,
    )
    assert segments == [sample["context_input_ids"]]


def test_collator_supervises_every_query_without_feeding_targets():
    samples = [generate_sample(8, 1, seed=seed) for seed in (17, 19)]
    collated = collate_zoology_batch(
        samples,
        n_pairs=8,
        kv_pairs_per_segment=8,
    )

    query_input_ids = collated["input_ids"]["query_input_ids"]
    labels = collated["labels"]
    positions = query_target_label_positions(8)
    assert labels.ne(-100).sum(dim=1).tolist() == [8, 8]
    for row, sample in enumerate(samples):
        assert labels[row, positions].tolist() == sample["targets"]
        assert not set(sample["targets"]).intersection(query_input_ids[row].tolist())
        assert labels[row, 0].item() == -100
    assert torch.equal(
        collated["input_ids"]["hop_distances"],
        torch.tensor([sample["hop_distances"] for sample in samples]),
    )


def test_metrics_report_query_all_queries_exact_match_and_hop_groups():
    labels = np.array(
        [
            [-100, 10, -100, 20, -100],
            [-100, 30, -100, 40, -100],
        ]
    )
    predictions = np.array(
        [
            [0, 10, 0, 20, 0, 0],
            [0, 99, 0, 40, 0, 0],
        ]
    )
    eval_pred = SimpleNamespace(
        predictions=(predictions, {"inner_loss": np.array([1.0, 3.0])}),
        label_ids=labels,
        inputs={"hop_distances": np.array([[1, 2], [1, 2]])},
    )

    metrics = compute_zoology_metrics(eval_pred)

    assert metrics["query_accuracy"] == pytest.approx(0.75)
    assert metrics["all_queries_exact_match"] == pytest.approx(0.5)
    assert "exact_match" not in metrics
    assert metrics["accuracy_hop_distance_1"] == pytest.approx(0.5)
    assert metrics["accuracy_hop_distance_2"] == pytest.approx(1.0)
    assert metrics["inner_loss"] == pytest.approx(2.0)


def test_zoology_model_defaults_match_first_experiment():
    args = ZoologyExperimentArgs(exp_path="unused", per_device_batch_size=2)
    validate_zoology_args(args)
    config = build_zoology_base_config(args)

    assert args.hf_subset == "N8-H1-V4096"
    assert config.vocab_size == 4096
    assert config.num_hidden_layers == 2
    assert config.hidden_size == 128
    assert config.num_attention_heads == 1
    assert config.attention_dropout == pytest.approx(0.1)
    assert args.n_mem_tokens == 4
    assert args.energy_model_type == "segment_delta_gru"
    assert args.energy_replay_weight == 0.0
    assert args.reading_optimization is False
    assert args.outer_lr_final_ratio == pytest.approx(0.2)


def test_outer_lr_scheduler_warms_up_then_decays_to_exact_ratio():
    parameter = torch.nn.Parameter(torch.tensor(0.0))
    optimizer = torch.optim.SGD([parameter], lr=2.0)
    trainer = ZoologyTrainer.__new__(ZoologyTrainer)
    trainer.args = SimpleNamespace(
        lr_scheduler_type="linear",
        get_warmup_steps=lambda _: 4,
    )
    trainer.optimizer = optimizer
    trainer.lr_scheduler = None
    trainer.outer_lr_final_ratio = 0.2
    scheduler = trainer.create_scheduler(num_training_steps=12)

    lrs = {0: scheduler.get_last_lr()[0]}
    for step in range(1, 13):
        optimizer.step()
        scheduler.step()
        lrs[step] = scheduler.get_last_lr()[0]

    assert lrs[0] == pytest.approx(0.0)
    assert lrs[4] == pytest.approx(2.0)
    assert lrs[8] == pytest.approx(1.2)
    assert lrs[12] == pytest.approx(0.4)


def test_zoology_cli_uses_unambiguous_all_queries_stop_metric_name():
    parser = HfArgumentParser(ZoologyExperimentArgs)
    destinations = {action.dest for action in parser._actions}
    assert "stop_all_queries_exact_match_value" in destinations
    assert "stop_exact_match_value" not in destinations


def test_attention_dropout_is_configurable_and_validated():
    args = ZoologyExperimentArgs(
        exp_path="unused",
        per_device_batch_size=2,
        attention_dropout=0.25,
    )
    validate_zoology_args(args)
    assert build_zoology_base_config(args).attention_dropout == pytest.approx(0.25)

    for invalid_dropout in (-0.1, 1.0):
        invalid_args = ZoologyExperimentArgs(
            exp_path="unused",
            per_device_batch_size=2,
            attention_dropout=invalid_dropout,
        )
        with pytest.raises(ValueError, match="attention_dropout"):
            validate_zoology_args(invalid_args)


def test_dataset_split_keeps_validation_for_selection_and_test_for_reporting():
    dataset = {
        "train": object(),
        "validation": object(),
        "test": object(),
    }
    train, validation, test = split_dataset(dataset)
    assert train is dataset["train"]
    assert validation is dataset["validation"]
    assert test is dataset["test"]

    with pytest.raises(ValueError, match="held-out test"):
        split_dataset({"train": object(), "validation": object()})


@pytest.mark.parametrize(
    ("script_name", "regime"),
    [
        ("run_energy_gradmem_curriculum_seg8.sh", "8"),
        ("run_energy_gradmem_curriculum_seg32.sh", "32"),
        ("run_energy_gradmem_curriculum_all.sh", "all"),
    ],
)
def test_curriculum_entry_points_define_aligned_schedules(script_name, regime):
    script = (REPO_ROOT / "scripts" / "zoology_mh" / script_name).read_text()
    assert f"SEGMENT_REGIME={regime}" in script
    schedules = [
        re.search(rf"{name}=\(([^)]*)\)", script).group(1).split()
        for name in ("CURRICULUM_N", "CE_WEIGHTS", "INNER_LRS")
    ]
    assert len(schedules[0]) == len(schedules[1]) == len(schedules[2]) == 6


def test_common_curriculum_explicitly_disables_replay_and_read_optimization():
    helper = (
        REPO_ROOT
        / "scripts"
        / "zoology_mh"
        / "run_energy_gradmem_curriculum_common.sh"
    ).read_text()
    assert "ENERGY_REPLAY_WEIGHT=0.0" in helper
    assert "READING_OPTIMIZATION=false" in helper
    assert "ENERGY_MODEL_TYPE=${ENERGY_MODEL_TYPE:-segment_delta_gru}" in helper
    assert "L=${L:-2}" in helper
    assert "N_HEAD=${N_HEAD:-1}" in helper
    assert "D=${D:-128}" in helper
    assert "N_MEM_TOKENS=${N_MEM_TOKENS:-4}" in helper
    assert "ATTENTION_DROPOUT=${ATTENTION_DROPOUT:-0.1}" in helper
    assert "LR_SCHEDULER_TYPE=${LR_SCHEDULER_TYPE:-linear}" in helper
    assert "OUTER_LR_FINAL_RATIO=${OUTER_LR_FINAL_RATIO:-0.2}" in helper
    assert '--outer_lr_final_ratio "$OUTER_LR_FINAL_RATIO"' in helper
    assert "ENABLE_CURRICULUM_GATE=${ENABLE_CURRICULUM_GATE:-true}" in helper
    assert "CURRICULUM_METRIC=${CURRICULUM_METRIC:-all_queries_exact_match}" in helper
    assert "CURRICULUM_METRIC_THRESHOLD=${CURRICULUM_METRIC_THRESHOLD:-0.9}" in helper
    assert "CURRICULUM_LOWER_IS_BETTER=${CURRICULUM_LOWER_IS_BETTER:-false}" in helper
    runner = (REPO_ROOT / "run_energy_gradmem_on_zoology_multihop.py").read_text()
    assert 'metric_name="all_queries_exact_match"' in runner
    assert "value=args.stop_all_queries_exact_match_value" in runner
    assert "--stop_all_queries_exact_match_value" in helper


def _write_gate_metrics(tmp_path, metrics):
    path = tmp_path / "progression_metrics.json"
    path.write_text(json.dumps(metrics), encoding="utf-8")
    return path


@pytest.mark.parametrize(
    ("value", "threshold", "lower_is_better", "expected"),
    [
        (0.8, 0.8, False, True),
        (0.9, 0.8, False, True),
        (0.7, 0.8, False, False),
        (0.2, 0.2, True, True),
        (0.1, 0.2, True, True),
        (0.3, 0.2, True, False),
    ],
)
def test_curriculum_gate_supports_both_comparison_directions(
    tmp_path,
    value,
    threshold,
    lower_is_better,
    expected,
):
    metrics_path = _write_gate_metrics(
        tmp_path,
        {"all_queries_exact_match": value},
    )
    passed, diagnostic = evaluate_curriculum_gate(
        metrics_path,
        "all_queries_exact_match",
        threshold,
        lower_is_better=lower_is_better,
    )
    assert passed is expected
    assert ("passed" if expected else "failed") in diagnostic


def test_disabled_curriculum_gate_does_not_require_metrics_file(tmp_path):
    passed, diagnostic = evaluate_curriculum_gate(
        tmp_path / "missing.json",
        "all_queries_exact_match",
        0.9,
        lower_is_better=False,
        enabled=False,
    )
    assert passed is True
    assert "disabled" in diagnostic.lower()


def test_curriculum_gate_rejects_missing_and_non_finite_metrics(tmp_path):
    passed, diagnostic = evaluate_curriculum_gate(
        tmp_path / "missing.json",
        "all_queries_exact_match",
        0.9,
        lower_is_better=False,
    )
    assert passed is False
    assert "file is missing" in diagnostic

    missing_metric_path = _write_gate_metrics(tmp_path, {"query_accuracy": 1.0})
    passed, diagnostic = evaluate_curriculum_gate(
        missing_metric_path,
        "all_queries_exact_match",
        0.9,
        lower_is_better=False,
    )
    assert passed is False
    assert "is missing" in diagnostic

    non_finite_path = _write_gate_metrics(
        tmp_path,
        {"all_queries_exact_match": float("nan")},
    )
    passed, diagnostic = evaluate_curriculum_gate(
        non_finite_path,
        "all_queries_exact_match",
        0.9,
        lower_is_better=False,
    )
    assert passed is False
    assert "must be finite" in diagnostic


def test_progression_metrics_json_uses_unprefixed_renamed_metric(tmp_path):
    raw_metrics = {
        "progression_loss": 0.25,
        "progression_query_accuracy": 0.75,
        "progression_all_queries_exact_match": 0.5,
        "progression_non_finite_diagnostic": float("inf"),
    }
    assert progression_metrics_for_json(raw_metrics) == {
        "loss": 0.25,
        "query_accuracy": 0.75,
        "all_queries_exact_match": 0.5,
        "non_finite_diagnostic": None,
    }

    metrics_path = save_progression_metrics(tmp_path, raw_metrics)
    saved = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert saved["all_queries_exact_match"] == pytest.approx(0.5)
    assert "exact_match" not in saved


@pytest.mark.forward
def test_integer_batch_runs_through_segment_delta_energy_without_target_leak():
    samples = [generate_sample(8, 1, seed=seed) for seed in (23, 29)]
    batch = collate_zoology_batch(
        samples,
        n_pairs=8,
        kv_pairs_per_segment=8,
    )
    base_config = build_zoology_base_config(
        ZoologyExperimentArgs(
            exp_path="unused",
            per_device_batch_size=2,
            n_layer=1,
            n_embd=32,
            n_head=1,
        )
    )
    model = EnergyGradMem(
        EnergyGradMemConfig(
            base_config=base_config,
            memory_backend="prefix",
            n_mem_tokens=2,
            K=1,
            last_K_second_order=1,
            lr=0.1,
            grad_mode="second",
            inner_objective="neural",
            energy_future_mode="none",
            energy_model_type="segment_delta_gru",
            energy_hidden_size=32,
            energy_segment_state_size=32,
            energy_replay_weight=0.0,
            reading_optimization=False,
            segment_write_mode="sequential",
        )
    )

    output = model(**batch)

    assert output["predictions"].shape == (
        2,
        len(samples[0]["query_input_ids"]) + 1,
        4096,
    )
    assert output["loss"].ndim == 0
    assert torch.isfinite(output["loss"])
    output["loss"].backward()
    assert torch.isfinite(model.token_energy_mlp[0].weight.grad).all()
