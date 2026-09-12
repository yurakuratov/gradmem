from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from energy_gradmem import EnergyGradMem, EnergyGradMemConfig
from run_energy_gradmem_on_zoology_multihop import (
    ZoologyExperimentArgs,
    build_zoology_base_config,
    collate_zoology_batch,
    compute_zoology_metrics,
    query_target_label_positions,
    segment_context_input_ids,
    split_dataset,
    validate_zoology_args,
)
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


def test_metrics_report_query_exact_match_and_hop_groups():
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
    assert metrics["exact_match"] == pytest.approx(0.5)
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
    ("script_name", "regime", "n_values", "ce_weights", "inner_lrs"),
    [
        (
            "run_energy_gradmem_curriculum_seg8.sh",
            "8",
            "8 16 16 32 64 128",
            "1 1 0 0 0 0",
            "1 0.1 0.1 0.1 0.03 0.03",
        ),
        (
            "run_energy_gradmem_curriculum_seg32.sh",
            "32",
            "8 16 32 64 64 128",
            "1 1 1 1 0 0",
            "1 1 1 0.1 0.1 0.1",
        ),
        (
            "run_energy_gradmem_curriculum_all.sh",
            "all",
            "8 8 16 32 64 128",
            "1 0 0 0 0 0",
            "1 1 1 1 1 1",
        ),
    ],
)
def test_curriculum_entry_points_encode_requested_schedules(
    script_name,
    regime,
    n_values,
    ce_weights,
    inner_lrs,
):
    script = (REPO_ROOT / "scripts" / "zoology_mh" / script_name).read_text()
    assert f"SEGMENT_REGIME={regime}" in script
    assert f"CURRICULUM_N=({n_values})" in script
    assert f"CE_WEIGHTS=({ce_weights})" in script
    assert f"INNER_LRS=({inner_lrs})" in script


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
