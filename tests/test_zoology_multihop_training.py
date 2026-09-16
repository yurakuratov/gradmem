import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from energy_gradmem import EnergyGradMem, EnergyGradMemConfig
from run_energy_gradmem_on_ar_multihop import (
    ARMultihopExperimentArgs,
    build_ar_multihop_base_config,
    collate_multihop_batch,
    compute_ar_multihop_metrics,
    segment_zoology_context_input_ids,
    validate_ar_multihop_args,
    zoology_query_target_label_positions,
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


def _zoology_args(**overrides):
    values = {
        "exp_path": "unused",
        "per_device_batch_size": 2,
        "dataset_format": "zoology",
    }
    values.update(overrides)
    return ARMultihopExperimentArgs(**values)


def test_zoology_argument_defaults_and_model_vocabulary_are_inferred():
    args = _zoology_args()
    assert args.hf_dataset is None
    assert args.hf_subset is None
    assert args.vocab_size is None

    validate_ar_multihop_args(args)
    config = build_ar_multihop_base_config(args)
    assert args.hf_dataset == "irodkin/zoology_multihop"
    assert args.hf_subset == "N8-H1-V4096"
    assert args.vocab_size == config.vocab_size == 4096


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("hf_dataset", "irodkin/ar_multihop"),
        ("hf_subset", "N8-H1-V16"),
        ("vocab_size", 27),
    ],
)
def test_zoology_rejects_dataset_defaults_from_the_other_format(field_name, value):
    args = _zoology_args(**{field_name: value})
    with pytest.raises(ValueError, match=field_name):
        validate_ar_multihop_args(args)


def test_zoology_context_segmentation_and_collation_preserve_preformatted_rows():
    sample = generate_sample(16, 1, seed=11)
    segments = segment_zoology_context_input_ids(
        sample["context_input_ids"],
        n_pairs=16,
        kv_pairs_per_segment=8,
    )
    assert len(segments) == 2
    assert segments[0][:2] == [BOS_ID, CONTEXT_START_ID]
    assert segments[-1][-2:] == [CONTEXT_END_ID, EOS_ID]
    assert (len(segments[0]) - 2) % KV_RECORD_WIDTH == 0
    assert (len(segments[1]) - 2) % KV_RECORD_WIDTH == 0

    reconstructed = [BOS_ID, CONTEXT_START_ID, *segments[0][2:], *segments[1][:-2], CONTEXT_END_ID, EOS_ID]
    assert parse_context_records(reconstructed) == parse_context_records(
        sample["context_input_ids"]
    )

    batch = collate_multihop_batch(
        [sample],
        dataset_format="zoology",
        n_pairs=16,
        hop_length=1,
        kv_pairs_per_segment=8,
    )
    positions = zoology_query_target_label_positions(16)
    assert batch["labels"].ne(-100).sum().item() == 16
    assert batch["labels"][0, positions].tolist() == sample["targets"]
    assert batch["input_ids"]["query_input_ids"][0].tolist() == sample[
        "query_input_ids"
    ]


def test_generalized_metrics_treat_zoology_as_target_width_one():
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
    metrics = compute_ar_multihop_metrics(
        SimpleNamespace(
            predictions=(predictions, {}),
            label_ids=labels,
            inputs={"hop_distances": np.array([[1, 2], [1, 2]])},
        )
    )
    assert metrics["token_accuracy"] == pytest.approx(0.75)
    assert metrics["query_accuracy"] == pytest.approx(0.75)
    assert metrics["all_queries_exact_match"] == pytest.approx(0.5)
    assert metrics["accuracy_hop_distance_1"] == pytest.approx(0.5)
    assert metrics["accuracy_hop_distance_2"] == pytest.approx(1.0)


@pytest.mark.forward
def test_zoology_batch_runs_end_to_end_forward_and_backward():
    samples = [generate_sample(8, 1, seed=seed) for seed in (23, 29)]
    batch = collate_multihop_batch(
        samples,
        dataset_format="zoology",
        n_pairs=8,
        hop_length=1,
        kv_pairs_per_segment=8,
    )
    args = _zoology_args(n_layer=1, n_embd=32, n_head=1)
    validate_ar_multihop_args(args)
    model = EnergyGradMem(
        EnergyGradMemConfig(
            base_config=build_ar_multihop_base_config(args),
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
    assert torch.isfinite(output["loss"])
    output["loss"].backward()
    assert torch.isfinite(model.token_energy_mlp[0].weight.grad).all()


@pytest.mark.parametrize(
    ("script_name", "regime"),
    [
        ("run_energy_gradmem_curriculum_seg8.sh", "8"),
        ("run_energy_gradmem_curriculum_seg32.sh", "32"),
        ("run_energy_gradmem_curriculum_all.sh", "all"),
    ],
)
def test_zoology_curriculum_scripts_preserve_schedules_and_use_shared_runner(
    script_name,
    regime,
):
    script_dir = REPO_ROOT / "scripts" / "zoology_mh"
    entry = (script_dir / script_name).read_text()
    wrapper = (script_dir / "run_energy_gradmem_curriculum_common.sh").read_text()
    shared = (
        REPO_ROOT
        / "scripts"
        / "ar_multihop"
        / "run_energy_gradmem_curriculum_common.sh"
    ).read_text()
    assert f"SEGMENT_REGIME={regime}" in entry
    schedules = [
        re.search(rf"{name}=\(([^)]*)\)", entry).group(1).split()
        for name in ("CURRICULUM_N", "CE_WEIGHTS", "INNER_LRS")
    ]
    assert len(schedules[0]) == len(schedules[1]) == len(schedules[2]) == 6
    assert "DATASET_FORMAT=zoology" in wrapper
    assert "DEFAULT_HF_DATASET=irodkin/zoology_multihop" in wrapper
    assert "DATASET_VOCAB_SUFFIX=4096" in wrapper
    assert "MODEL_VOCAB_SIZE=4096" in wrapper
    assert "DEFAULT_WANDB_PROJECT=zoology_multihop" in wrapper
    assert "RUN_NAME_PREFIX=energy_gradmem_zoology_mh" in wrapper
    assert "INCLUDE_HOP_IN_WANDB_NAME=false" in wrapper
    assert "zoology_curriculum_gate.py" in wrapper
    assert '"$REPO_ROOT/run_energy_gradmem_on_ar_multihop.py"' in shared
    assert '--dataset_format "$DATASET_FORMAT"' in shared
