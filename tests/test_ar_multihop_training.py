import json
import math
import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from transformers import HfArgumentParser

from ar_multihop_curriculum_gate import evaluate_curriculum_gate
from ar_multihop_dataset import entity_length_for_configuration, generate_sample
from energy_gradmem import EnergyGradMem, EnergyGradMemConfig
from run_energy_gradmem_on_ar_multihop import (
    ARMultihopExperimentArgs,
    ARMultihopTrainer,
    BOS_ID,
    CONTEXT_END_ID,
    CONTEXT_START_ID,
    EOS_ID,
    KV_END_ID,
    KV_SEPARATOR_ID,
    KV_START_ID,
    MASK_ID,
    MODEL_VOCAB_SIZE,
    NUM_SPECIAL_TOKENS,
    PAD_ID,
    QUERY_END_ID,
    QUERY_START_ID,
    SPECIAL_TOKENS,
    aligned_target_prediction_positions,
    build_ar_multihop_base_config,
    collate_ar_multihop_batch,
    compute_ar_multihop_metrics,
    format_context_segments,
    formatted_kv_record_width,
    model_entity_id,
    progression_metrics_for_json,
    query_mask_positions,
    query_record_width,
    query_target_label_positions,
    save_progression_metrics,
    split_dataset,
    validate_ar_multihop_args,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    ("n_pairs", "hop_length", "pairs_per_segment", "expected_segments"),
    [
        (8, 1, 8, 1),
        (16, 1, 8, 2),
        (128, 1, 8, 16),
        (64, 4, 32, 2),
        (128, 8, 32, 4),
        (128, 8, 128, 1),
    ],
)
def test_context_segmentation_occurs_only_between_complete_kv_records(
    n_pairs,
    hop_length,
    pairs_per_segment,
    expected_segments,
):
    sample = generate_sample(n_pairs, hop_length, seed=11)
    entity_length = entity_length_for_configuration(n_pairs, hop_length)
    record_width = formatted_kv_record_width(entity_length)
    segments = format_context_segments(
        sample["context_keys"],
        sample["context_values"],
        n_pairs=n_pairs,
        kv_pairs_per_segment=pairs_per_segment,
        entity_length=entity_length,
    )

    assert len(segments) == expected_segments
    assert segments[0][:2] == [BOS_ID, CONTEXT_START_ID]
    assert segments[-1][-2:] == [CONTEXT_END_ID, EOS_ID]
    reconstructed_records = []
    for segment_index, segment in enumerate(segments):
        body_start = 2 if segment_index == 0 else 0
        body_end = -2 if segment_index == len(segments) - 1 else None
        body = segment[body_start:body_end]
        assert len(body) % record_width == 0
        reconstructed_records.extend(
            body[offset : offset + record_width]
            for offset in range(0, len(body), record_width)
        )

    assert len(reconstructed_records) == n_pairs
    for record in reconstructed_records:
        assert record[0] == KV_START_ID
        assert record[entity_length + 1] == KV_SEPARATOR_ID
        assert record[-1] == KV_END_ID


def test_collator_offsets_entities_and_inserts_distinct_special_tokens():
    sample = generate_sample(16, 1, seed=13)
    entity_length = entity_length_for_configuration(16, 1)
    batch = collate_ar_multihop_batch(
        [sample],
        n_pairs=16,
        hop_length=1,
        kv_pairs_per_segment=8,
    )

    assert set(SPECIAL_TOKENS.values()) == set(range(NUM_SPECIAL_TOKENS))
    assert len(set(SPECIAL_TOKENS.values())) == len(SPECIAL_TOKENS)
    assert MODEL_VOCAB_SIZE == NUM_SPECIAL_TOKENS + 16 == 27
    assert PAD_ID != MASK_ID
    assert model_entity_id(0) == NUM_SPECIAL_TOKENS
    assert model_entity_id(15) == MODEL_VOCAB_SIZE - 1
    for segment in batch["input_ids"]["context_input_ids"]:
        assert segment.min().item() >= 0
        assert segment.max().item() < MODEL_VOCAB_SIZE
    first_record = batch["input_ids"]["context_input_ids"][0][
        0, 2 : 2 + formatted_kv_record_width(entity_length)
    ]
    expected_key = [model_entity_id(digit) for digit in sample["context_keys"][0]]
    expected_value = [
        model_entity_id(digit) for digit in sample["context_values"][0]
    ]
    assert first_record.tolist() == [
        KV_START_ID,
        *expected_key,
        KV_SEPARATOR_ID,
        *expected_value,
        KV_END_ID,
    ]


def test_query_collation_uses_one_mask_per_target_digit_without_target_insertion():
    n_pairs, hop_length = 16, 1
    entity_length = entity_length_for_configuration(n_pairs, hop_length)
    sample = generate_sample(n_pairs, hop_length, seed=17)
    batch = collate_ar_multihop_batch(
        [sample],
        n_pairs=n_pairs,
        hop_length=hop_length,
        kv_pairs_per_segment=8,
    )
    query_ids = batch["input_ids"]["query_input_ids"][0]
    labels = batch["labels"][0]
    masks = query_mask_positions(n_pairs, entity_length)
    targets = query_target_label_positions(n_pairs, entity_length)

    assert query_ids[0].item() == BOS_ID
    assert query_ids[-1].item() == EOS_ID
    assert query_ids.eq(MASK_ID).sum().item() == n_pairs * entity_length
    assert query_ids[masks].eq(MASK_ID).all()
    assert labels.ne(-100).sum().item() == n_pairs * entity_length
    expected_targets = torch.tensor(
        [model_entity_id(digit) for target in sample["targets"] for digit in target]
    )
    assert torch.equal(labels[targets], expected_targets)
    assert query_ids[masks].eq(MASK_ID).all()

    width = query_record_width(entity_length)
    for query_index in range(n_pairs):
        start = 1 + query_index * width
        record = query_ids[start : start + width]
        assert record[0].item() == QUERY_START_ID
        assert record[-1].item() == QUERY_END_ID
        expected_key = [
            model_entity_id(digit) for digit in sample["query_keys"][query_index]
        ]
        assert record[1 : 1 + entity_length].tolist() == expected_key
        assert record[1 + entity_length : -1].eq(MASK_ID).all()


@pytest.mark.parametrize("label_shift", [0, 1])
def test_causal_labels_and_read_masks_select_mask_hidden_states(label_shift):
    n_pairs, entity_length = 2, 2
    query_length = 2 + n_pairs * query_record_width(entity_length)
    labels = torch.full((1, query_length), -100, dtype=torch.long)
    label_positions = query_target_label_positions(n_pairs, entity_length)
    labels[0, label_positions] = torch.tensor([11, 12, 13, 14])
    mask_positions = query_mask_positions(n_pairs, entity_length)

    pred_len = query_length + 1 if label_shift == 0 else query_length
    expected_prediction_positions = aligned_target_prediction_positions(
        n_pairs,
        entity_length,
        label_shift=label_shift,
    )
    model = _small_energy_model(reading_optimization=True)
    read_mask = model._read_optimization_mask(
        torch.ones(1, query_length, dtype=torch.long),
        labels.ne(-100),
        {"pred_len": pred_len, "label_shift": label_shift},
    )
    assert read_mask.nonzero(as_tuple=False)[:, 1].tolist() == expected_prediction_positions

    # Prefix predictions contain one pre-query hidden state; other backends do
    # not. In both conventions the selected model hidden states are the MASKs.
    hidden_input_positions = [
        position - 1 if label_shift == 0 else position
        for position in expected_prediction_positions
    ]
    assert hidden_input_positions == mask_positions

    logits = torch.full((1, pred_len, MODEL_VOCAB_SIZE), -10.0)
    for label_position, target in zip(label_positions, labels[0, label_positions]):
        prediction_position = label_position - label_shift
        logits[0, prediction_position, target] = 10.0
    loss = EnergyGradMem._target_loss(
        logits,
        labels,
        {"label_shift": label_shift},
    )
    assert loss.item() < 1e-6


def test_metrics_distinguish_tokens_complete_queries_and_complete_samples():
    n_pairs, entity_length = 2, 2
    sequence_length = 2 + n_pairs * query_record_width(entity_length)
    label_positions = query_target_label_positions(n_pairs, entity_length)
    labels = np.full((2, sequence_length), -100, dtype=np.int64)
    labels[:, label_positions] = np.array([[11, 12, 13, 14], [15, 16, 17, 18]])
    predictions = np.zeros((2, sequence_length + 1), dtype=np.int64)
    predictions[:, label_positions] = labels[:, label_positions]
    predictions[0, label_positions[-1]] = 19
    eval_pred = SimpleNamespace(
        predictions=(predictions, {"inner_loss": np.array([1.0, 3.0])}),
        label_ids=labels,
        inputs={"hop_distances": np.array([[1, 2], [1, 2]])},
    )

    metrics = compute_ar_multihop_metrics(eval_pred)
    assert metrics["token_accuracy"] == pytest.approx(7 / 8)
    assert metrics["query_accuracy"] == pytest.approx(3 / 4)
    assert metrics["all_queries_exact_match"] == pytest.approx(1 / 2)
    assert metrics["accuracy_hop_distance_1"] == pytest.approx(1.0)
    assert metrics["accuracy_hop_distance_2"] == pytest.approx(0.5)
    assert metrics["inner_loss"] == pytest.approx(2.0)


def test_ar_multihop_model_and_runner_defaults():
    args = ARMultihopExperimentArgs(exp_path="unused", per_device_batch_size=2)
    validate_ar_multihop_args(args)
    config = build_ar_multihop_base_config(args)

    assert args.hf_dataset == "irodkin/ar_multihop"
    assert args.hf_subset == "N8-H1-V16"
    assert config.vocab_size == MODEL_VOCAB_SIZE
    assert config.pad_token_id == PAD_ID
    assert config.num_hidden_layers == 2
    assert config.hidden_size == 128
    assert config.num_attention_heads == 1
    assert config.attention_dropout == pytest.approx(0.1)
    assert args.n_mem_tokens == 4
    assert args.energy_model_type == "segment_delta_gru"
    assert args.energy_replay_weight == 0.0
    assert args.reading_optimization is False
    assert args.outer_lr_final_ratio == pytest.approx(0.2)


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("hf_dataset", "irodkin/zoology_multihop"),
        ("hf_subset", "N8-H1-V4096"),
        ("vocab_size", 4096),
    ],
)
def test_ar_multihop_rejects_dataset_defaults_from_the_other_format(
    field_name,
    value,
):
    args = ARMultihopExperimentArgs(
        exp_path="unused",
        per_device_batch_size=2,
        **{field_name: value},
    )
    with pytest.raises(ValueError, match=field_name):
        validate_ar_multihop_args(args)


def test_dataset_format_must_be_selected_explicitly_from_supported_values():
    args = ARMultihopExperimentArgs(
        exp_path="unused",
        per_device_batch_size=2,
        dataset_format="unknown",
    )
    with pytest.raises(ValueError, match="dataset_format"):
        validate_ar_multihop_args(args)


def test_outer_lr_scheduler_warms_up_then_decays_to_exact_ratio():
    parameter = torch.nn.Parameter(torch.tensor(0.0))
    optimizer = torch.optim.SGD([parameter], lr=2.0)
    trainer = ARMultihopTrainer.__new__(ARMultihopTrainer)
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


def test_ar_cli_uses_unambiguous_stop_metric_name():
    destinations = {
        action.dest for action in HfArgumentParser(ARMultihopExperimentArgs)._actions
    }
    assert "stop_all_queries_exact_match_value" in destinations
    assert "stop_exact_match_value" not in destinations


def test_attention_dropout_and_scheduler_ratio_are_validated():
    custom_dropout = ARMultihopExperimentArgs(
        exp_path="unused",
        per_device_batch_size=2,
        attention_dropout=0.25,
    )
    validate_ar_multihop_args(custom_dropout)
    assert build_ar_multihop_base_config(custom_dropout).attention_dropout == pytest.approx(
        0.25
    )

    for field_name, values in (
        ("attention_dropout", (-0.1, 1.0)),
        ("outer_lr_final_ratio", (0.0, -0.1, math.inf, math.nan, 1.1)),
    ):
        for value in values:
            kwargs = {field_name: value}
            args = ARMultihopExperimentArgs(
                exp_path="unused",
                per_device_batch_size=2,
                **kwargs,
            )
            with pytest.raises(ValueError, match=field_name):
                validate_ar_multihop_args(args)


def test_dataset_split_keeps_validation_for_selection_and_test_for_reporting():
    dataset = {"train": object(), "validation": object(), "test": object()}
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
def test_ar_curriculum_entry_points_preserve_aligned_schedules(script_name, regime):
    script = (REPO_ROOT / "scripts" / "ar_multihop" / script_name).read_text()
    assert f"SEGMENT_REGIME={regime}" in script
    schedules = [
        re.search(rf"{name}=\(([^)]*)\)", script).group(1).split()
        for name in ("CURRICULUM_N", "CE_WEIGHTS", "INNER_LRS")
    ]
    assert len(schedules[0]) == len(schedules[1]) == len(schedules[2]) == 6


def test_common_curriculum_uses_configured_h_and_ar_defaults():
    helper = (
        REPO_ROOT
        / "scripts"
        / "ar_multihop"
        / "run_energy_gradmem_curriculum_common.sh"
    ).read_text()
    assert "DATASET_FORMAT=${DATASET_FORMAT:-ar_multihop}" in helper
    assert "DATASET_VOCAB_SUFFIX=${DATASET_VOCAB_SUFFIX:-16}" in helper
    assert "HOP_LENGTH=${HOP_LENGTH:-1}" in helper
    assert (
        "HF_SUBSET=N${DATASET_N}-H${HOP_LENGTH}-V${DATASET_VOCAB_SUFFIX}"
        in helper
    )
    assert '--hop_length "$HOP_LENGTH"' in helper
    assert "DEFAULT_HF_DATASET=${DEFAULT_HF_DATASET:-irodkin/ar_multihop}" in helper
    assert "MODEL_VOCAB_SIZE=${MODEL_VOCAB_SIZE:-27}" in helper
    assert '"$REPO_ROOT/run_energy_gradmem_on_ar_multihop.py"' in helper
    assert '--dataset_format "$DATASET_FORMAT"' in helper
    assert "CURRICULUM_METRIC=${CURRICULUM_METRIC:-query_accuracy}" in helper
    assert "CURRICULUM_METRIC_THRESHOLD=${CURRICULUM_METRIC_THRESHOLD:-0.9}" in helper
    assert "ENERGY_REPLAY_WEIGHT=0.0" in helper
    assert "READING_OPTIMIZATION=false" in helper


def _write_gate_metrics(tmp_path, metrics):
    path = tmp_path / "progression_metrics.json"
    path.write_text(json.dumps(metrics), encoding="utf-8")
    return path


@pytest.mark.parametrize(
    ("value", "threshold", "lower_is_better", "expected"),
    [
        (0.9, 0.8, False, True),
        (0.7, 0.8, False, False),
        (0.1, 0.2, True, True),
        (0.3, 0.2, True, False),
    ],
)
def test_curriculum_gate_supports_both_directions(
    tmp_path,
    value,
    threshold,
    lower_is_better,
    expected,
):
    path = _write_gate_metrics(tmp_path, {"query_accuracy": value})
    passed, diagnostic = evaluate_curriculum_gate(
        path,
        "query_accuracy",
        threshold,
        lower_is_better=lower_is_better,
    )
    assert passed is expected
    assert ("passed" if expected else "failed") in diagnostic


def test_curriculum_gate_disabled_missing_and_nonfinite_cases(tmp_path):
    missing_path = tmp_path / "missing.json"
    passed, _ = evaluate_curriculum_gate(
        missing_path,
        "query_accuracy",
        0.9,
        lower_is_better=False,
        enabled=False,
    )
    assert passed
    passed, diagnostic = evaluate_curriculum_gate(
        missing_path,
        "query_accuracy",
        0.9,
        lower_is_better=False,
    )
    assert not passed and "missing" in diagnostic
    missing_metric_path = _write_gate_metrics(tmp_path, {"token_accuracy": 1.0})
    passed, diagnostic = evaluate_curriculum_gate(
        missing_metric_path,
        "query_accuracy",
        0.9,
        lower_is_better=False,
    )
    assert not passed and "is missing" in diagnostic
    path = _write_gate_metrics(tmp_path, {"query_accuracy": float("nan")})
    passed, diagnostic = evaluate_curriculum_gate(
        path,
        "query_accuracy",
        0.9,
        lower_is_better=False,
    )
    assert not passed and "finite" in diagnostic


def test_progression_metrics_json_strips_prefix_and_handles_nonfinite(tmp_path):
    raw = {
        "progression_query_accuracy": 0.75,
        "progression_all_queries_exact_match": 0.5,
        "progression_nonfinite": float("inf"),
    }
    serialized = progression_metrics_for_json(raw)
    assert serialized == {
        "query_accuracy": 0.75,
        "all_queries_exact_match": 0.5,
        "nonfinite": None,
    }
    saved = json.loads(save_progression_metrics(tmp_path, raw).read_text())
    assert saved == serialized


def _small_energy_model(*, reading_optimization=False):
    args = ARMultihopExperimentArgs(
        exp_path="unused",
        per_device_batch_size=2,
        n_layer=1,
        n_embd=32,
        n_head=1,
    )
    validate_ar_multihop_args(args)
    return EnergyGradMem(
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
            reading_optimization=reading_optimization,
            K_read=1,
            segment_write_mode="sequential",
        )
    )


@pytest.mark.forward
def test_raw_multitoken_batch_runs_without_target_leak():
    samples = [generate_sample(16, 1, seed=seed) for seed in (23, 29)]
    batch = collate_ar_multihop_batch(
        samples,
        n_pairs=16,
        hop_length=1,
        kv_pairs_per_segment=8,
    )
    model = _small_energy_model()
    output = model(**batch)

    query_length = batch["input_ids"]["query_input_ids"].size(1)
    assert batch["labels"].ne(-100).sum(dim=1).tolist() == [32, 32]
    assert output["predictions"].shape == (2, query_length + 1, MODEL_VOCAB_SIZE)
    assert output["loss"].ndim == 0 and torch.isfinite(output["loss"])
    output["loss"].backward()
    assert torch.isfinite(model.token_energy_mlp[0].weight.grad).all()
