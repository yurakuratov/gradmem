import json
from pathlib import Path
from unittest.mock import patch

import torch

from evaluate_energy_shaping import (
    METRIC_FAMILY_VERSIONS,
    METRIC_CACHE_DIRNAME,
    ResolvedRun,
    build_parser,
    checkpoint_file_digest,
    checkpoint_metric_cache_root,
    evaluate_resolved_run,
    load_metric_cache,
    metric_cache_path,
    metric_cache_signature,
    metric_family_configs,
    parse_int_list,
    relabel_metric_rows,
)


def make_spec(tmp_path: Path, alias: str = "comparison-name") -> ResolvedRun:
    run_path = tmp_path / "run_1"
    checkpoint_path = run_path / "checkpoint-10" / "model.safetensors"
    checkpoint_path.parent.mkdir(parents=True)
    checkpoint_path.write_bytes(b"weights")
    return ResolvedRun(
        alias=alias,
        source_path=run_path,
        run_path=run_path,
        run_id="run_1",
        seed=7,
        checkpoint_selector="best",
        checkpoint_path=checkpoint_path,
        checkpoint_name="checkpoint-10",
        is_recorded_best=False,
    )


def test_metric_cache_is_checkpoint_local_and_signature_scoped(tmp_path):
    spec = make_spec(tmp_path)
    signature = metric_cache_signature("abc", "task", {"n_values": [8], "inner_steps": [2]})
    cache_path = metric_cache_path(spec, signature)

    assert checkpoint_metric_cache_root(spec) == spec.checkpoint_path.parent / METRIC_CACHE_DIRNAME
    assert cache_path.parent == checkpoint_metric_cache_root(spec) / "task"
    assert metric_cache_path(spec, signature) == cache_path

    changed = metric_cache_signature("abc", "task", {"n_values": [8], "inner_steps": [4]})
    assert metric_cache_path(spec, changed) != cache_path


def test_metric_cache_reuses_only_complete_exact_signature(tmp_path):
    spec = make_spec(tmp_path)
    signature = metric_cache_signature("abc", "matching", {"n_values": [8]})
    cache_path = metric_cache_path(spec, signature)
    cache_path.mkdir(parents=True)
    payload = {"summary": [{"top1_accuracy": 0.75}], "rows": []}
    (cache_path / "metrics.json").write_text(json.dumps(payload))
    (cache_path / "manifest.json").write_text(json.dumps({
        "status": "complete",
        "signature": signature,
        "objective_type": "learned_energy",
        "tokenizer_signature_sha256": "tokenizer",
    }))

    loaded = load_metric_cache(cache_path, signature)
    assert loaded is not None
    assert loaded[0] == payload

    changed = metric_cache_signature("abc", "matching", {"n_values": [16]})
    assert load_metric_cache(cache_path, changed) is None

    manifest = json.loads((cache_path / "manifest.json").read_text())
    manifest["status"] = "running"
    (cache_path / "manifest.json").write_text(json.dumps(manifest))
    assert load_metric_cache(cache_path, signature) is None


def test_relabel_metric_rows_keeps_cached_rows_alias_neutral(tmp_path):
    spec = make_spec(tmp_path, alias="shaped")
    cached = [{"model": "checkpoint", "N": 8, "K": 2, "exact_match": 1.0}]

    materialized = relabel_metric_rows(cached, spec)

    assert cached == [{"model": "checkpoint", "N": 8, "K": 2, "exact_match": 1.0}]
    assert materialized == [{
        "model": "shaped",
        "N": 8,
        "K": 2,
        "exact_match": 1.0,
        "alias": "shaped",
        "run_id": "run_1",
        "seed": 7,
        "checkpoint": "checkpoint-10",
        "checkpoint_path": str(spec.checkpoint_path),
    }]


def test_complete_caches_materialize_without_model_or_dataset_loading(tmp_path):
    spec = make_spec(tmp_path, alias="cached")
    args = build_parser().parse_args([
        "--model", "cached", str(spec.run_path), str(spec.checkpoint_path),
        "--output-dir", str(tmp_path / "comparison"),
        "--data-root", str(tmp_path / "missing-data"),
        "--n-values", "8",
        "--inner-steps", "2",
        "--skip-landscape",
    ])
    args.n_values = parse_int_list(args.n_values)
    args.inner_steps = parse_int_list(args.inner_steps)
    args.landscape_n_values = parse_int_list(args.landscape_n_values)
    args.matching_n_values = args.n_values

    task_summary = {
        "model": "checkpoint",
        "objective_type": "learned_energy",
        "N": 8,
        "K": 2,
        "num_examples": 1,
        "exact_match": 1.0,
        "token_accuracy": 1.0,
        "target_loss": 0.1,
        "objective_decrease": 0.2,
        "runtime_seconds": 1.0,
    }
    payloads = {
        "task": {"rows": [], "summary": [task_summary]},
        "matching": {"summary": [], "rows": []},
        "interpolation": {"rows": [], "summary": []},
        "radial": {"rows": []},
        "contours": {"rows": []},
    }
    digest = checkpoint_file_digest(spec.checkpoint_path)
    for family, config in metric_family_configs(args).items():
        signature = metric_cache_signature(digest, family, config)
        cache_path = metric_cache_path(spec, signature)
        cache_path.mkdir(parents=True)
        (cache_path / "metrics.json").write_text(json.dumps(payloads[family]))
        (cache_path / "manifest.json").write_text(json.dumps({
            "status": "complete",
            "signature": signature,
            "objective_type": "learned_energy",
            "tokenizer_signature_sha256": "tokenizer",
        }))

    loaded_datasets = {}
    with patch("evaluate_energy_shaping.load_frozen_model", side_effect=AssertionError("model loaded")):
        output, tokenizer_digest = evaluate_resolved_run(
            spec, args, loaded_datasets, device=torch.device("cpu")
        )

    assert tokenizer_digest == "tokenizer"
    assert loaded_datasets == {}
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["status"] == "complete"
    assert set(manifest["metric_caches"]) == set(METRIC_FAMILY_VERSIONS)
    assert all(cache["reused"] for cache in manifest["metric_caches"].values())
