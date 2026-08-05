#!/usr/bin/env python3
"""Populate checkpoint-local energy-shaping metric caches under a runs tree."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluate_energy_shaping import (  # noqa: E402
    CHECKPOINT_DIR_RE,
    ResolvedRun,
    build_parser as build_evaluator_parser,
    evaluate_resolved_run,
    load_run_cli_args,
    parse_int_list,
    recorded_best_checkpoint,
    resolve_device,
    seed_everything,
    validate_args as validate_evaluator_args,
)


RESERVED_EVALUATOR_OPTIONS = {
    "--aliases",
    "--baseline-run",
    "--cache-only",
    "--checkpoint",
    "--checkpoint-selector",
    "--experiment-paths",
    "--include-incomplete",
    "--model",
    "--output-dir",
    "--resume",
    "--shaped-run",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    parser.add_argument(
        "--index-dir",
        type=Path,
        default=None,
        help="small manifests for the cache sweep (default: RUNS_DIR/.energy_shaping_cache_build)",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="stop at the first checkpoint that cannot be evaluated",
    )
    parser.add_argument(
        "evaluator_args",
        nargs=argparse.REMAINDER,
        help="options forwarded to evaluate_energy_shaping.py; place them after --",
    )
    return parser


def forwarded_args(values: Sequence[str]) -> list[str]:
    result = list(values)
    if result and result[0] == "--":
        result = result[1:]
    forbidden = [value for value in result if value.split("=", 1)[0] in RESERVED_EVALUATOR_OPTIONS]
    if forbidden:
        raise ValueError(f"These evaluator options are managed by this script: {forbidden}")
    return result


def is_supported_checkpoint(checkpoint_dir: Path) -> tuple[bool, str]:
    config_path = checkpoint_dir / "config.json"
    weights_path = checkpoint_dir / "model.safetensors"
    run_config_path = checkpoint_dir.parent / "config.json"
    if not config_path.is_file() or not weights_path.is_file():
        return False, "missing checkpoint config or model.safetensors"
    if not run_config_path.is_file():
        return False, "parent run has no config.json"
    try:
        config = json.loads(config_path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        return False, f"invalid checkpoint config: {error}"
    if "GradMemGPT" not in config.get("architectures", []):
        return False, "not a GradMemGPT checkpoint"
    backend = config.get("memory_backend", "prefix")
    if backend != "prefix":
        return False, f"unsupported memory backend {backend!r}"
    return True, ""


def discover_checkpoints(runs_dir: Path) -> tuple[list[Path], list[tuple[Path, str]]]:
    supported = []
    skipped = []
    for weights_path in sorted(runs_dir.rglob("model.safetensors")):
        checkpoint_dir = weights_path.parent
        if CHECKPOINT_DIR_RE.fullmatch(checkpoint_dir.name) is None:
            continue
        is_supported, reason = is_supported_checkpoint(checkpoint_dir)
        if is_supported:
            supported.append(checkpoint_dir)
        else:
            skipped.append((checkpoint_dir, reason))
    return supported, skipped


def checkpoint_alias(runs_dir: Path, checkpoint_dir: Path) -> str:
    relative = checkpoint_dir.relative_to(runs_dir).as_posix()
    digest = hashlib.sha256(relative.encode("utf-8")).hexdigest()[:16]
    return f"checkpoint_{digest}"


def resolved_checkpoint(runs_dir: Path, checkpoint_dir: Path) -> ResolvedRun:
    run_path = checkpoint_dir.parent.resolve()
    cli_args = load_run_cli_args(run_path)
    if cli_args.get("seed") is None:
        raise ValueError(f"Run does not record a seed: {run_path}")
    checkpoint_path = (checkpoint_dir / "model.safetensors").resolve()
    best = recorded_best_checkpoint(run_path)
    return ResolvedRun(
        alias=checkpoint_alias(runs_dir, checkpoint_dir),
        source_path=run_path,
        run_path=run_path,
        run_id=run_path.name,
        seed=int(cli_args["seed"]),
        checkpoint_selector=checkpoint_dir.name,
        checkpoint_path=checkpoint_path,
        checkpoint_name=checkpoint_dir.name,
        is_recorded_best=bool(best is not None and best == checkpoint_path),
    )


def evaluator_namespace(values: Sequence[str], index_dir: Path) -> argparse.Namespace:
    args = build_evaluator_parser().parse_args([
        "--model", "cache", ".", "checkpoint-placeholder",
        *values,
    ])
    args.n_values = parse_int_list(args.n_values)
    args.inner_steps = parse_int_list(args.inner_steps)
    args.landscape_n_values = parse_int_list(args.landscape_n_values)
    args.matching_n_values = (
        args.n_values if args.matching_n_values is None else parse_int_list(args.matching_n_values)
    )
    args.output_dir = index_dir
    args.cache_only = True
    args.resume = True
    validate_evaluator_args(args)
    return args


def main() -> int:
    sweep_args = build_parser().parse_args()
    runs_dir = sweep_args.runs_dir.resolve()
    if not runs_dir.is_dir():
        raise FileNotFoundError(f"Runs directory does not exist: {runs_dir}")
    extra_args = forwarded_args(sweep_args.evaluator_args)
    index_dir = (
        sweep_args.index_dir.resolve()
        if sweep_args.index_dir is not None
        else runs_dir / ".energy_shaping_cache_build"
    )
    eval_args = evaluator_namespace(extra_args, index_dir)
    checkpoints, skipped = discover_checkpoints(runs_dir)
    if not checkpoints:
        raise ValueError(f"No supported GradMemGPT prefix checkpoints found under {runs_dir}")

    print(f"Found {len(checkpoints)} supported checkpoints under {runs_dir}", flush=True)
    for path, reason in skipped:
        print(f"[skip] {path}: {reason}", flush=True)
    if sweep_args.dry_run:
        for checkpoint_dir in checkpoints:
            print(f"[would cache] {checkpoint_dir}")
        return 0

    index_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(eval_args.device)
    seed_everything(eval_args.seed)
    loaded_datasets: dict[int, Any] = {}
    failures: list[tuple[Path, str]] = []
    for position, checkpoint_dir in enumerate(checkpoints, start=1):
        print(f"[{position}/{len(checkpoints)}] {checkpoint_dir}", flush=True)
        try:
            spec = resolved_checkpoint(runs_dir, checkpoint_dir)
            evaluate_resolved_run(spec, eval_args, loaded_datasets, device)
        except Exception as error:
            failures.append((checkpoint_dir, str(error)))
            print(f"[failed] {checkpoint_dir}: {error}", file=sys.stderr, flush=True)
            if sweep_args.fail_fast:
                raise

    print(
        f"Cache sweep complete: {len(checkpoints) - len(failures)} succeeded, "
        f"{len(failures)} failed, {len(skipped)} skipped",
        flush=True,
    )
    if failures:
        for path, reason in failures:
            print(f"[failure summary] {path}: {reason}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
