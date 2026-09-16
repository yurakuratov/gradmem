#!/usr/bin/env python3
"""Check whether an AR-multihop curriculum stage may progress."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Tuple


def parse_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise ValueError(f"expected true or false; got {value!r}")


def evaluate_curriculum_gate(
    metrics_path: Path,
    metric_name: str,
    threshold: float,
    *,
    lower_is_better: bool,
    enabled: bool = True,
) -> Tuple[bool, str]:
    """Return a gate decision and a diagnostic suitable for shell logs."""

    if not enabled:
        return True, "Curriculum gate disabled; continuing without a metric check."

    threshold = float(threshold)
    if not math.isfinite(threshold):
        return False, f"Curriculum gate threshold must be finite; got {threshold!r}."
    if not metric_name:
        return False, "Curriculum gate metric name must not be empty."
    if not metrics_path.is_file():
        return False, f"Curriculum progression metrics file is missing: {metrics_path}"

    try:
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return False, f"Could not read curriculum progression metrics {metrics_path}: {error}"
    if not isinstance(metrics, dict):
        return False, f"Curriculum progression metrics must be a JSON object: {metrics_path}"
    if metric_name not in metrics:
        return False, (
            f"Curriculum metric {metric_name!r} is missing from {metrics_path}; "
            f"available metrics: {sorted(metrics)}"
        )

    raw_value = metrics[metric_name]
    if isinstance(raw_value, bool):
        return False, f"Curriculum metric {metric_name!r} must be numeric, not boolean."
    try:
        metric_value = float(raw_value)
    except (TypeError, ValueError):
        return False, (
            f"Curriculum metric {metric_name!r} must be finite and numeric; "
            f"got {raw_value!r}."
        )
    if not math.isfinite(metric_value):
        return False, (
            f"Curriculum metric {metric_name!r} must be finite; got {metric_value!r}."
        )

    comparison = "<=" if lower_is_better else ">="
    passed = metric_value <= threshold if lower_is_better else metric_value >= threshold
    status = "passed" if passed else "failed"
    diagnostic = (
        f"Curriculum gate {status}: {metric_name}={metric_value:.8g} "
        f"must be {comparison} {threshold:.8g}."
    )
    return passed, diagnostic


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics-file", type=Path, required=True)
    parser.add_argument("--metric", required=True)
    parser.add_argument("--threshold", type=float, required=True)
    parser.add_argument("--lower-is-better", default="false")
    parser.add_argument("--enabled", default="true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        enabled = parse_bool(args.enabled)
        lower_is_better = parse_bool(args.lower_is_better)
    except ValueError as error:
        print(f"Curriculum gate configuration error: {error}")
        return 2

    passed, diagnostic = evaluate_curriculum_gate(
        args.metrics_file,
        args.metric,
        args.threshold,
        lower_is_better=lower_is_better,
        enabled=enabled,
    )
    print(diagnostic)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
