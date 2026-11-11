"""Controlled linear associative-memory experiments."""

from .model import (
    LinearAssociativeWriter,
    build_commit_schedule,
    error_decomposition,
    exact_snapshot_read,
    folded_linear_attention_read,
    linear_attention_read,
    operation_estimates,
    two_fact_expected_errors,
)

__all__ = [
    "LinearAssociativeWriter",
    "build_commit_schedule",
    "error_decomposition",
    "exact_snapshot_read",
    "folded_linear_attention_read",
    "linear_attention_read",
    "operation_estimates",
    "two_fact_expected_errors",
]

