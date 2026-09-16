#!/usr/bin/env python3
"""Backward-compatible Zoology entry point for the shared curriculum gate."""

from ar_multihop_curriculum_gate import (
    evaluate_curriculum_gate,
    main,
    parse_args,
    parse_bool,
)

__all__ = ["evaluate_curriculum_gate", "main", "parse_args", "parse_bool"]


if __name__ == "__main__":
    raise SystemExit(main())
