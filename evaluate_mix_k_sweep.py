#!/usr/bin/env python3
"""Evaluate mix-KV checkpoints by orientation and evaluation-time WRITE steps."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Any

import datasets
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from analyze_mix_write_components import collate_diagnostics
from evaluate_kv_checkpoints import load_model, load_run_args, move_to_device, resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", action="append", required=True)
    parser.add_argument("--alias", action="append", required=True)
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--tokenizer-path", type=Path)
    parser.add_argument("--inner-steps", type=int, nargs="+", default=[0, 1, 2, 4, 8])
    parser.add_argument("--max-examples", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-csv", type=Path, required=True)
    args = parser.parse_args()
    if len(args.alias) != len(args.checkpoint):
        parser.error("provide exactly one --alias per --checkpoint")
    if args.max_examples <= 0 or args.batch_size <= 0:
        parser.error("--max-examples and --batch-size must be positive")
    if any(step < 0 for step in args.inner_steps):
        parser.error("--inner-steps values must be non-negative")
    return args


def evaluate(model: Any, dataset: Any, tokenizer: Any, batch_size: int,
             inner_steps: int, device: torch.device) -> list[dict[str, Any]]:
    model.K = inner_steps
    model.last_K_second_order = 0
    totals: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0, 0])
    ignore_token_ids = [tokenizer.convert_tokens_to_ids(token) for token in ("!", "|")]
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda rows: collate_diagnostics(rows, tokenizer),
    )
    for batch in loader:
        directions = batch.pop("directions")
        batch.pop("key_mask")
        batch.pop("value_mask")
        batch = move_to_device(batch, device)
        output = model(batch["input_ids"], labels=batch["labels"])
        logits = output["predictions"]
        labels = batch["labels"]
        if logits.size(1) == labels.size(1) + 1:
            predictions, aligned_labels = logits[:, :-1].argmax(dim=-1), labels
        elif logits.size(1) == labels.size(1):
            predictions, aligned_labels = logits[:, :-1].argmax(dim=-1), labels[:, 1:]
        else:
            raise ValueError("Unexpected READ prediction/label alignment")
        mask = aligned_labels.ne(-100)
        for token_id in ignore_token_ids:
            mask &= aligned_labels.ne(token_id)
        correct = predictions.eq(aligned_labels) & mask
        for row_index, direction in enumerate(directions):
            token_count = int(mask[row_index].sum())
            token_correct = int(correct[row_index].sum())
            bucket = totals[direction]
            bucket[0] += token_correct
            bucket[1] += token_count
            bucket[2] += int(token_correct == token_count)
            bucket[3] += 1

    rows = []
    for direction, (token_correct, token_count, exact, examples) in sorted(totals.items()):
        rows.append({
            "K": inner_steps,
            "direction": direction,
            "token_accuracy": token_correct / token_count,
            "exact_match": exact / examples,
            "exact_matches": exact,
            "example_count": examples,
        })
    return rows


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    dataset = datasets.load_from_disk(str(args.data_path))["valid"]
    dataset = dataset.select(range(min(args.max_examples, len(dataset))))
    rows = []
    for alias, checkpoint in zip(args.alias, args.checkpoint):
        model, checkpoint_dir, run_args = load_model(Path(checkpoint), device)
        tokenizer_path = args.tokenizer_path or run_args.get("tokenizer_path")
        if not tokenizer_path:
            raise ValueError(f"No tokenizer path available for {checkpoint}")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
        training_steps = model.K
        for inner_steps in args.inner_steps:
            for row in evaluate(model, dataset, tokenizer, args.batch_size, inner_steps, device):
                rows.append({
                    "alias": alias,
                    "checkpoint": str(checkpoint_dir),
                    "training_K": training_steps,
                    **row,
                })
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {args.output_csv}")


if __name__ == "__main__":
    main()
