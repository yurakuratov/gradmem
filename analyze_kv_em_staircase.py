#!/usr/bin/env python3
"""Probe READ errors and WRITE token roles across KV-retrieval checkpoints."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import datasets
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from analyze_mix_write_components import token_components
from evaluate_kv_checkpoints import load_model, load_run_args, move_to_device, resolve_device
from run_gradmemgpt_on_kv_retrieval import collate_fn


PAIR_RE = re.compile(r"!([^!:|]+):([^!:|]+)!")
ROLE_NAMES = ("key_1", "key_2", "value_1", "value_2", "syntax", "filler")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", action="append", required=True)
    parser.add_argument("--alias", action="append", required=True)
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--tokenizer-path", type=Path)
    parser.add_argument("--max-examples", type=int)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    if len(args.alias) != len(args.checkpoint):
        parser.error("provide exactly one --alias per --checkpoint")
    if args.max_examples is not None and args.max_examples <= 0:
        parser.error("--max-examples must be positive")
    return args


def character_roles(context: str) -> dict[str, list[bool]]:
    roles = {name: [False] * len(context) for name in ROLE_NAMES}
    occupied = [False] * len(context)
    for match in PAIR_RE.finditer(context):
        key_start, key_end = match.span(1)
        value_start, value_end = match.span(2)
        if key_end - key_start != 2 or value_end - value_start != 2:
            raise ValueError(f"Expected two-character keys and values: {match.group(0)!r}")
        for name, index in (
            ("key_1", key_start),
            ("key_2", key_start + 1),
            ("value_1", value_start),
            ("value_2", value_start + 1),
        ):
            roles[name][index] = True
            occupied[index] = True
    if not any(roles["key_1"]) or not any(roles["value_1"]):
        raise ValueError(f"Could not parse key/value spans from context: {context!r}")
    for index, char in enumerate(context):
        if occupied[index]:
            continue
        roles["syntax" if char in "!?:|" else "filler"][index] = True
    return roles


def collate_diagnostics(rows: list[dict[str, str]], tokenizer: Any) -> dict[str, Any]:
    model_batch = collate_fn(rows, tokenizer)
    contexts = [row["context"] for row in rows]
    encoded = tokenizer(
        contexts,
        return_tensors="pt",
        add_special_tokens=True,
        padding=True,
        pad_to_multiple_of=8,
        return_offsets_mapping=True,
    )
    context_ids = model_batch["input_ids"]["context_input_ids"]
    if not torch.equal(encoded.input_ids, context_ids):
        raise ValueError("Diagnostic tokenization does not match the model collator")
    role_masks = {name: torch.zeros_like(context_ids, dtype=torch.bool) for name in ROLE_NAMES}
    query_pair_indices = []
    for row_index, (context, offsets) in enumerate(zip(contexts, encoded.offset_mapping.tolist())):
        row = rows[row_index]
        char_roles = character_roles(context)
        query_key = row["query"][2:-1]
        pair_keys = [match.group(1) for match in PAIR_RE.finditer(context)]
        try:
            query_pair_indices.append(pair_keys.index(query_key) + 1)
        except ValueError as error:
            raise ValueError(f"Query key {query_key!r} is absent from context") from error
        for token_index, (start, end) in enumerate(offsets):
            if end <= start:
                continue
            for name in ROLE_NAMES:
                role_masks[name][row_index, token_index] = any(char_roles[name][start:end])
    return {**model_batch, "role_masks": role_masks, "query_pair_indices": query_pair_indices}


def read_predictions(model: Any, backend: Any, memory_state: dict[str, Any], batch_ctx: dict[str, Any]):
    read_batch = backend.build_read_inputs(memory_state, batch_ctx)
    with torch.no_grad(), backend.activation_context(memory_state), model._disable_write_lora():
        output = model.model(
            inputs_embeds=read_batch["inputs_embeds"],
            return_dict=True,
            **read_batch.get("model_kwargs", {}),
        )
    logits = output.logits[
        :, read_batch["logits_start"]:read_batch["logits_start"] + read_batch["pred_len"], :
    ]
    return logits, read_batch


def add_read_totals(
    totals: dict[tuple[int, str], float],
    error_predictions: dict[tuple[int, int], Counter[int]],
    inner_step: int,
    logits: torch.Tensor,
    read_batch: dict[str, Any],
    labels: torch.Tensor,
    ignored_ids: set[int],
    query_pair_indices: list[int],
) -> None:
    if logits.size(1) == labels.size(1) + 1:
        aligned_logits, aligned_labels = logits[:, :-1], labels
    elif logits.size(1) == labels.size(1):
        aligned_logits, aligned_labels = logits[:, :-1], labels[:, 1:]
    else:
        raise ValueError("Unexpected READ prediction/label alignment")
    mask = aligned_labels.ne(-100)
    for token_id in ignored_ids:
        mask &= aligned_labels.ne(token_id)
    predictions = aligned_logits.argmax(dim=-1)
    losses = F.cross_entropy(
        aligned_logits.reshape(-1, aligned_logits.size(-1)),
        aligned_labels.reshape(-1),
        ignore_index=-100,
        reduction="none",
    ).view_as(aligned_labels)
    for row in range(labels.size(0)):
        positions = mask[row].nonzero(as_tuple=False).flatten()
        if positions.numel() != 2:
            raise ValueError(f"Expected exactly two scored value tokens, got {positions.numel()}")
        correct = []
        for target_position, sequence_position in enumerate(positions.tolist(), start=1):
            is_correct = bool(predictions[row, sequence_position] == aligned_labels[row, sequence_position])
            target_id = int(aligned_labels[row, sequence_position])
            correct.append(is_correct)
            totals[(inner_step, f"value_{target_position}_correct")] += int(is_correct)
            totals[(inner_step, f"value_{target_position}_ce")] += float(losses[row, sequence_position])
            totals[(inner_step, f"value_{target_position}_token_{target_id}_examples")] += 1
            totals[(inner_step, f"value_{target_position}_token_{target_id}_correct")] += int(is_correct)
            if not is_correct:
                predicted_id = int(predictions[row, sequence_position])
                error_predictions[(inner_step, target_position)][predicted_id] += 1
        category = (
            "both_correct" if all(correct) else
            "only_value_1" if correct[0] else
            "only_value_2" if correct[1] else
            "neither_correct"
        )
        totals[(inner_step, category)] += 1
        totals[(inner_step, "examples")] += 1
        pair_index = query_pair_indices[row]
        totals[(inner_step, f"pair_{pair_index}_examples")] += 1
        totals[(inner_step, f"pair_{pair_index}_value_1_correct")] += int(correct[0])
        totals[(inner_step, f"pair_{pair_index}_value_2_correct")] += int(correct[1])
        totals[(inner_step, f"pair_{pair_index}_exact_match")] += int(all(correct))


def add_write_totals(
    totals: dict[tuple[int, str], float],
    inner_step: int,
    reconstruction: torch.Tensor,
    reconstruction_correct: torch.Tensor,
    energy: torch.Tensor,
    role_masks: dict[str, torch.Tensor],
) -> None:
    for role, mask in role_masks.items():
        count = int(mask.sum())
        totals[(inner_step, f"write_{role}_count")] += count
        totals[(inner_step, f"write_{role}_reconstruction_ce")] += float(
            reconstruction[mask].detach().double().sum().cpu()
        )
        totals[(inner_step, f"write_{role}_reconstruction_correct")] += int(
            reconstruction_correct[mask].sum()
        )
        totals[(inner_step, f"write_{role}_energy")] += float(energy[mask].detach().double().sum().cpu())


def evaluate_checkpoint(
    checkpoint: Path,
    alias: str,
    dataset: Any,
    tokenizer: Any,
    batch_size: int,
    device: torch.device,
) -> dict[str, Any]:
    model, checkpoint_dir, _ = load_model(checkpoint, device)
    if model.memory_backend != "prefix":
        raise ValueError("Diagnostics require prefix memory")
    totals: dict[tuple[int, str], float] = defaultdict(float)
    error_predictions: dict[tuple[int, int], Counter[int]] = defaultdict(Counter)
    ignored_ids = {tokenizer.convert_tokens_to_ids(token) for token in ("!", "|")}
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda rows: collate_diagnostics(rows, tokenizer),
    )
    for batch in loader:
        role_masks = {name: value.to(device) for name, value in batch.pop("role_masks").items()}
        query_pair_indices = batch.pop("query_pair_indices")
        batch = move_to_device(batch, device)
        input_ids = batch["input_ids"]
        backend = model.memory_backend_impl
        batch_ctx = backend.prepare_batch(
            input_ids["context_input_ids"], input_ids["query_input_ids"], model.model.config.pad_token_id
        )
        memory_state, initial_state = backend.init_memory_state(input_ids["context_input_ids"].size(0))
        initial_memory = initial_state["mem_batch"]
        opt_state: dict[str, dict[str, torch.Tensor]] = {}
        for inner_step in range(model.K + 1):
            memory_state = {
                name: value.detach().requires_grad_(True) if isinstance(value, torch.Tensor) else value
                for name, value in memory_state.items()
            }
            with torch.enable_grad(), backend.activation_context(memory_state):
                write_batch = backend.build_write_inputs(memory_state, batch_ctx)
                reconstruction, reconstruction_correct, energy = token_components(model, write_batch)
                denominator = write_batch["mask"].sum(dim=1).clamp_min(1)
                objective = (energy * write_batch["mask"]).sum(dim=1) / denominator
                gradients = torch.autograd.grad(objective.sum(), backend.inner_params(memory_state))
            add_write_totals(
                totals, inner_step, reconstruction, reconstruction_correct, energy, role_masks
            )
            grad_norm = gradients[0].detach().float().flatten(1).norm(dim=1)
            memory = memory_state["mem_batch"].detach().float()
            totals[(inner_step, "gradient_norm")] += float(grad_norm.double().sum().cpu())
            totals[(inner_step, "memory_norm")] += float(memory.flatten(1).norm(dim=1).double().sum().cpu())
            totals[(inner_step, "delta_memory_norm")] += float(
                (memory - initial_memory).flatten(1).norm(dim=1).double().sum().cpu()
            )
            totals[(inner_step, "state_examples")] += memory.size(0)
            logits, read_batch = read_predictions(model, backend, memory_state, batch_ctx)
            add_read_totals(
                totals,
                error_predictions,
                inner_step,
                logits,
                read_batch,
                batch["labels"],
                ignored_ids,
                query_pair_indices,
            )
            if inner_step == model.K:
                break
            new_params = []
            for index, (parameter, gradient) in enumerate(zip(backend.inner_params(memory_state), gradients)):
                if model.use_adam:
                    updated = model._adam_step(
                        parameter, gradient, opt_state.setdefault(str(index), {}), inner_step + 1, model.lr
                    )
                else:
                    updated = model._sgd_step(
                        parameter,
                        gradient,
                        clip_value=model.inner_clip_value,
                        clip_norm=model.inner_clip_norm,
                    )
                new_params.append(updated.detach())
            backend.assign_inner_params(memory_state, new_params)

    rows = []
    for inner_step in range(model.K + 1):
        examples = totals[(inner_step, "examples")]
        state_examples = totals[(inner_step, "state_examples")]
        row: dict[str, Any] = {
            "checkpoint": str(checkpoint_dir),
            "alias": alias,
            "inner_step": inner_step,
            "examples": int(examples),
            "value_1_accuracy": totals[(inner_step, "value_1_correct")] / examples,
            "value_2_accuracy": totals[(inner_step, "value_2_correct")] / examples,
            "value_1_ce": totals[(inner_step, "value_1_ce")] / examples,
            "value_2_ce": totals[(inner_step, "value_2_ce")] / examples,
            "exact_match": totals[(inner_step, "both_correct")] / examples,
            "only_value_1_correct": totals[(inner_step, "only_value_1")] / examples,
            "only_value_2_correct": totals[(inner_step, "only_value_2")] / examples,
            "neither_correct": totals[(inner_step, "neither_correct")] / examples,
            "gradient_norm": totals[(inner_step, "gradient_norm")] / state_examples,
            "memory_norm": totals[(inner_step, "memory_norm")] / state_examples,
            "delta_memory_norm": totals[(inner_step, "delta_memory_norm")] / state_examples,
            "by_pair_index": [],
            "by_target_token": {"value_1": [], "value_2": []},
        }
        for pair_index in range(1, 9):
            pair_examples = totals[(inner_step, f"pair_{pair_index}_examples")]
            row["by_pair_index"].append({
                "pair_index": pair_index,
                "examples": int(pair_examples),
                "value_1_accuracy": (
                    totals[(inner_step, f"pair_{pair_index}_value_1_correct")] / pair_examples
                ),
                "value_2_accuracy": (
                    totals[(inner_step, f"pair_{pair_index}_value_2_correct")] / pair_examples
                ),
                "exact_match": (
                    totals[(inner_step, f"pair_{pair_index}_exact_match")] / pair_examples
                ),
            })
        alphabet_ids = [
            token_id for token_id in range(tokenizer.vocab_size)
            if tokenizer.decode([token_id]) not in {"", "!", "?", ":", "|"}
            and token_id not in tokenizer.all_special_ids
        ]
        for target_position in (1, 2):
            for token_id in alphabet_ids:
                token_examples = totals[(
                    inner_step, f"value_{target_position}_token_{token_id}_examples"
                )]
                if not token_examples:
                    continue
                row["by_target_token"][f"value_{target_position}"].append({
                    "token": tokenizer.decode([token_id]),
                    "examples": int(token_examples),
                    "accuracy": totals[(
                        inner_step, f"value_{target_position}_token_{token_id}_correct"
                    )] / token_examples,
                })
        for role in ROLE_NAMES:
            count = totals[(inner_step, f"write_{role}_count")]
            row[f"write_{role}_reconstruction_ce"] = (
                totals[(inner_step, f"write_{role}_reconstruction_ce")] / count if count else None
            )
            row[f"write_{role}_reconstruction_accuracy"] = (
                totals[(inner_step, f"write_{role}_reconstruction_correct")] / count if count else None
            )
            row[f"write_{role}_energy"] = (
                totals[(inner_step, f"write_{role}_energy")] / count if count else None
            )
        for target_position in (1, 2):
            counter = error_predictions[(inner_step, target_position)]
            row[f"value_{target_position}_error_count"] = sum(counter.values())
            row[f"value_{target_position}_top_error_predictions"] = [
                {"token": tokenizer.decode([token_id]), "count": count}
                for token_id, count in counter.most_common(10)
            ]
        rows.append(row)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {"alias": alias, "checkpoint": str(checkpoint_dir), "rows": rows}


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    dataset = datasets.load_from_disk(str(args.data_path))["valid"]
    if args.max_examples is not None:
        dataset = dataset.select(range(min(args.max_examples, len(dataset))))
    results = []
    for alias, checkpoint_text in zip(args.alias, args.checkpoint):
        checkpoint = Path(checkpoint_text).expanduser().resolve()
        run_args = load_run_args(checkpoint)
        tokenizer_path = args.tokenizer_path or run_args.get("tokenizer_path")
        if not tokenizer_path:
            raise ValueError(f"No tokenizer path available for {checkpoint}")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
        print(f"Evaluating {alias}: {checkpoint}", flush=True)
        results.append(evaluate_checkpoint(
            checkpoint, alias, dataset, tokenizer, args.batch_size, device
        ))
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(results, indent=2) + "\n")
    print(f"Wrote {args.output_json}")


if __name__ == "__main__":
    main()
