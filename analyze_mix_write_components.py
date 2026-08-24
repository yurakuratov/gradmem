#!/usr/bin/env python3
"""Measure semantic key/value WRITE diagnostics along checkpoint trajectories."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import datasets
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from evaluate_kv_checkpoints import load_model, load_run_args, move_to_device, resolve_device
from grad_memgpt import get_backbone
from run_gradmemgpt_on_kv_retrieval import collate_fn


PAIR_RE = re.compile(r"!([^!:|]+):([^!:|]+)!")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", action="append", required=True)
    parser.add_argument("--alias", action="append", required=True)
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--tokenizer-path", type=Path)
    parser.add_argument("--max-examples", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    args = parser.parse_args()
    if len(args.alias) != len(args.checkpoint):
        parser.error("provide exactly one --alias per --checkpoint")
    if args.max_examples <= 0 or args.batch_size <= 0:
        parser.error("--max-examples and --batch-size must be positive")
    return args


def semantic_character_masks(context: str) -> tuple[str, list[bool], list[bool]]:
    direction = context[0] if context[:1] in {"F", "B"} else "inverse"
    key_mask = [False] * len(context)
    value_mask = [False] * len(context)
    for match in PAIR_RE.finditer(context):
        left_span = range(match.start(1), match.end(1))
        right_span = range(match.start(2), match.end(2))
        if direction == "F":
            key_span, value_span = left_span, right_span
        else:
            value_span, key_span = left_span, right_span
        for index in key_span:
            key_mask[index] = True
        for index in value_span:
            value_mask[index] = True
    if not any(key_mask) or not any(value_mask):
        raise ValueError(f"Could not parse key/value spans from context: {context!r}")
    return direction, key_mask, value_mask


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

    key_masks = torch.zeros_like(context_ids, dtype=torch.bool)
    value_masks = torch.zeros_like(context_ids, dtype=torch.bool)
    directions = []
    for row_index, (context, offsets) in enumerate(zip(contexts, encoded.offset_mapping.tolist())):
        direction, key_chars, value_chars = semantic_character_masks(context)
        directions.append(direction)
        for token_index, (start, end) in enumerate(offsets):
            if end <= start:
                continue
            key_masks[row_index, token_index] = any(key_chars[start:end])
            value_masks[row_index, token_index] = any(value_chars[start:end])
    return {
        **model_batch,
        "key_mask": key_masks,
        "value_mask": value_masks,
        "directions": directions,
    }


def token_components(model: Any, write_batch: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    model_kwargs = dict(write_batch.get("model_kwargs", {}))
    if "attention_mask" in write_batch:
        model_kwargs["attention_mask"] = write_batch["attention_mask"]
    model_kwargs["output_hidden_states"] = model.use_layerwise_energy
    outputs = get_backbone(model.model)(
        inputs_embeds=write_batch["inputs_embeds"],
        return_dict=True,
        **model_kwargs,
    )

    hidden = outputs.last_hidden_state[:, write_batch["logits_start"]:, :]
    output_head = model.write_head if model.use_write_head else model.model.get_output_embeddings()
    logits = output_head(hidden)
    labels = write_batch["lm_labels"][:, write_batch.get("label_shift", 0):]
    reconstruction = F.cross_entropy(
        logits[:, :-1].reshape(-1, logits.size(-1)),
        labels.reshape(-1),
        ignore_index=-100,
        reduction="none",
    ).view(labels.shape)
    reconstruction_correct = logits[:, :-1].argmax(dim=-1).eq(labels)

    def layer_token_energy(layer_hidden: torch.Tensor, layer_norm: Any, energy_head: Any) -> torch.Tensor:
        context_hidden = layer_hidden[:, write_batch["context_start"]:, :]
        return F.softplus(energy_head(layer_norm(context_hidden))).squeeze(-1)

    if model.energy_head is None:
        energy = torch.full_like(reconstruction, float("nan"))
    elif model.use_layerwise_energy:
        energy = torch.stack([
            layer_token_energy(layer_hidden, layer_norm, energy_head)
            for layer_hidden, layer_norm, energy_head in zip(
                outputs.hidden_states[1:], model.energy_ln, model.energy_head
            )
        ]).sum(dim=0)
    else:
        energy = layer_token_energy(outputs.last_hidden_state, model.energy_ln, model.energy_head)
    return reconstruction, reconstruction_correct, energy


def objective_from_components(model: Any, reconstruction: torch.Tensor, energy: torch.Tensor,
                              mask: torch.Tensor) -> torch.Tensor:
    denominator = mask.sum(dim=1).clamp_min(1)
    reconstruction_mean = (reconstruction * mask).sum(dim=1) / denominator
    if model.write_objective == "reconstruction":
        return reconstruction_mean
    energy_mean = (energy * mask).sum(dim=1) / denominator
    if model.write_objective == "energy":
        return energy_mean
    return model.write_reconstruction_weight * reconstruction_mean + model.write_energy_weight * energy_mean


def add_component_sums(accumulator: dict[tuple[int, str, str], list[float]], step: int,
                       directions: list[str], component: str, values: torch.Tensor,
                       token_mask: torch.Tensor) -> None:
    for direction in sorted(set(directions)):
        example_mask = torch.tensor(
            [value == direction for value in directions], device=values.device, dtype=torch.bool
        )
        selected = token_mask & example_mask[:, None]
        if selected.any():
            bucket = accumulator[(step, direction, component)]
            bucket[0] += float(values[selected].detach().double().sum().cpu())
            bucket[1] += int(selected.sum())


def evaluate_checkpoint(checkpoint: str, alias: str, dataset: Any, tokenizer: Any,
                        batch_size: int, device: torch.device) -> list[dict[str, Any]]:
    model, checkpoint_dir, _ = load_model(Path(checkpoint), device)
    if model.memory_backend != "prefix":
        raise ValueError("WRITE-component diagnostics require prefix memory")
    totals: dict[tuple[int, str, str], list[float]] = defaultdict(lambda: [0.0, 0])
    accuracy_totals: dict[tuple[int, str, str], list[int]] = defaultdict(lambda: [0, 0])
    gradient_totals: dict[tuple[int, str], list[float]] = defaultdict(lambda: [0.0, 0])
    read_totals: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0, 0])
    ignore_token_ids = [tokenizer.convert_tokens_to_ids(token) for token in ("!", "|")]
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda rows: collate_diagnostics(rows, tokenizer),
    )

    for batch in loader:
        directions = batch.pop("directions")
        key_mask = batch.pop("key_mask").to(device)
        value_mask = batch.pop("value_mask").to(device)
        batch = move_to_device(batch, device)
        input_ids = batch["input_ids"]
        with torch.no_grad():
            read_output = model(input_ids, labels=batch["labels"])
        logits = read_output["predictions"]
        labels = batch["labels"]
        if logits.size(1) == labels.size(1) + 1:
            predictions, aligned_labels = logits[:, :-1].argmax(dim=-1), labels
        elif logits.size(1) == labels.size(1):
            predictions, aligned_labels = logits[:, :-1].argmax(dim=-1), labels[:, 1:]
        else:
            raise ValueError("Unexpected READ prediction/label alignment")
        score_mask = aligned_labels.ne(-100)
        for token_id in ignore_token_ids:
            score_mask &= aligned_labels.ne(token_id)
        correct = predictions.eq(aligned_labels) & score_mask
        for row_index, direction in enumerate(directions):
            token_count = int(score_mask[row_index].sum())
            token_correct = int(correct[row_index].sum())
            bucket = read_totals[direction]
            bucket[0] += token_correct
            bucket[1] += token_count
            bucket[2] += int(token_correct == token_count)
            bucket[3] += 1
        backend = model.memory_backend_impl
        batch_ctx = backend.prepare_batch(
            input_ids["context_input_ids"],
            input_ids["query_input_ids"],
            model.model.config.pad_token_id,
        )
        memory_state, _ = backend.init_memory_state(len(directions))
        opt_state: dict[str, dict[str, torch.Tensor]] = {}

        for step in range(model.K + 1):
            memory_state = {
                name: value.detach().requires_grad_(True) if isinstance(value, torch.Tensor) else value
                for name, value in memory_state.items()
            }
            with torch.enable_grad(), backend.activation_context(memory_state):
                write_batch = backend.build_write_inputs(memory_state, batch_ctx)
                reconstruction, reconstruction_correct, energy = token_components(model, write_batch)
                objective = objective_from_components(model, reconstruction, energy, write_batch["mask"])
                inner_params = backend.inner_params(memory_state)
                gradients = torch.autograd.grad(objective.sum(), inner_params)

            for component, values in (("reconstruction_key", reconstruction),
                                      ("reconstruction_value", reconstruction),
                                      ("energy_key", energy), ("energy_value", energy)):
                mask = key_mask if component.endswith("key") else value_mask
                if not torch.isnan(values).all():
                    add_component_sums(totals, step, directions, component, values, mask)
            for role, mask in (("key", key_mask), ("value", value_mask)):
                for direction in sorted(set(directions)):
                    example_mask = torch.tensor(
                        [value == direction for value in directions],
                        device=device,
                        dtype=torch.bool,
                    )
                    selected = mask & example_mask[:, None]
                    bucket = accuracy_totals[(step, direction, role)]
                    bucket[0] += int(reconstruction_correct[selected].sum())
                    bucket[1] += int(selected.sum())
            grad_norm = gradients[0].detach().float().flatten(1).norm(dim=1)
            for direction in sorted(set(directions)):
                selected = torch.tensor([value == direction for value in directions], device=device)
                bucket = gradient_totals[(step, direction)]
                bucket[0] += float(grad_norm[selected].double().sum().cpu())
                bucket[1] += int(selected.sum())

            if step == model.K:
                break
            updated = []
            for index, (parameter, gradient) in enumerate(zip(inner_params, gradients)):
                if model.use_adam:
                    value = model._adam_step(
                        parameter, gradient, opt_state.setdefault(str(index), {}), step + 1, model.lr
                    )
                else:
                    value = model._sgd_step(
                        parameter, gradient, clip_value=model.inner_clip_value, clip_norm=model.inner_clip_norm
                    )
                updated.append(value.detach())
            backend.assign_inner_params(memory_state, updated)

    rows = []
    for (step, direction, component), (total, count) in sorted(totals.items()):
        grad_total, grad_count = gradient_totals[(step, direction)]
        token_correct, token_count, exact_matches, example_count = read_totals[direction]
        role = "key" if component.endswith("key") else "value"
        reconstruction_correct, reconstruction_count = accuracy_totals[(step, direction, role)]
        rows.append({
            "alias": alias,
            "checkpoint": str(checkpoint_dir),
            "write_objective": model.write_objective,
            "step": step,
            "direction": direction,
            "component": component,
            "mean": total / count,
            "reconstruction_token_accuracy": reconstruction_correct / reconstruction_count,
            "token_count": count,
            "gradient_norm_mean": grad_total / grad_count,
            "example_count": grad_count,
            "token_accuracy": token_correct / token_count,
            "exact_match": exact_matches / example_count,
        })
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return rows


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    dataset = datasets.load_from_disk(str(args.data_path))["valid"]
    dataset = dataset.select(range(min(args.max_examples, len(dataset))))
    all_rows = []
    for alias, checkpoint in zip(args.alias, args.checkpoint):
        run_args = load_run_args(Path(checkpoint).expanduser().resolve())
        tokenizer_path = args.tokenizer_path or run_args.get("tokenizer_path")
        if not tokenizer_path:
            raise ValueError(f"No tokenizer path available for {checkpoint}")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
        all_rows.extend(evaluate_checkpoint(
            checkpoint, alias, dataset, tokenizer, args.batch_size, device
        ))

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(all_rows, indent=2) + "\n")
    with args.output_csv.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"Wrote {len(all_rows)} rows to {args.output_json} and {args.output_csv}")


if __name__ == "__main__":
    main()
