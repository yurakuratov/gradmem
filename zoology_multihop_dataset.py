"""Deterministic generator for the integer-encoded zoology multihop dataset.

The module has no model segmentation logic.  Each example contains one flat
context, one flat collection of independently shuffled queries, and aligned
target and hop-distance arrays.
"""

from __future__ import annotations

import hashlib
import random
from collections import Counter
from dataclasses import dataclass
from typing import Dict, Iterator, List, Mapping, Optional, Sequence, Tuple


VOCAB_SIZE = 4096
SPECIAL_TOKENS: Mapping[str, int] = {
    "PAD": 0,
    "BOS": 1,
    "EOS": 2,
    "CONTEXT_START": 3,
    "CONTEXT_END": 4,
    "KV_START": 5,
    "KV_SEPARATOR": 6,
    "KV_END": 7,
    "QUERY_START": 8,
    "QUERY_END": 9,
}
ENTITY_START_ID = len(SPECIAL_TOKENS)
ENTITY_STOP_ID = VOCAB_SIZE
ENTITY_COUNT = ENTITY_STOP_ID - ENTITY_START_ID
KV_RECORD_WIDTH = 5
QUERY_RECORD_WIDTH = 3

PAD_ID = SPECIAL_TOKENS["PAD"]
BOS_ID = SPECIAL_TOKENS["BOS"]
EOS_ID = SPECIAL_TOKENS["EOS"]
CONTEXT_START_ID = SPECIAL_TOKENS["CONTEXT_START"]
CONTEXT_END_ID = SPECIAL_TOKENS["CONTEXT_END"]
KV_START_ID = SPECIAL_TOKENS["KV_START"]
KV_SEPARATOR_ID = SPECIAL_TOKENS["KV_SEPARATOR"]
KV_END_ID = SPECIAL_TOKENS["KV_END"]
QUERY_START_ID = SPECIAL_TOKENS["QUERY_START"]
QUERY_END_ID = SPECIAL_TOKENS["QUERY_END"]

DEFAULT_N_VALUES = (8, 16, 32, 64, 128)
DEFAULT_H_VALUES = (1, 2, 4, 8)
DEFAULT_SPLIT_SIZES = {
    "train": 1_000_000,
    "validation": 5_000,
    "test": 10_000,
}
DEFAULT_SPLIT_SEEDS = {
    "train": 42,
    "validation": 43,
    "test": 44,
}
SPLIT_NAMES = tuple(DEFAULT_SPLIT_SIZES)

if len(set(SPECIAL_TOKENS.values())) != len(SPECIAL_TOKENS):
    raise RuntimeError("special-token IDs must be unique")
if set(SPECIAL_TOKENS.values()) != set(range(ENTITY_START_ID)):
    raise RuntimeError("special-token IDs must occupy one contiguous prefix")
if len(SPECIAL_TOKENS) + ENTITY_COUNT != VOCAB_SIZE:
    raise RuntimeError("special and entity tokens must total exactly VOCAB_SIZE")


@dataclass(frozen=True)
class SampleDebugInfo:
    """Generation-only semantic data that is not stored in the full dataset."""

    chains: Tuple[Tuple[int, ...], ...]
    mapping: Tuple[Tuple[int, int], ...]
    context_record_order: Tuple[Tuple[int, int], ...]
    query_order: Tuple[Tuple[int, int, int], ...]


def configuration_name(n_pairs: int, hop_length: int) -> str:
    validate_configuration(n_pairs, hop_length)
    return f"N{int(n_pairs)}-H{int(hop_length)}-V{VOCAB_SIZE}"


def validate_configuration(n_pairs: int, hop_length: int) -> None:
    n_pairs = int(n_pairs)
    hop_length = int(hop_length)
    if n_pairs <= 0:
        raise ValueError("N must be positive")
    if hop_length <= 0:
        raise ValueError("H must be positive")
    if n_pairs % hop_length:
        raise ValueError(f"N must be divisible by H; got N={n_pairs}, H={hop_length}")
    required_nodes = n_pairs + n_pairs // hop_length
    if required_nodes > ENTITY_COUNT:
        raise ValueError(
            f"N={n_pairs}, H={hop_length} requires {required_nodes} unique nodes, "
            f"but only {ENTITY_COUNT} entity tokens are available"
        )


def _derive_seed(*components: object) -> int:
    payload = "\0".join(str(component) for component in components).encode("utf-8")
    digest = hashlib.sha256(b"zoology-multihop-v1\0" + payload).digest()
    return int.from_bytes(digest[:16], "big")


def _encode_kv_record(key: int, value: int) -> List[int]:
    return [KV_START_ID, key, KV_SEPARATOR_ID, value, KV_END_ID]


def _encode_query_record(key: int) -> List[int]:
    return [QUERY_START_ID, key, QUERY_END_ID]


def _generate_sample_with_debug(
    n_pairs: int,
    hop_length: int,
    *,
    seed: int,
    sample_id: int = 0,
    validate: bool = True,
) -> Tuple[Dict[str, object], SampleDebugInfo]:
    validate_configuration(n_pairs, hop_length)
    n_pairs = int(n_pairs)
    hop_length = int(hop_length)
    chain_count = n_pairs // hop_length
    node_count = chain_count * (hop_length + 1)

    node_rng = random.Random(_derive_seed(seed, "nodes"))
    context_rng = random.Random(_derive_seed(seed, "context-order"))
    query_rng = random.Random(_derive_seed(seed, "query-order"))

    sampled_nodes = node_rng.sample(range(ENTITY_START_ID, ENTITY_STOP_ID), node_count)
    chains = tuple(
        tuple(sampled_nodes[offset : offset + hop_length + 1])
        for offset in range(0, node_count, hop_length + 1)
    )

    mapping = [
        (chain[position], chain[position + 1])
        for chain in chains
        for position in range(hop_length)
    ]
    context_records = list(mapping)
    context_rng.shuffle(context_records)

    queries = [
        (chain[position], chain[-1], hop_length - position)
        for chain in chains
        for position in range(hop_length)
    ]
    query_rng.shuffle(queries)

    context_input_ids = [BOS_ID, CONTEXT_START_ID]
    for key, value in context_records:
        context_input_ids.extend(_encode_kv_record(key, value))
    context_input_ids.extend((CONTEXT_END_ID, EOS_ID))

    query_input_ids = [BOS_ID]
    for key, _, _ in queries:
        query_input_ids.extend(_encode_query_record(key))
    query_input_ids.append(EOS_ID)

    sample: Dict[str, object] = {
        "sample_id": int(sample_id),
        "context_input_ids": context_input_ids,
        "query_input_ids": query_input_ids,
        "targets": [target for _, target, _ in queries],
        "hop_distances": [distance for _, _, distance in queries],
    }
    debug = SampleDebugInfo(
        chains=chains,
        mapping=tuple(mapping),
        context_record_order=tuple(context_records),
        query_order=tuple(queries),
    )
    if validate:
        validate_sample(sample, n_pairs, hop_length, expected_mapping=dict(mapping))
    return sample, debug


def generate_sample(
    n_pairs: int,
    hop_length: int,
    *,
    seed: int,
    sample_id: int = 0,
    validate: bool = True,
) -> Dict[str, object]:
    """Generate one deterministic sample from a sample-specific seed."""

    sample, _ = _generate_sample_with_debug(
        n_pairs,
        hop_length,
        seed=seed,
        sample_id=sample_id,
        validate=validate,
    )
    return sample


def generate_split_records(
    n_pairs: int,
    hop_length: int,
    num_samples: int,
    split_seed: int,
    split_name: str,
    validate: bool = True,
) -> Iterator[Dict[str, object]]:
    """Yield a deterministic split without retaining generated samples."""

    validate_configuration(n_pairs, hop_length)
    if split_name not in SPLIT_NAMES:
        raise ValueError(f"split_name must be one of {SPLIT_NAMES}; got {split_name!r}")
    if int(num_samples) < 0:
        raise ValueError("num_samples must be non-negative")
    for sample_id in range(int(num_samples)):
        sample_seed = _derive_seed(
            "split",
            split_name,
            int(split_seed),
            int(n_pairs),
            int(hop_length),
            sample_id,
        )
        yield generate_sample(
            n_pairs,
            hop_length,
            seed=sample_seed,
            sample_id=sample_id,
            validate=validate,
        )


def parse_context_records(context_input_ids: Sequence[int]) -> List[Tuple[int, int]]:
    ids = list(context_input_ids)
    if len(ids) < 4 or ids[:2] != [BOS_ID, CONTEXT_START_ID]:
        raise ValueError("context must start with [BOS, CONTEXT_START]")
    if ids[-2:] != [CONTEXT_END_ID, EOS_ID]:
        raise ValueError("context must end with [CONTEXT_END, EOS]")
    body = ids[2:-2]
    if len(body) % KV_RECORD_WIDTH:
        raise ValueError("context KV records must all have width 5")

    records = []
    for offset in range(0, len(body), KV_RECORD_WIDTH):
        record = body[offset : offset + KV_RECORD_WIDTH]
        if (
            record[0] != KV_START_ID
            or record[2] != KV_SEPARATOR_ID
            or record[4] != KV_END_ID
        ):
            raise ValueError(f"invalid KV record at record index {offset // KV_RECORD_WIDTH}")
        records.append((record[1], record[3]))
    return records


def parse_query_keys(query_input_ids: Sequence[int]) -> List[int]:
    ids = list(query_input_ids)
    if len(ids) < 2 or ids[0] != BOS_ID or ids[-1] != EOS_ID:
        raise ValueError("query collection must be bounded by BOS and EOS")
    body = ids[1:-1]
    if len(body) % QUERY_RECORD_WIDTH:
        raise ValueError("query records must all have width 3")

    keys = []
    for offset in range(0, len(body), QUERY_RECORD_WIDTH):
        record = body[offset : offset + QUERY_RECORD_WIDTH]
        if record[0] != QUERY_START_ID or record[2] != QUERY_END_ID:
            raise ValueError(f"invalid query record at record index {offset // QUERY_RECORD_WIDTH}")
        keys.append(record[1])
    return keys


def _require_entity(entity: int, field_name: str) -> None:
    if not ENTITY_START_ID <= int(entity) < ENTITY_STOP_ID:
        raise ValueError(
            f"{field_name} must use the shared entity pool "
            f"[{ENTITY_START_ID}, {ENTITY_STOP_ID}); got {entity}"
        )


def validate_sample(
    sample: Mapping[str, object],
    n_pairs: int,
    hop_length: int,
    *,
    expected_mapping: Optional[Mapping[int, int]] = None,
) -> None:
    """Validate structural and semantic invariants of one stored sample."""

    validate_configuration(n_pairs, hop_length)
    n_pairs = int(n_pairs)
    hop_length = int(hop_length)
    chain_count = n_pairs // hop_length

    required_fields = {
        "sample_id",
        "context_input_ids",
        "query_input_ids",
        "targets",
        "hop_distances",
    }
    missing = required_fields.difference(sample)
    if missing:
        raise ValueError(f"sample is missing fields: {sorted(missing)}")

    context_input_ids = list(sample["context_input_ids"])
    query_input_ids = list(sample["query_input_ids"])
    targets = list(sample["targets"])
    hop_distances = list(sample["hop_distances"])
    for field_name, tokens in (
        ("context_input_ids", context_input_ids),
        ("query_input_ids", query_input_ids),
        ("targets", targets),
    ):
        if any(not 0 <= int(token) < VOCAB_SIZE for token in tokens):
            raise ValueError(f"{field_name} contains a token outside [0, {VOCAB_SIZE})")

    records = parse_context_records(context_input_ids)
    if len(records) != n_pairs:
        raise ValueError(f"expected {n_pairs} KV records; found {len(records)}")
    keys = [key for key, _ in records]
    values = [value for _, value in records]
    for key in keys:
        _require_entity(key, "key")
    for value in values:
        _require_entity(value, "value")
    if len(set(keys)) != n_pairs:
        raise ValueError("KV keys must be unique")
    if len(set(values)) != n_pairs:
        raise ValueError("KV values must be unique")

    mapping = dict(records)
    if expected_mapping is not None and mapping != dict(expected_mapping):
        raise ValueError("shuffled context records do not reconstruct the expected mapping")

    key_set = set(keys)
    value_set = set(values)
    roots = key_set - value_set
    terminals = value_set - key_set
    if len(roots) != chain_count or len(terminals) != chain_count:
        raise ValueError(
            f"expected {chain_count} chain roots and terminals; "
            f"found {len(roots)} roots and {len(terminals)} terminals"
        )

    visited_keys = set()
    visited_nodes = set()
    terminal_by_key: Dict[int, Tuple[int, int]] = {}
    for root in roots:
        current = root
        path = [current]
        local_nodes = {current}
        path_keys = []
        while current in mapping:
            path_keys.append(current)
            next_node = mapping[current]
            if next_node in local_nodes:
                raise ValueError("chains must be acyclic")
            local_nodes.add(next_node)
            path.append(next_node)
            current = next_node
        if len(path_keys) != hop_length:
            raise ValueError(
                f"every chain must contain exactly {hop_length} edges; "
                f"found {len(path_keys)}"
            )
        if visited_nodes.intersection(local_nodes):
            raise ValueError("different chains must be node-disjoint")
        visited_nodes.update(local_nodes)
        visited_keys.update(path_keys)
        terminal = path[-1]
        for position, key in enumerate(path_keys):
            terminal_by_key[key] = (terminal, hop_length - position)

    if visited_keys != key_set or visited_nodes != key_set | value_set:
        raise ValueError("KV records do not form the required complete disjoint chains")

    query_keys = parse_query_keys(query_input_ids)
    if len(query_keys) != n_pairs or len(set(query_keys)) != n_pairs:
        raise ValueError("queries must contain every key exactly once")
    if set(query_keys) != key_set:
        raise ValueError("query keys must exactly match the KV keys")
    if len(targets) != n_pairs or len(hop_distances) != n_pairs:
        raise ValueError("targets and hop_distances must align one-to-one with queries")
    query_token_set = set(query_input_ids)
    if any(target in query_token_set for target in targets):
        raise ValueError("target entities must not occur in query_input_ids")

    for key, target, distance in zip(query_keys, targets, hop_distances):
        _require_entity(target, "target")
        expected_target, expected_distance = terminal_by_key[key]
        if target != expected_target:
            raise ValueError(f"incorrect terminal target for query key {key}")
        if int(distance) != expected_distance:
            raise ValueError(f"incorrect hop distance for query key {key}")

    expected_depth_counts = {depth: chain_count for depth in range(1, hop_length + 1)}
    if dict(Counter(int(distance) for distance in hop_distances)) != expected_depth_counts:
        raise ValueError("each hop distance must occur exactly N/H times")


def entity_name(token_id: int) -> str:
    _require_entity(token_id, "entity")
    return f"E{int(token_id):04d}"


def decode_sample(sample: Mapping[str, object], max_items: int = 12) -> str:
    """Render a compact semantic view without implying that entities are text."""

    records = parse_context_records(sample["context_input_ids"])
    query_keys = parse_query_keys(sample["query_input_ids"])
    targets = list(sample["targets"])
    distances = list(sample["hop_distances"])

    def clipped(lines: List[str]) -> List[str]:
        if len(lines) <= max_items:
            return lines
        return lines[:max_items] + [f"  ... ({len(lines) - max_items} more)"]

    record_lines = clipped(
        [f"  {entity_name(key)} -> {entity_name(value)}" for key, value in records]
    )
    query_lines = clipped(
        [
            f"  {entity_name(key)} => {entity_name(target)} ({distance} hop{'s' if distance != 1 else ''})"
            for key, target, distance in zip(query_keys, targets, distances)
        ]
    )
    return "\n".join(
        [
            f"sample_id={sample['sample_id']}",
            "context records (stored shuffled order):",
            *record_lines,
            "queries (stored independently shuffled order; targets shown only for inspection):",
            *query_lines,
        ]
    )
