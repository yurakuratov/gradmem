"""Deterministic raw generator for multihop associative retrieval.

Rows contain only base-16 entity digits and aligned semantic arrays. Model-side
structural formatting and context segmentation deliberately live in the
training collator, not in this dataset module.
"""

from __future__ import annotations

import hashlib
import random
from collections import Counter
from dataclasses import dataclass
from typing import Dict, Iterator, List, Mapping, Optional, Sequence, Tuple


ENTITY_ALPHABET_SIZE = 16
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
RAW_ENTITY_FIELDS = ("context_keys", "context_values", "query_keys", "targets")

Entity = Tuple[int, ...]


@dataclass(frozen=True)
class SampleDebugInfo:
    """Generation-only integer-node data omitted from Hub rows."""

    chains: Tuple[Tuple[int, ...], ...]
    mapping: Tuple[Tuple[int, int], ...]
    context_record_order: Tuple[Tuple[int, int], ...]
    query_order: Tuple[Tuple[int, int, int], ...]


def validate_configuration(n_pairs: int, hop_length: int) -> None:
    n_pairs = int(n_pairs)
    hop_length = int(hop_length)
    if n_pairs <= 0:
        raise ValueError("N must be positive")
    if hop_length <= 0:
        raise ValueError("H must be positive")
    if n_pairs % hop_length:
        raise ValueError(f"N must be divisible by H; got N={n_pairs}, H={hop_length}")


def node_count_for_configuration(n_pairs: int, hop_length: int) -> int:
    validate_configuration(n_pairs, hop_length)
    return int(n_pairs) + int(n_pairs) // int(hop_length)


def entity_length_for_configuration(n_pairs: int, hop_length: int) -> int:
    """Return minimal positive L with ``16**L >= N + N/H`` using integers."""

    required_nodes = node_count_for_configuration(n_pairs, hop_length)
    entity_length = 1
    capacity = ENTITY_ALPHABET_SIZE
    while capacity < required_nodes:
        entity_length += 1
        capacity *= ENTITY_ALPHABET_SIZE
    return entity_length


def configuration_name(n_pairs: int, hop_length: int) -> str:
    validate_configuration(n_pairs, hop_length)
    return f"N{int(n_pairs)}-H{int(hop_length)}-V{ENTITY_ALPHABET_SIZE}"


def encode_entity(node_id: int, entity_length: int) -> List[int]:
    """Encode an integer as fixed-width, leading-zero-preserving base 16."""

    node_id = int(node_id)
    entity_length = int(entity_length)
    if entity_length < 1:
        raise ValueError("entity_length must be positive")
    capacity = ENTITY_ALPHABET_SIZE**entity_length
    if not 0 <= node_id < capacity:
        raise ValueError(f"node_id must be in [0, {capacity}); got {node_id}")
    digits = [0] * entity_length
    remainder = node_id
    for position in range(entity_length - 1, -1, -1):
        remainder, digits[position] = divmod(remainder, ENTITY_ALPHABET_SIZE)
    return digits


def decode_entity(digits: Sequence[int]) -> int:
    if not digits:
        raise ValueError("an entity must contain at least one digit")
    node_id = 0
    for digit in digits:
        digit = int(digit)
        if not 0 <= digit < ENTITY_ALPHABET_SIZE:
            raise ValueError(f"entity digit must be in [0, 16); got {digit}")
        node_id = node_id * ENTITY_ALPHABET_SIZE + digit
    return node_id


def _derive_seed(*components: object) -> int:
    payload = "\0".join(str(component) for component in components).encode("utf-8")
    digest = hashlib.sha256(b"ar-multihop-v1\0" + payload).digest()
    return int.from_bytes(digest[:16], "big")


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
    node_count = node_count_for_configuration(n_pairs, hop_length)
    entity_length = entity_length_for_configuration(n_pairs, hop_length)
    capacity = ENTITY_ALPHABET_SIZE**entity_length

    node_rng = random.Random(_derive_seed(seed, "nodes"))
    context_rng = random.Random(_derive_seed(seed, "context-order"))
    query_rng = random.Random(_derive_seed(seed, "query-order"))

    sampled_nodes = node_rng.sample(range(capacity), node_count)
    chains = tuple(
        tuple(sampled_nodes[offset : offset + hop_length + 1])
        for offset in range(0, node_count, hop_length + 1)
    )
    if len(chains) != chain_count:
        raise RuntimeError("internal chain construction error")

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

    sample: Dict[str, object] = {
        "sample_id": int(sample_id),
        "context_keys": [
            encode_entity(key, entity_length) for key, _ in context_records
        ],
        "context_values": [
            encode_entity(value, entity_length) for _, value in context_records
        ],
        "query_keys": [encode_entity(key, entity_length) for key, _, _ in queries],
        "targets": [encode_entity(target, entity_length) for _, target, _ in queries],
        "hop_distances": [distance for _, _, distance in queries],
    }
    debug = SampleDebugInfo(
        chains=chains,
        mapping=tuple(mapping),
        context_record_order=tuple(context_records),
        query_order=tuple(queries),
    )
    if validate:
        expected_mapping = {
            tuple(encode_entity(key, entity_length)): tuple(
                encode_entity(value, entity_length)
            )
            for key, value in mapping
        }
        validate_sample(
            sample,
            n_pairs,
            hop_length,
            expected_mapping=expected_mapping,
        )
    return sample, debug


def generate_sample(
    n_pairs: int,
    hop_length: int,
    *,
    seed: int,
    sample_id: int = 0,
    validate: bool = True,
) -> Dict[str, object]:
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
    """Yield a deterministic split without retaining generated rows."""

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


def _entities_from_field(
    sample: Mapping[str, object],
    field_name: str,
    expected_count: int,
    entity_length: int,
) -> List[Entity]:
    rows = list(sample[field_name])
    if len(rows) != expected_count:
        raise ValueError(
            f"{field_name} must have shape [{expected_count}, {entity_length}]"
        )
    entities = []
    for row in rows:
        digits = tuple(int(digit) for digit in row)
        if len(digits) != entity_length:
            raise ValueError(
                f"{field_name} must have shape [{expected_count}, {entity_length}]"
            )
        if any(not 0 <= digit < ENTITY_ALPHABET_SIZE for digit in digits):
            raise ValueError(f"{field_name} contains a symbol outside [0, 16)")
        entities.append(digits)
    return entities


def validate_sample(
    sample: Mapping[str, object],
    n_pairs: int,
    hop_length: int,
    *,
    expected_mapping: Optional[Mapping[Entity, Entity]] = None,
) -> None:
    """Validate raw shapes and complete multihop chain/query semantics."""

    validate_configuration(n_pairs, hop_length)
    n_pairs = int(n_pairs)
    hop_length = int(hop_length)
    chain_count = n_pairs // hop_length
    entity_length = entity_length_for_configuration(n_pairs, hop_length)
    required_fields = {"sample_id", *RAW_ENTITY_FIELDS, "hop_distances"}
    missing = required_fields.difference(sample)
    if missing:
        raise ValueError(f"sample is missing fields: {sorted(missing)}")
    if isinstance(sample["sample_id"], (list, tuple, dict)):
        raise ValueError("sample_id must be a scalar integer")
    int(sample["sample_id"])

    context_keys = _entities_from_field(
        sample, "context_keys", n_pairs, entity_length
    )
    context_values = _entities_from_field(
        sample, "context_values", n_pairs, entity_length
    )
    query_keys = _entities_from_field(sample, "query_keys", n_pairs, entity_length)
    targets = _entities_from_field(sample, "targets", n_pairs, entity_length)
    hop_distances = [int(distance) for distance in sample["hop_distances"]]
    if len(hop_distances) != n_pairs:
        raise ValueError(f"hop_distances must have shape [{n_pairs}]")

    if len(set(context_keys)) != n_pairs:
        raise ValueError("context keys must be unique")
    if len(set(context_values)) != n_pairs:
        raise ValueError("context values must be unique")
    mapping = dict(zip(context_keys, context_values))
    if expected_mapping is not None and mapping != dict(expected_mapping):
        raise ValueError("shuffled context pairs do not reconstruct the expected mapping")

    key_set = set(context_keys)
    value_set = set(context_values)
    roots = key_set - value_set
    terminals = value_set - key_set
    if len(roots) != chain_count or len(terminals) != chain_count:
        raise ValueError(
            f"expected {chain_count} roots and terminals; found "
            f"{len(roots)} roots and {len(terminals)} terminals"
        )

    visited_keys = set()
    visited_nodes = set()
    terminal_by_key: Dict[Entity, Tuple[Entity, int]] = {}
    for root in roots:
        current = root
        local_nodes = {current}
        path_keys = []
        while current in mapping:
            path_keys.append(current)
            next_node = mapping[current]
            if next_node in local_nodes:
                raise ValueError("chains must be acyclic")
            local_nodes.add(next_node)
            current = next_node
        if len(path_keys) != hop_length:
            raise ValueError(f"every chain must contain exactly {hop_length} edges")
        if visited_nodes.intersection(local_nodes):
            raise ValueError("different chains must be node-disjoint")
        visited_nodes.update(local_nodes)
        visited_keys.update(path_keys)
        terminal = current
        for position, key in enumerate(path_keys):
            terminal_by_key[key] = (terminal, hop_length - position)

    if visited_keys != key_set or visited_nodes != key_set | value_set:
        raise ValueError("context pairs do not form complete node-disjoint chains")
    if len(set(query_keys)) != n_pairs or set(query_keys) != key_set:
        raise ValueError("query_keys must contain every context key exactly once")

    for key, target, distance in zip(query_keys, targets, hop_distances):
        expected_target, expected_distance = terminal_by_key[key]
        if target != expected_target:
            raise ValueError(f"incorrect terminal target for query key {key}")
        if distance != expected_distance:
            raise ValueError(f"incorrect hop distance for query key {key}")
    expected_depth_counts = {depth: chain_count for depth in range(1, hop_length + 1)}
    if dict(Counter(hop_distances)) != expected_depth_counts:
        raise ValueError("each hop distance must occur exactly N/H times")


def format_entity(digits: Sequence[int]) -> str:
    return "".join(f"{int(digit):X}" for digit in digits)


def decode_sample(sample: Mapping[str, object], max_items: int = 12) -> str:
    """Render a compact human-readable view of one raw sample."""

    context = list(zip(sample["context_keys"], sample["context_values"]))
    queries = list(
        zip(sample["query_keys"], sample["targets"], sample["hop_distances"])
    )

    def clipped(lines: List[str]) -> List[str]:
        if len(lines) <= max_items:
            return lines
        return lines[:max_items] + [f"  ... ({len(lines) - max_items} more)"]

    return "\n".join(
        [
            f"sample_id={sample['sample_id']}",
            "context pairs (stored shuffled order):",
            *clipped(
                [
                    f"  {format_entity(key)} -> {format_entity(value)}"
                    for key, value in context
                ]
            ),
            "queries (independently shuffled; targets shown only for inspection):",
            *clipped(
                [
                    f"  {format_entity(key)} => {format_entity(target)} ({distance} hop{'s' if distance != 1 else ''})"
                    for key, target, distance in queries
                ]
            ),
        ]
    )
