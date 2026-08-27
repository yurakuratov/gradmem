"""Zoology multi-query associative recall data generation.

Adapted from HazyResearch/zoology:
https://github.com/HazyResearch/zoology/blob/main/zoology/data/multiquery_ar.py

Upstream commit: 1ad20d193b6113cae1e8f3c655c300d7b4b3f4bb

``multiquery_ar`` follows the upstream task layout while adding explicit
four-token framing around each KV pair. ``dense_multiquery_ar`` defaults to a
local vectorized, uniformly permuted dense implementation. It also provides a
vectorized power-law mode and a ``zoology`` mode that preserves the upstream
query-ordering behavior.
"""

import json
from pathlib import Path

import numpy as np
import torch


ZOOLOGY_MQAR_SOURCE = (
    "https://github.com/HazyResearch/zoology/blob/"
    "1ad20d193b6113cae1e8f3c655c300d7b4b3f4bb/zoology/data/multiquery_ar.py"
)
ZOOLOGY_MQAR_COMMIT = "1ad20d193b6113cae1e8f3c655c300d7b4b3f4bb"
QUERY_SAMPLING_UNIFORM = "uniform"
QUERY_SAMPLING_POWER_LAW = "power_law"
QUERY_SAMPLING_ZOOLOGY = "zoology"
QUERY_SAMPLING_CHOICES = (
    QUERY_SAMPLING_UNIFORM,
    QUERY_SAMPLING_POWER_LAW,
    QUERY_SAMPLING_ZOOLOGY,
)
PAIR_OPEN_TOKEN = 0
PAIR_CLOSE_TOKEN = 1
FIRST_DATA_TOKEN = 2
PAIR_WIDTH = 4


class MQARDataset(torch.utils.data.Dataset):
    def __init__(self, inputs, labels, slices=None):
        self.inputs = inputs
        self.labels = labels
        self.slices = slices or {}

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        return {
            "input_ids": self.inputs[index],
            "labels": self.labels[index],
        }


def load_saved_mqar_datasets(data_path):
    """Load a DatasetDict and metadata produced by prepare_mqar_datasets.py."""
    from datasets import load_from_disk
    from datasets.features import features

    # Older datasets versions can define List but omit it from the registry
    # used to deserialize dataset_info.json. Keep the saved data untouched.
    if 'List' not in features._FEATURE_TYPES:
        features._FEATURE_TYPES['List'] = getattr(features, 'List', features.Sequence)

    data_path = Path(data_path)
    metadata_path = data_path / 'mqar_metadata.json'
    if not metadata_path.is_file():
        raise FileNotFoundError(f'MQAR metadata does not exist: {metadata_path}')
    dataset_dict = load_from_disk(str(data_path))
    if not {'train', 'valid'}.issubset(dataset_dict):
        raise ValueError(f'MQAR dataset must contain train and valid splits: {data_path}')
    with metadata_path.open() as metadata_file:
        metadata = json.load(metadata_file)
    for split in ('train', 'valid'):
        missing_columns = {'input_ids', 'labels'} - set(dataset_dict[split].column_names)
        if missing_columns:
            raise ValueError(f'MQAR {split} split is missing columns: {sorted(missing_columns)}')
        dataset_dict[split].set_format(type='torch', columns=['input_ids', 'labels'])
    return dataset_dict['train'], dataset_dict['valid'], metadata


def _add_context_noise(
    dataset: MQARDataset,
    *,
    vocab_size: int,
    noise_lvl: float,
    seed: int,
) -> MQARDataset:
    """Insert independent key/value noise between useful KV pairs.

    ``noise_lvl`` is the number of added noise tokens relative to the clean
    context length, rounded to the nearest token. A positive level always
    adds at least one noise token. Noise is inserted between framed KV pairs.
    """
    if not np.isfinite(noise_lvl) or not 0.0 <= noise_lvl <= 1.0:
        raise ValueError(f"mqar_noise_lvl must be finite and in [0, 1], got {noise_lvl!r}.")

    context_size = int(dataset.slices.get("context_size", 0))
    if context_size <= 0 or context_size >= dataset.inputs.shape[1]:
        raise ValueError(f"Invalid MQAR context_size={context_size} for generated data.")
    if context_size % PAIR_WIDTH:
        raise ValueError(f"MQAR context_size must be divisible by {PAIR_WIDTH}.")

    noise_tokens = 0 if noise_lvl == 0.0 else max(1, int(round(context_size * noise_lvl)))
    slices = {
        **dataset.slices,
        "clean_context_size": context_size,
        "context_size": context_size,
        "noise_tokens": noise_tokens,
        "mqar_noise_lvl": float(noise_lvl),
        "noise_layout": "noise|< K V >|noise|< K V >|...|noise",
        "pair_open_token": PAIR_OPEN_TOKEN,
        "pair_close_token": PAIR_CLOSE_TOKEN,
    }
    if noise_tokens == 0:
        return MQARDataset(dataset.inputs, dataset.labels, slices=slices)

    num_examples = dataset.inputs.size(0)
    rng = np.random.default_rng(seed)
    key_vocab_size = vocab_size // 2
    num_key_noise_tokens = noise_tokens // 2
    num_value_noise_tokens = noise_tokens - num_key_noise_tokens
    noise_keys = rng.integers(
        FIRST_DATA_TOKEN,
        key_vocab_size,
        size=(num_examples, num_key_noise_tokens),
        dtype=np.int64,
    )
    noise_values = rng.integers(
        key_vocab_size,
        vocab_size,
        size=(num_examples, num_value_noise_tokens),
        dtype=np.int64,
    )
    noise = np.concatenate([noise_keys, noise_values], axis=1)
    noise_order = np.argsort(rng.random((num_examples, noise_tokens)), axis=1)
    noise = np.take_along_axis(noise, noise_order, axis=1)

    clean_context = dataset.inputs[:, :context_size]
    query_inputs = dataset.inputs[:, context_size:]
    query_labels = dataset.labels[:, context_size:]
    num_pairs = context_size // PAIR_WIDTH
    num_gaps = num_pairs + 1
    if noise_tokens == 1:
        gap_indices = np.array([rng.integers(num_gaps)])
    else:
        interior_noise_tokens = (
            rng.integers(1, num_gaps - 1, size=noise_tokens - 2)
            if num_gaps > 2 else
            np.empty(0, dtype=np.int64)
        )
        gap_indices = np.concatenate([
            np.array([0, num_gaps - 1]),
            interior_noise_tokens,
        ])
    gap_order = np.argsort(gap_indices, kind='stable')
    gap_counts = np.bincount(gap_indices, minlength=num_gaps)
    sorted_noise = torch.from_numpy(noise[:, gap_order])

    context_parts = []
    noise_offset = 0
    for pair_index in range(num_pairs):
        gap_count = int(gap_counts[pair_index])
        context_parts.append(sorted_noise[:, noise_offset:noise_offset + gap_count])
        pair_start = pair_index * PAIR_WIDTH
        context_parts.append(clean_context[:, pair_start:pair_start + PAIR_WIDTH])
        noise_offset += gap_count
    final_gap_count = int(gap_counts[-1])
    context_parts.append(sorted_noise[:, noise_offset:noise_offset + final_gap_count])

    noisy_context = torch.cat(context_parts, dim=1)
    noisy_context_size = noisy_context.shape[1]
    inputs = torch.cat([noisy_context, query_inputs], dim=1)
    context_labels = torch.full(
        (num_examples, noisy_context_size),
        -100,
        dtype=dataset.labels.dtype,
    )
    labels = torch.cat([context_labels, query_labels], dim=1)
    slices.update({
        "context_size": noisy_context_size,
        "input_seq_len": int(inputs.shape[1]),
        "noise_gap_counts": gap_counts.tolist(),
        "pair_open_token": PAIR_OPEN_TOKEN,
        "pair_close_token": PAIR_CLOSE_TOKEN,
    })
    return MQARDataset(inputs, labels, slices=slices)


def multiquery_ar(
    vocab_size: int,
    num_examples: int,
    input_seq_len: int,
    seed: int,
    power_a: float = 0.01,
    num_kv_pairs: int = 8,
    num_passes: int = 1,
    random_non_queries: bool = True,
    include_slices: bool = True,
    mqar_noise_lvl: float = 0.0,
    **kwargs,
) -> MQARDataset:
    """Generate the synthetic MQAR task from the Zoology paper."""
    assert input_seq_len % 2 == 0, "input_seq_len must be even"
    assert vocab_size > input_seq_len
    context_size = num_kv_pairs * PAIR_WIDTH * num_passes
    assert context_size + num_kv_pairs * 2 <= input_seq_len

    np.random.seed(seed)

    # create keys so that each key is present exactly once in each example
    key_vocab_size = vocab_size // 2
    key_choices = np.arange(FIRST_DATA_TOKEN, key_vocab_size)
    value_choices = np.arange(key_vocab_size, vocab_size)

    keys_unshuffled = np.tile(key_choices, (num_examples, 1))
    keys = np.apply_along_axis(
        np.random.choice, 1, keys_unshuffled, replace=False, size=num_kv_pairs
    )

    values_unshuffled = np.tile(value_choices, (num_examples, 1))
    values = np.apply_along_axis(
        np.random.choice, 1, values_unshuffled, replace=False, size=num_kv_pairs
    )

    # create sequences
    kvs = np.empty((num_examples, num_kv_pairs * PAIR_WIDTH), dtype=np.int64)
    kvs[:, 0::PAIR_WIDTH] = PAIR_OPEN_TOKEN
    kvs[:, 1::PAIR_WIDTH] = keys
    kvs[:, 2::PAIR_WIDTH] = values
    kvs[:, 3::PAIR_WIDTH] = PAIR_CLOSE_TOKEN
    kvs = np.tile(kvs, (1, num_passes))

    # compute power law
    space = (input_seq_len - context_size) // 2
    p = power_a * np.arange(1, space + 1) ** (power_a - 1)
    p = p / p.sum()

    x = np.stack([np.arange(space, dtype=int)] * num_examples)
    gaps = np.apply_along_axis(
        np.random.choice,
        axis=1,
        arr=x,
        replace=False,
        p=p,
        size=num_kv_pairs,
    )

    # queries and answers
    queries = np.zeros(
        (num_examples, input_seq_len - context_size + 1), dtype=np.int64
    )
    np.put_along_axis(queries, (gaps * 2), values=keys, axis=1)
    examples = np.concatenate([kvs, queries], axis=1)

    labels = np.full((num_examples, input_seq_len + 1), -100, dtype=np.int64)
    np.put_along_axis(
        labels, (gaps * 2) + context_size + 1, values=values, axis=1
    )

    inputs, labels = torch.tensor(examples[:, :-1]), torch.tensor(labels[:, 1:])

    # Replace only query filler zeros; preserve the pair-open framing tokens.
    if random_non_queries:
        non_query_mask = torch.zeros_like(inputs, dtype=torch.bool)
        non_query_mask[:, context_size:] = inputs[:, context_size:] == PAIR_OPEN_TOKEN
        inputs[non_query_mask] = torch.randint(
            FIRST_DATA_TOKEN, vocab_size, size=inputs.shape
        )[non_query_mask]
    dataset = MQARDataset(
        inputs,
        labels,
        slices={
            "num_kv_pairs": num_kv_pairs,
            "input_seq_len": input_seq_len,
            "num_passes": num_passes,
            "context_size": context_size,
        },
    )
    return _add_context_noise(
        dataset,
        vocab_size=vocab_size,
        noise_lvl=mqar_noise_lvl,
        seed=seed + 1,
    )


def _validate_dense_args(
    *,
    vocab_size: int,
    num_examples: int,
    input_seq_len: int,
    num_kv_pairs: int,
    num_passes: int,
    random_non_queries: bool,
    query_sampling: str,
    power_a: float,
):
    if query_sampling not in QUERY_SAMPLING_CHOICES:
        raise ValueError(
            f"Unsupported query_sampling={query_sampling!r}; "
            f"expected one of {QUERY_SAMPLING_CHOICES}."
        )
    if num_examples < 0:
        raise ValueError("num_examples must be non-negative.")
    if num_kv_pairs <= 0:
        raise ValueError("num_kv_pairs must be positive.")
    if num_passes != 1:
        raise ValueError("Dense MQAR currently requires num_passes=1.")
    if random_non_queries:
        raise ValueError(
            "Dense MQAR has no non-query filler positions; random_non_queries must be false."
        )
    if query_sampling == QUERY_SAMPLING_POWER_LAW and (
        not np.isfinite(power_a) or power_a <= 0
    ):
        raise ValueError(
            "power_a must be finite and strictly positive for query_sampling='power_law'; "
            f"got {power_a!r}."
        )

    context_size = num_kv_pairs * PAIR_WIDTH
    dense_seq_len = context_size + num_kv_pairs
    if input_seq_len != dense_seq_len:
        raise ValueError(
            "Dense MQAR input_seq_len must equal 5 * num_kv_pairs: "
            f"input_seq_len={input_seq_len}, expected={dense_seq_len}."
        )

    key_vocab_size = vocab_size // 2
    num_key_choices = key_vocab_size - FIRST_DATA_TOKEN
    num_value_choices = vocab_size - key_vocab_size
    if num_kv_pairs > num_key_choices:
        raise ValueError(
            "num_kv_pairs exceeds the number of distinct key tokens: "
            f"num_kv_pairs={num_kv_pairs}, key_choices={num_key_choices}."
        )
    if num_kv_pairs > num_value_choices:
        raise ValueError(
            "num_kv_pairs exceeds the number of distinct value tokens: "
            f"num_kv_pairs={num_kv_pairs}, value_choices={num_value_choices}."
        )


def _sample_uniform_unique_rows(
    rng: np.random.Generator,
    *,
    low: int,
    high: int,
    num_rows: int,
    row_width: int,
) -> np.ndarray:
    """Sample uniform ordered rows without replacement using bounded memory."""
    if num_rows == 0:
        return np.empty((0, row_width), dtype=np.int64)

    population_size = high - low
    log_unique_probability = sum(
        np.log1p(-offset / population_size) for offset in range(row_width)
    )
    if log_unique_probability < np.log(0.1):
        choices = np.arange(low, high, dtype=np.int64)
        return np.stack(
            [rng.choice(choices, size=row_width, replace=False) for _ in range(num_rows)]
        )

    samples = rng.integers(
        low,
        high,
        size=(num_rows, row_width),
        dtype=np.int64,
    )
    while True:
        ordered = np.sort(samples, axis=1)
        duplicate_rows = np.any(np.diff(ordered, axis=1) == 0, axis=1)
        num_duplicate_rows = int(duplicate_rows.sum())
        if num_duplicate_rows == 0:
            return samples
        samples[duplicate_rows] = rng.integers(
            low,
            high,
            size=(num_duplicate_rows, row_width),
            dtype=np.int64,
        )


def _sample_vectorized_key_values(
    *,
    rng: np.random.Generator,
    vocab_size: int,
    num_examples: int,
    num_kv_pairs: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    key_vocab_size = vocab_size // 2
    keys = _sample_uniform_unique_rows(
        rng,
        low=FIRST_DATA_TOKEN,
        high=key_vocab_size,
        num_rows=num_examples,
        row_width=num_kv_pairs,
    )
    values = _sample_uniform_unique_rows(
        rng,
        low=key_vocab_size,
        high=vocab_size,
        num_rows=num_examples,
        row_width=num_kv_pairs,
    )

    context_size = num_kv_pairs * PAIR_WIDTH
    context = np.empty((num_examples, context_size), dtype=np.int64)
    context[:, 0::PAIR_WIDTH] = PAIR_OPEN_TOKEN
    context[:, 1::PAIR_WIDTH] = keys
    context[:, 2::PAIR_WIDTH] = values
    context[:, 3::PAIR_WIDTH] = PAIR_CLOSE_TOKEN
    return keys, values, context


def _build_vectorized_dense_dataset(
    *,
    context: np.ndarray,
    queries: np.ndarray,
    answers: np.ndarray,
    input_seq_len: int,
    num_kv_pairs: int,
    num_passes: int,
    query_sampling: str,
    query_distribution: str,
    effective_power_a: float | None,
) -> MQARDataset:
    context_size = num_kv_pairs * PAIR_WIDTH
    inputs_array = np.concatenate([context, queries], axis=1)
    labels_array = np.full_like(inputs_array, -100)
    labels_array[:, context_size:] = answers
    if not np.all(context[:, 0::PAIR_WIDTH] == PAIR_OPEN_TOKEN):
        raise RuntimeError("Dense MQAR pair-open framing tokens are malformed.")
    if not np.all(context[:, 3::PAIR_WIDTH] == PAIR_CLOSE_TOKEN):
        raise RuntimeError("Dense MQAR pair-close framing tokens are malformed.")

    return MQARDataset(
        torch.from_numpy(inputs_array),
        torch.from_numpy(labels_array),
        slices={
            "num_kv_pairs": num_kv_pairs,
            "input_seq_len": input_seq_len,
            "num_passes": num_passes,
            "context_size": context_size,
            "query_layout": "dense",
            "query_sampling": query_sampling,
            "query_distribution": query_distribution,
            "generator": "vectorized_dense",
            "effective_power_a": effective_power_a,
        },
    )


def _uniform_dense_multiquery_ar(
    *,
    vocab_size: int,
    num_examples: int,
    input_seq_len: int,
    seed: int,
    num_kv_pairs: int,
    num_passes: int,
) -> MQARDataset:
    rng = np.random.default_rng(seed)
    keys, values, context = _sample_vectorized_key_values(
        rng=rng,
        vocab_size=vocab_size,
        num_examples=num_examples,
        num_kv_pairs=num_kv_pairs,
    )

    # Sorting independent continuous scores gives a uniform random permutation
    # without allocating an array proportional to the vocabulary size.
    query_permutation = np.argsort(
        rng.random((num_examples, num_kv_pairs)),
        axis=1,
    )
    queries = np.take_along_axis(keys, query_permutation, axis=1)
    answers = np.take_along_axis(values, query_permutation, axis=1)

    return _build_vectorized_dense_dataset(
        context=context,
        queries=queries,
        answers=answers,
        input_seq_len=input_seq_len,
        num_kv_pairs=num_kv_pairs,
        num_passes=num_passes,
        query_sampling=QUERY_SAMPLING_UNIFORM,
        query_distribution="uniform_permutation",
        effective_power_a=None,
    )


def _power_law_dense_multiquery_ar(
    *,
    vocab_size: int,
    num_examples: int,
    input_seq_len: int,
    seed: int,
    power_a: float,
    num_kv_pairs: int,
    num_passes: int,
) -> MQARDataset:
    rng = np.random.default_rng(seed)
    keys, values, context = _sample_vectorized_key_values(
        rng=rng,
        vocab_size=vocab_size,
        num_examples=num_examples,
        num_kv_pairs=num_kv_pairs,
    )

    positions = np.arange(1, num_kv_pairs + 1, dtype=np.float64)
    weights = positions ** (power_a - 1)
    arrival_times = rng.exponential(
        scale=1.0 / weights,
        size=(num_examples, num_kv_pairs),
    )
    gaps = np.argsort(arrival_times, axis=1)

    # gaps[row, pair] is the destination query slot for that context pair,
    # matching upstream np.put_along_axis semantics.
    queries = np.empty_like(keys)
    answers = np.empty_like(values)
    np.put_along_axis(queries, gaps, keys, axis=1)
    np.put_along_axis(answers, gaps, values, axis=1)

    return _build_vectorized_dense_dataset(
        context=context,
        queries=queries,
        answers=answers,
        input_seq_len=input_seq_len,
        num_kv_pairs=num_kv_pairs,
        num_passes=num_passes,
        query_sampling=QUERY_SAMPLING_POWER_LAW,
        query_distribution="power_law_plackett_luce_permutation",
        effective_power_a=power_a,
    )


def _zoology_dense_multiquery_ar(
    *,
    vocab_size: int,
    num_examples: int,
    input_seq_len: int,
    seed: int,
    power_a: float,
    num_kv_pairs: int,
    num_passes: int,
    include_slices: bool,
    **kwargs,
) -> MQARDataset:
    """Preserve the upstream-derived query-ordering behavior."""
    context_size = num_kv_pairs * PAIR_WIDTH

    # Six tokens per pair is the minimum legal source layout: four framed
    # context tokens plus a query token and its otherwise-unused label slot.
    source = multiquery_ar(
        vocab_size=vocab_size,
        num_examples=num_examples,
        input_seq_len=num_kv_pairs * 6,
        seed=seed,
        power_a=power_a,
        num_kv_pairs=num_kv_pairs,
        num_passes=num_passes,
        random_non_queries=False,
        include_slices=include_slices,
        **kwargs,
    )

    source_query_inputs = source.inputs[:, context_size:]
    source_query_labels = source.labels[:, context_size:]
    source_query_mask = source_query_labels != -100
    if not torch.all(source_query_mask.sum(dim=1) == num_kv_pairs):
        raise RuntimeError("Unexpected number of supervised queries in Zoology MQAR output.")

    dense_queries = source_query_inputs[source_query_mask].reshape(
        num_examples, num_kv_pairs
    )
    dense_answers = source_query_labels[source_query_mask].reshape(
        num_examples, num_kv_pairs
    )
    inputs = torch.cat([source.inputs[:, :context_size], dense_queries], dim=1)
    labels = torch.full_like(inputs, -100)
    labels[:, context_size:] = dense_answers

    if not torch.all(inputs[:, :context_size][:, 0::PAIR_WIDTH] == PAIR_OPEN_TOKEN):
        raise RuntimeError("Dense MQAR pair-open framing tokens are malformed.")
    if not torch.all(inputs[:, :context_size][:, 3::PAIR_WIDTH] == PAIR_CLOSE_TOKEN):
        raise RuntimeError("Dense MQAR pair-close framing tokens are malformed.")
    return MQARDataset(
        inputs,
        labels,
        slices={
            "num_kv_pairs": num_kv_pairs,
            "input_seq_len": input_seq_len,
            "num_passes": num_passes,
            "context_size": context_size,
            "query_layout": "dense",
            "query_sampling": QUERY_SAMPLING_ZOOLOGY,
            "query_distribution": "power_law_weighted_permutation",
            "generator": "upstream_compaction",
            "effective_power_a": power_a,
        },
    )


def dense_multiquery_ar(
    vocab_size: int,
    num_examples: int,
    input_seq_len: int,
    seed: int,
    power_a: float = 0.01,
    num_kv_pairs: int = 8,
    num_passes: int = 1,
    random_non_queries: bool = False,
    include_slices: bool = True,
    query_sampling: str = QUERY_SAMPLING_UNIFORM,
    mqar_noise_lvl: float = 0.0,
    **kwargs,
) -> MQARDataset:
    """Generate MQAR with a dense KV context and contiguous query keys.

    Uniform and power-law modes sample the Zoology key/value distribution
    directly and use vectorized query permutations. Zoology mode delegates to
    the upstream generator at its minimum legal sequence length. With
    ``mqar_noise_lvl=0``, all modes yield exactly:

         < K1 V1 > ... < Kn Vn > Q1 ... Qn

     With noise enabled, random tokens are inserted between the framed KV
     pairs. Queries and their labels are unchanged.

     Labels contain corresponding values at the query-key positions. Tokens 0
     and 1 are reserved for pair framing and never occur in keys, values, or
     noise.

    Choose ``query_sampling`` according to the experiment:

    - ``uniform`` (default): use for a clean associative-retrieval task without
      context-position/query-position bias. With eight pairs, a predictor that
      ignores the query key and uses only query position is limited to 12.5%.
    - ``power_law``: use to match Zoology's power-law query-order distribution
      efficiently. At ``power_a=0.01`` and eight pairs, this distribution has a
      measurable position-only shortcut (about 21.8% best token accuracy), but
      is statistically equivalent to Zoology query placement.
    - ``zoology``: use when the upstream query-ordering and compaction path are
      required; do not use it merely to reproduce the statistical distribution.

    Performance reference for 100k examples with vocab size 8192 and eight
    pairs on the benchmark host: uniform 0.121 s / 0.46 GiB peak RSS,
    power-law 0.131 s / 0.47 GiB, and Zoology 22.55 s / 6.65 GiB. Thus the
    vectorized modes were roughly 172--187x faster at this scale. Timings are
    hardware-specific; the large Zoology memory cost follows from allocations
    proportional to ``num_examples * vocab_size``.

    ``power_a`` is ignored in uniform mode. Vectorized power-law and Zoology
    modes have the same query-permutation distribution, but are not
    seed-for-seed compatible. ``random_non_queries`` is accepted for API
    compatibility but must be false because dense examples have no non-query
    positions to randomize.
    """
    if "dense_sampling" in kwargs:
        raise TypeError(
            "dense_sampling was removed; use query_sampling instead."
        )

    _validate_dense_args(
        vocab_size=vocab_size,
        num_examples=num_examples,
        input_seq_len=input_seq_len,
        num_kv_pairs=num_kv_pairs,
        num_passes=num_passes,
        random_non_queries=random_non_queries,
        query_sampling=query_sampling,
        power_a=power_a,
    )

    if query_sampling == QUERY_SAMPLING_UNIFORM:
        dataset = _uniform_dense_multiquery_ar(
            vocab_size=vocab_size,
            num_examples=num_examples,
            input_seq_len=input_seq_len,
            seed=seed,
            num_kv_pairs=num_kv_pairs,
            num_passes=num_passes,
        )
    elif query_sampling == QUERY_SAMPLING_POWER_LAW:
        dataset = _power_law_dense_multiquery_ar(
            vocab_size=vocab_size,
            num_examples=num_examples,
            input_seq_len=input_seq_len,
            seed=seed,
            power_a=power_a,
            num_kv_pairs=num_kv_pairs,
            num_passes=num_passes,
        )
    else:
        dataset = _zoology_dense_multiquery_ar(
            vocab_size=vocab_size,
            num_examples=num_examples,
            input_seq_len=input_seq_len,
            seed=seed,
            power_a=power_a,
            num_kv_pairs=num_kv_pairs,
            num_passes=num_passes,
            include_slices=include_slices,
            **kwargs,
        )
    return _add_context_noise(
        dataset,
        vocab_size=vocab_size,
        noise_lvl=mqar_noise_lvl,
        seed=seed + 1,
    )


def build_mqar_datasets(
    *,
    vocab_size: int,
    input_seq_len: int,
    num_kv_pairs: int,
    train_num_examples: int,
    valid_num_examples: int,
    power_a: float,
    random_non_queries: bool,
    data_seed: int,
    dense_queries: bool = False,
    query_sampling: str = QUERY_SAMPLING_UNIFORM,
    mqar_noise_lvl: float = 0.0,
):
    """Build fixed, disjoint train and validation splits used by the controls."""
    max_seed = 2**32
    np.random.seed(data_seed)
    train_seed = int(np.random.randint(0, max_seed // 2))
    valid_seed = int(np.random.randint(max_seed // 2, max_seed))

    common = {
        "vocab_size": vocab_size,
        "input_seq_len": input_seq_len,
        "power_a": power_a,
        "num_kv_pairs": num_kv_pairs,
        "random_non_queries": random_non_queries,
        "mqar_noise_lvl": mqar_noise_lvl,
    }
    generator = dense_multiquery_ar if dense_queries else multiquery_ar
    if dense_queries:
        common["query_sampling"] = query_sampling
    train_dataset = generator(
        num_examples=train_num_examples,
        seed=train_seed,
        **common,
    )
    valid_dataset = generator(
        num_examples=valid_num_examples,
        seed=valid_seed,
        **common,
    )
    return train_dataset, valid_dataset, train_seed, valid_seed
