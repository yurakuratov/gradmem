"""Zoology multi-query associative recall data generation.

Adapted from HazyResearch/zoology:
https://github.com/HazyResearch/zoology/blob/main/zoology/data/multiquery_ar.py

Upstream commit: 1ad20d193b6113cae1e8f3c655c300d7b4b3f4bb

``multiquery_ar`` is kept identical to upstream except for replacing Zoology's
``DataSegment`` with the local ``MQARDataset`` wrapper. ``dense_multiquery_ar``
defaults to a local vectorized, uniformly permuted dense implementation. It
also provides a vectorized power-law mode and a ``zoology`` mode that preserves
the exact upstream-derived tensors.
"""

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
    **kwargs,
) -> MQARDataset:
    """Generate the synthetic MQAR task from the Zoology paper."""
    assert input_seq_len % 2 == 0, "input_seq_len must be even"
    assert vocab_size > input_seq_len
    assert num_kv_pairs * 2 * num_passes + num_kv_pairs * 2 <= input_seq_len

    np.random.seed(seed)

    # two tokens for key and value
    context_size = num_kv_pairs * 2 * num_passes

    # create keys so that each key is present exactly once in each example
    key_vocab_size = vocab_size // 2
    key_choices = np.arange(1, key_vocab_size)
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
    kvs = np.zeros((num_examples, context_size), dtype=np.int64)
    kvs[:, 0::2] = keys
    kvs[:, 1::2] = values
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

    # replace all the 0 with random values
    if random_non_queries:
        inputs[inputs == 0] = torch.randint(vocab_size, size=inputs.shape)[inputs == 0]
    return MQARDataset(
        inputs,
        labels,
        slices={
            "num_kv_pairs": num_kv_pairs,
            "input_seq_len": input_seq_len,
            "num_passes": num_passes,
        },
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

    context_size = num_kv_pairs * 2
    dense_seq_len = context_size + num_kv_pairs
    if input_seq_len != dense_seq_len:
        raise ValueError(
            "Dense MQAR input_seq_len must equal 3 * num_kv_pairs: "
            f"input_seq_len={input_seq_len}, expected={dense_seq_len}."
        )

    key_vocab_size = vocab_size // 2
    num_key_choices = key_vocab_size - 1  # Token 0 is reserved for fillers.
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
        low=1,
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

    context_size = num_kv_pairs * 2
    context = np.empty((num_examples, context_size), dtype=np.int64)
    context[:, 0::2] = keys
    context[:, 1::2] = values
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
    context_size = num_kv_pairs * 2
    inputs_array = np.concatenate([context, queries], axis=1)
    labels_array = np.full_like(inputs_array, -100)
    labels_array[:, context_size:] = answers
    if np.any(inputs_array == 0):
        raise RuntimeError("Dense MQAR unexpectedly contains filler token 0.")

    return MQARDataset(
        torch.from_numpy(inputs_array),
        torch.from_numpy(labels_array),
        slices={
            "num_kv_pairs": num_kv_pairs,
            "input_seq_len": input_seq_len,
            "num_passes": num_passes,
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
    """Preserve the previous upstream-derived dense tensors exactly."""
    context_size = num_kv_pairs * 2

    # Four tokens per pair is the minimum legal Zoology layout: two context
    # tokens plus a query token and its otherwise-unused prediction slot.
    source = multiquery_ar(
        vocab_size=vocab_size,
        num_examples=num_examples,
        input_seq_len=num_kv_pairs * 4,
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

    if torch.any(inputs == 0):
        raise RuntimeError("Dense MQAR unexpectedly contains filler token 0.")
    return MQARDataset(
        inputs,
        labels,
        slices={
            "num_kv_pairs": num_kv_pairs,
            "input_seq_len": input_seq_len,
            "num_passes": num_passes,
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
    **kwargs,
) -> MQARDataset:
    """Generate MQAR with a dense KV context and contiguous query keys.

    Uniform and power-law modes sample the Zoology key/value distribution
    directly and use vectorized query permutations. Zoology mode delegates to
    the upstream generator at its minimum legal sequence length and preserves
    exact historical tensors. All modes yield exactly:

        K1 V1 ... Kn Vn Q1 ... Qn

    Labels contain corresponding values at the query-key positions. Returned
    inputs contain no filler tokens and never contain query answers.

    Choose ``query_sampling`` according to the experiment:

    - ``uniform`` (default): use for a clean associative-retrieval task without
      context-position/query-position bias. With eight pairs, a predictor that
      ignores the query key and uses only query position is limited to 12.5%.
    - ``power_law``: use to match Zoology's power-law query-order distribution
      efficiently. At ``power_a=0.01`` and eight pairs, this distribution has a
      measurable position-only shortcut (about 21.8% best token accuracy), but
      is statistically equivalent to Zoology query placement.
    - ``zoology``: use only when exact historical tensors for the same seed are
      required. It preserves the previous upstream generation and compaction
      path; do not use it merely to reproduce the statistical distribution.

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
        return _uniform_dense_multiquery_ar(
            vocab_size=vocab_size,
            num_examples=num_examples,
            input_seq_len=input_seq_len,
            seed=seed,
            num_kv_pairs=num_kv_pairs,
            num_passes=num_passes,
        )
    if query_sampling == QUERY_SAMPLING_POWER_LAW:
        return _power_law_dense_multiquery_ar(
            vocab_size=vocab_size,
            num_examples=num_examples,
            input_seq_len=input_seq_len,
            seed=seed,
            power_a=power_a,
            num_kv_pairs=num_kv_pairs,
            num_passes=num_passes,
        )
    return _zoology_dense_multiquery_ar(
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
