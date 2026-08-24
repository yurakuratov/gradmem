import random

from kv_dataset_utils import QUERY_PAIR_RE, generate_sequence, query_target_spans


def test_all_pairs_query_requests_each_context_pair_once():
    random.seed(7)
    sample = generate_sequence(
        num_kv_pairs=8,
        n_segments=4,
        query_all_pairs=True,
    )

    keys = [pair[1:pair.index(":")] for pair in sample["kv_pairs"]]
    values = [pair.split(":", 1)[1][:-1] for pair in sample["kv_pairs"]]
    values_by_key = dict(zip(keys, values))
    query_records = list(QUERY_PAIR_RE.finditer(sample["query"]))
    query_keys = [match.group(1) for match in query_records]
    query_values = [match.group(2) for match in query_records]

    assert sample["context"].endswith("|")
    assert set(query_keys) == set(keys)
    assert len(query_keys) == len(keys) == 8
    assert len(set(query_keys)) == 8
    assert query_values == [values_by_key[key] for key in query_keys]
    assert sample["target"] == ""
    spans = query_target_spans(sample["query"], sample["target"])
    assert [sample["query"][start:end] for start, end in spans] == query_values
    assert sample["input_sequence"] == sample["context"] + sample["query"]


def test_all_pairs_query_keeps_context_generation_unchanged():
    random.seed(11)
    single = generate_sequence(num_kv_pairs=8, query_all_pairs=False)
    random.seed(11)
    all_pairs = generate_sequence(num_kv_pairs=8, query_all_pairs=True)

    assert all_pairs["context"] == single["context"]
    assert all_pairs["kv_pairs"] == single["kv_pairs"]
    assert all_pairs["segment_ids_to_kv_ids"] == single["segment_ids_to_kv_ids"]


def test_legacy_single_query_still_uses_target_field():
    random.seed(3)
    sample = generate_sequence(num_kv_pairs=1)

    assert sample["query"].startswith("?!")
    assert sample["query"].endswith(":")
    assert sample["target"].endswith("!|")
    assert query_target_spans(sample["query"], sample["target"]) == [
        (len(sample["query"]), len(sample["query"]) + len(sample["target"]))
    ]
