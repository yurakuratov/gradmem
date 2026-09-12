from collections import Counter

import pytest

import push_zoology_multihop_dataset as upload_pipeline
from push_zoology_multihop_dataset import (
    FEATURES,
    initialize_hub_repository,
    materialize_configuration,
    metadata_configs_from_repo_files,
    process_full_configuration,
    sync_hub_metadata,
    validate_materialized_dataset,
)
from zoology_multihop_dataset import (
    BOS_ID,
    CONTEXT_END_ID,
    CONTEXT_START_ID,
    DEFAULT_H_VALUES,
    DEFAULT_N_VALUES,
    DEFAULT_SPLIT_SEEDS,
    ENTITY_COUNT,
    ENTITY_START_ID,
    ENTITY_STOP_ID,
    EOS_ID,
    KV_END_ID,
    KV_RECORD_WIDTH,
    KV_SEPARATOR_ID,
    KV_START_ID,
    QUERY_END_ID,
    QUERY_RECORD_WIDTH,
    QUERY_START_ID,
    SPECIAL_TOKENS,
    VOCAB_SIZE,
    _generate_sample_with_debug,
    configuration_name,
    decode_sample,
    generate_sample,
    generate_split_records,
    parse_context_records,
    parse_query_keys,
    validate_configuration,
    validate_sample,
)


def follow_mapping(mapping, key):
    distance = 0
    seen = set()
    while key in mapping:
        assert key not in seen
        seen.add(key)
        key = mapping[key]
        distance += 1
    return key, distance


def test_vocabulary_is_exactly_4096_with_one_shared_entity_pool():
    assert SPECIAL_TOKENS == {
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
    assert set(SPECIAL_TOKENS.values()) == set(range(ENTITY_START_ID))
    assert ENTITY_COUNT == ENTITY_STOP_ID - ENTITY_START_ID
    assert len(SPECIAL_TOKENS) + ENTITY_COUNT == VOCAB_SIZE == 4096
    assert range(ENTITY_START_ID, ENTITY_STOP_ID).start == 10
    assert range(ENTITY_START_ID, ENTITY_STOP_ID).stop == 4096


@pytest.mark.parametrize("n_pairs", DEFAULT_N_VALUES)
@pytest.mark.parametrize("hop_length", DEFAULT_H_VALUES)
def test_every_release_configuration_generates_and_validates(n_pairs, hop_length):
    sample = generate_sample(n_pairs, hop_length, seed=12345)
    validate_sample(sample, n_pairs, hop_length)

    assert configuration_name(n_pairs, hop_length) == f"N{n_pairs}-H{hop_length}-V4096"
    assert len(sample["context_input_ids"]) == 4 + KV_RECORD_WIDTH * n_pairs
    assert len(sample["query_input_ids"]) == 2 + QUERY_RECORD_WIDTH * n_pairs
    assert len(sample["targets"]) == n_pairs
    assert len(sample["hop_distances"]) == n_pairs
    assert not any("segment" in field for field in sample)

    all_token_ids = (
        sample["context_input_ids"]
        + sample["query_input_ids"]
        + sample["targets"]
    )
    assert all(0 <= token_id < VOCAB_SIZE for token_id in all_token_ids)


def test_fixed_width_records_and_shuffled_context_reconstruct_mapping():
    sample, debug = _generate_sample_with_debug(32, 4, seed=7)
    context = sample["context_input_ids"]
    queries = sample["query_input_ids"]

    assert context[:2] == [BOS_ID, CONTEXT_START_ID]
    assert context[-2:] == [CONTEXT_END_ID, EOS_ID]
    for offset in range(2, len(context) - 2, KV_RECORD_WIDTH):
        assert context[offset] == KV_START_ID
        assert context[offset + 2] == KV_SEPARATOR_ID
        assert context[offset + 4] == KV_END_ID

    assert queries[0] == BOS_ID and queries[-1] == EOS_ID
    for offset in range(1, len(queries) - 1, QUERY_RECORD_WIDTH):
        assert queries[offset] == QUERY_START_ID
        assert queries[offset + 2] == QUERY_END_ID

    reconstructed = dict(parse_context_records(context))
    assert reconstructed == dict(debug.mapping)
    assert tuple(parse_context_records(context)) == debug.context_record_order


def test_chains_are_disjoint_acyclic_and_have_exact_length():
    n_pairs = 32
    hop_length = 8
    sample, debug = _generate_sample_with_debug(n_pairs, hop_length, seed=19)
    mapping = dict(parse_context_records(sample["context_input_ids"]))
    keys = [key for key, _ in debug.mapping]
    values = [value for _, value in debug.mapping]

    assert len(debug.chains) == n_pairs // hop_length
    assert all(len(chain) == hop_length + 1 for chain in debug.chains)
    assert len({node for chain in debug.chains for node in chain}) == (
        n_pairs + n_pairs // hop_length
    )
    assert len(keys) == len(set(keys)) == n_pairs
    assert len(values) == len(set(values)) == n_pairs
    assert all(ENTITY_START_ID <= node < ENTITY_STOP_ID for node in keys + values)

    for chain in debug.chains:
        assert len(set(chain)) == len(chain)
        assert all(mapping[chain[index]] == chain[index + 1] for index in range(hop_length))
        assert chain[-1] not in mapping


def test_queries_cover_all_keys_with_correct_terminal_targets_and_depths():
    n_pairs = 16
    hop_length = 4
    sample = generate_sample(n_pairs, hop_length, seed=29)
    mapping = dict(parse_context_records(sample["context_input_ids"]))
    query_keys = parse_query_keys(sample["query_input_ids"])

    assert len(query_keys) == len(set(query_keys)) == n_pairs
    assert set(query_keys) == set(mapping)
    assert not set(sample["targets"]).intersection(sample["query_input_ids"])
    assert Counter(sample["hop_distances"]) == {
        depth: n_pairs // hop_length for depth in range(1, hop_length + 1)
    }
    for key, target, distance in zip(
        query_keys,
        sample["targets"],
        sample["hop_distances"],
    ):
        expected_target, expected_distance = follow_mapping(mapping, key)
        assert target == expected_target
        assert distance == expected_distance


def test_query_and_context_orders_use_independent_deterministic_streams():
    samples = [_generate_sample_with_debug(16, 4, seed=seed) for seed in range(8)]
    for (sample, debug), (repeated, repeated_debug) in zip(
        samples,
        [_generate_sample_with_debug(16, 4, seed=seed) for seed in range(8)],
    ):
        assert sample == repeated
        assert debug == repeated_debug

    assert any(
        tuple(key for key, _ in debug.context_record_order)
        != tuple(key for key, _, _ in debug.query_order)
        for _, debug in samples
    )


def test_generation_is_reproducible_and_split_streams_are_distinct():
    kwargs = dict(n_pairs=8, hop_length=2, num_samples=3, split_seed=101)
    train = list(generate_split_records(split_name="train", **kwargs))
    repeated_train = list(generate_split_records(split_name="train", **kwargs))
    validation = list(generate_split_records(split_name="validation", **kwargs))
    test = list(generate_split_records(split_name="test", **kwargs))

    assert train == repeated_train
    assert train != validation
    assert train != test
    assert validation != test
    assert list(DEFAULT_SPLIT_SEEDS.values()) == [42, 43, 44]


def test_validation_rejects_target_leak_and_incorrect_target():
    sample = generate_sample(8, 2, seed=37)
    leaked = {field: list(value) if isinstance(value, list) else value for field, value in sample.items()}
    leaked["query_input_ids"][1] = leaked["targets"][0]
    with pytest.raises(ValueError):
        validate_sample(leaked, 8, 2)

    incorrect = {
        field: list(value) if isinstance(value, list) else value
        for field, value in sample.items()
    }
    incorrect["targets"][0] = ENTITY_START_ID
    with pytest.raises(ValueError, match="incorrect terminal target"):
        validate_sample(incorrect, 8, 2)


def test_invalid_configuration_is_rejected():
    with pytest.raises(ValueError, match="divisible"):
        validate_configuration(10, 4)
    with pytest.raises(ValueError, match="positive"):
        validate_configuration(8, 0)


def test_arrow_materialization_is_incremental_and_has_documented_schema(tmp_path):
    split_sizes = {"train": 3, "validation": 2, "test": 2}
    dataset = materialize_configuration(
        8,
        2,
        split_sizes=split_sizes,
        split_seeds=DEFAULT_SPLIT_SEEDS,
        cache_dir=tmp_path / "cache",
    )
    validate_materialized_dataset(dataset, 8, 2, split_sizes)

    assert dataset["train"].features == FEATURES
    assert len(dataset["train"]) == 3
    assert "targets shown only for inspection" in decode_sample(dataset["train"][0])


def _process_small_configuration(tmp_path, *, push):
    process_full_configuration(
        8,
        2,
        split_sizes={"train": 1, "validation": 1, "test": 1},
        split_seeds=DEFAULT_SPLIT_SEEDS,
        cache_dir=tmp_path / "cache",
        output_dir=tmp_path / "output",
        overwrite_local=False,
        push=push,
        repo_id="owner/test-dataset",
        max_shard_size="1MB",
    )


def test_local_save_removes_generation_cache_only_after_success(tmp_path):
    _process_small_configuration(tmp_path, push=False)

    config_name = configuration_name(8, 2)
    assert (tmp_path / "output" / config_name).is_dir()
    assert not (tmp_path / "cache" / config_name).exists()


class _FakeDataset:
    def __init__(self, upload_error=None):
        self.upload_error = upload_error
        self.push_calls = []

    def push_to_hub(self, *args, **kwargs):
        self.push_calls.append((args, kwargs))
        if self.upload_error is not None:
            raise self.upload_error


def _install_fake_materialization(monkeypatch, fake_dataset):
    def fake_materialize(n_pairs, hop_length, *, cache_dir, **_kwargs):
        (cache_dir / configuration_name(n_pairs, hop_length)).mkdir(
            parents=True,
            exist_ok=True,
        )
        return fake_dataset

    monkeypatch.setattr(
        upload_pipeline,
        "materialize_configuration",
        fake_materialize,
    )
    monkeypatch.setattr(
        upload_pipeline,
        "validate_materialized_dataset",
        lambda *_args, **_kwargs: None,
    )


def test_push_uploads_directly_and_removes_cache_after_success(
    tmp_path,
    monkeypatch,
):
    fake_dataset = _FakeDataset()
    _install_fake_materialization(monkeypatch, fake_dataset)
    monkeypatch.setattr(
        upload_pipeline,
        "save_configuration",
        lambda *_args, **_kwargs: pytest.fail("push mode must not save a full copy"),
    )

    _process_small_configuration(tmp_path, push=True)

    config_name = configuration_name(8, 2)
    assert fake_dataset.push_calls == [
        (
            ("owner/test-dataset",),
            {"config_name": config_name, "max_shard_size": "1MB"},
        )
    ]
    assert not (tmp_path / "output" / config_name).exists()
    assert not (tmp_path / "cache" / config_name).exists()


def test_failed_upload_preserves_completed_generation_cache(tmp_path, monkeypatch):
    fake_dataset = _FakeDataset(upload_error=RuntimeError("upload failed"))
    _install_fake_materialization(monkeypatch, fake_dataset)

    with pytest.raises(RuntimeError, match="upload failed"):
        _process_small_configuration(tmp_path, push=True)

    config_name = configuration_name(8, 2)
    assert (tmp_path / "cache" / config_name).is_dir()
    assert not (tmp_path / "output" / config_name).exists()


def test_failed_local_save_preserves_completed_generation_cache(
    tmp_path,
    monkeypatch,
):
    fake_dataset = _FakeDataset()
    _install_fake_materialization(monkeypatch, fake_dataset)

    def fail_save(*_args, **_kwargs):
        raise OSError("save failed")

    monkeypatch.setattr(upload_pipeline, "save_configuration", fail_save)
    with pytest.raises(OSError, match="save failed"):
        _process_small_configuration(tmp_path, push=False)

    config_name = configuration_name(8, 2)
    assert (tmp_path / "cache" / config_name).is_dir()
    assert not (tmp_path / "output" / config_name).exists()


class _FakeHubApi:
    def __init__(self, card_path, *, readme_exists=False, repo_files=()):
        self.card_path = card_path
        self.readme_exists = readme_exists
        self.repo_files = list(repo_files)
        self.create_calls = []
        self.upload_calls = []

    def create_repo(self, **kwargs):
        self.create_calls.append(kwargs)

    def file_exists(self, *_args, **_kwargs):
        return self.readme_exists

    def list_repo_files(self, *_args, **_kwargs):
        return self.repo_files

    def hf_hub_download(self, *_args, **_kwargs):
        return str(self.card_path)

    def upload_file(self, **kwargs):
        self.upload_calls.append(kwargs)


def test_repository_initialization_does_not_overwrite_existing_metadata(tmp_path):
    card_path = tmp_path / "README.md"
    card_path.write_text("---\nconfigs:\n- config_name: existing\n---\n", encoding="utf-8")
    api = _FakeHubApi(card_path, readme_exists=True)

    initialize_hub_repository("owner/dataset", api=api)

    assert len(api.create_calls) == 1
    assert api.upload_calls == []


def test_repository_initialization_uploads_card_when_absent(tmp_path):
    api = _FakeHubApi(tmp_path / "unused", readme_exists=False)

    initialize_hub_repository("owner/dataset", api=api)

    assert len(api.upload_calls) == 1
    assert api.upload_calls[0]["path_in_repo"] == "README.md"


def _parquet_files(config_name):
    return [
        f"{config_name}/{split_name}-00000-of-00001.parquet"
        for split_name in ("train", "validation", "test")
    ]


def test_metadata_index_collects_all_complete_uploaded_configurations():
    first = configuration_name(8, 1)
    second = configuration_name(16, 2)
    incomplete = configuration_name(32, 4)
    configs = metadata_configs_from_repo_files(
        [
            "README.md",
            *_parquet_files(first),
            *_parquet_files(second),
            f"{incomplete}/train-00000-of-00001.parquet",
        ]
    )

    assert list(configs) == [first, second]
    assert configs[first]["data_files"] == [
        {"split": "train", "path": f"{first}/train-*"},
        {"split": "validation", "path": f"{first}/validation-*"},
        {"split": "test", "path": f"{first}/test-*"},
    ]


def test_sync_hub_metadata_preserves_card_body_and_indexes_all_files(tmp_path):
    first = configuration_name(8, 1)
    second = configuration_name(16, 2)
    card_path = tmp_path / "README.md"
    card_path.write_text(
        "---\npretty_name: Existing card\nconfigs:\n"
        f"- config_name: {second}\n"
        f"  data_files:\n  - split: train\n    path: {second}/train-*\n"
        "---\n# Existing body\n",
        encoding="utf-8",
    )
    api = _FakeHubApi(
        card_path,
        readme_exists=True,
        repo_files=[*_parquet_files(first), *_parquet_files(second)],
    )

    indexed = sync_hub_metadata("owner/dataset", api=api)

    assert indexed == (first, second)
    uploaded_card = api.upload_calls[-1]["path_or_fileobj"].decode("utf-8")
    assert "pretty_name: Existing card" in uploaded_card
    assert "# Existing body" in uploaded_card
    assert f"config_name: {first}" in uploaded_card
    assert f"config_name: {second}" in uploaded_card
