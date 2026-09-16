from collections import Counter

import pytest

import push_ar_multihop_dataset as upload_pipeline
from ar_multihop_dataset import (
    DEFAULT_H_VALUES,
    DEFAULT_N_VALUES,
    DEFAULT_SPLIT_SEEDS,
    ENTITY_ALPHABET_SIZE,
    RAW_ENTITY_FIELDS,
    _generate_sample_with_debug,
    configuration_name,
    decode_entity,
    entity_length_for_configuration,
    generate_sample,
    generate_split_records,
    node_count_for_configuration,
    validate_configuration,
    validate_sample,
)
from push_ar_multihop_dataset import (
    features_for_configuration,
    initialize_hub_repository,
    materialize_configuration,
    metadata_configs_from_repo_files,
    process_full_configuration,
    sync_hub_metadata,
    validate_materialized_dataset,
)


def _decoded_pairs(sample):
    return {
        decode_entity(key): decode_entity(value)
        for key, value in zip(sample["context_keys"], sample["context_values"])
    }


def _follow_mapping(mapping, key):
    seen = set()
    distance = 0
    while key in mapping:
        assert key not in seen
        seen.add(key)
        key = mapping[key]
        distance += 1
    return key, distance


@pytest.mark.parametrize("n_pairs", DEFAULT_N_VALUES)
@pytest.mark.parametrize("hop_length", DEFAULT_H_VALUES)
def test_entity_length_is_minimal_and_covers_all_required_nodes(n_pairs, hop_length):
    entity_length = entity_length_for_configuration(n_pairs, hop_length)
    node_count = node_count_for_configuration(n_pairs, hop_length)

    assert 16**entity_length >= node_count
    assert entity_length == 1 or 16 ** (entity_length - 1) < node_count
    assert configuration_name(n_pairs, hop_length) == f"N{n_pairs}-H{hop_length}-V16"


def test_exact_power_and_n16_h1_entity_lengths_use_integer_arithmetic():
    assert node_count_for_configuration(8, 1) == 16
    assert entity_length_for_configuration(8, 1) == 1
    assert node_count_for_configuration(16, 1) == 32
    assert entity_length_for_configuration(16, 1) == 2


@pytest.mark.parametrize("n_pairs", DEFAULT_N_VALUES)
@pytest.mark.parametrize("hop_length", DEFAULT_H_VALUES)
def test_every_configuration_has_fixed_raw_shapes_and_only_base16_symbols(
    n_pairs,
    hop_length,
):
    sample = generate_sample(n_pairs, hop_length, seed=12345)
    entity_length = entity_length_for_configuration(n_pairs, hop_length)
    validate_sample(sample, n_pairs, hop_length)

    assert set(sample) == {"sample_id", *RAW_ENTITY_FIELDS, "hop_distances"}
    assert isinstance(sample["sample_id"], int)
    for field_name in RAW_ENTITY_FIELDS:
        assert len(sample[field_name]) == n_pairs
        assert all(len(entity) == entity_length for entity in sample[field_name])
        assert all(
            0 <= symbol < ENTITY_ALPHABET_SIZE
            for entity in sample[field_name]
            for symbol in entity
        )
    assert len(sample["hop_distances"]) == n_pairs
    assert not any(
        name in sample
        for name in ("input_ids", "context_input_ids", "query_input_ids", "segments")
    )


def test_unique_nodes_and_chain_edges_survive_base16_encoding_and_context_shuffle():
    n_pairs, hop_length = 32, 4
    sample, debug = _generate_sample_with_debug(n_pairs, hop_length, seed=7)
    mapping = _decoded_pairs(sample)
    context_order = tuple(
        (decode_entity(key), decode_entity(value))
        for key, value in zip(sample["context_keys"], sample["context_values"])
    )

    assert len(debug.chains) == n_pairs // hop_length
    assert len({node for chain in debug.chains for node in chain}) == (
        n_pairs + n_pairs // hop_length
    )
    assert context_order == debug.context_record_order
    assert mapping == dict(debug.mapping)
    for chain in debug.chains:
        assert len(chain) == hop_length + 1
        assert len(set(chain)) == hop_length + 1
        for edge_index in range(hop_length):
            assert mapping[chain[edge_index]] == chain[edge_index + 1]
        assert chain[-1] not in mapping


def test_internal_values_equal_next_keys_and_queries_have_correct_targets_and_hops():
    n_pairs, hop_length = 16, 4
    sample = generate_sample(n_pairs, hop_length, seed=29)
    mapping = _decoded_pairs(sample)
    query_keys = [decode_entity(key) for key in sample["query_keys"]]
    targets = [decode_entity(target) for target in sample["targets"]]

    assert len(query_keys) == len(set(query_keys)) == n_pairs
    assert set(query_keys) == set(mapping)
    assert Counter(sample["hop_distances"]) == {
        depth: n_pairs // hop_length for depth in range(1, hop_length + 1)
    }
    for key, target, distance in zip(query_keys, targets, sample["hop_distances"]):
        expected_target, expected_distance = _follow_mapping(mapping, key)
        assert target == expected_target
        assert distance == expected_distance


def test_context_and_query_orders_use_independent_deterministic_streams():
    generated = [_generate_sample_with_debug(16, 4, seed=seed) for seed in range(8)]
    repeated = [_generate_sample_with_debug(16, 4, seed=seed) for seed in range(8)]
    assert generated == repeated
    assert any(
        tuple(key for key, _ in debug.context_record_order)
        != tuple(key for key, _, _ in debug.query_order)
        for _, debug in generated
    )


def test_generation_is_reproducible_and_split_streams_are_distinct():
    kwargs = dict(n_pairs=8, hop_length=2, num_samples=3, split_seed=101)
    train = list(generate_split_records(split_name="train", **kwargs))
    assert train == list(generate_split_records(split_name="train", **kwargs))
    validation = list(generate_split_records(split_name="validation", **kwargs))
    test = list(generate_split_records(split_name="test", **kwargs))
    assert train != validation != test
    assert train != test
    assert list(DEFAULT_SPLIT_SEEDS.values()) == [42, 43, 44]


def test_invalid_configuration_and_raw_shape_are_rejected():
    with pytest.raises(ValueError, match="divisible"):
        validate_configuration(10, 4)
    with pytest.raises(ValueError, match="positive"):
        validate_configuration(8, 0)

    malformed = generate_sample(16, 1, seed=5)
    malformed["context_keys"][0] = [0]
    with pytest.raises(ValueError, match="shape"):
        validate_sample(malformed, 16, 1)


def test_arrow_materialization_is_incremental_and_fixed_shape(tmp_path):
    split_sizes = {"train": 3, "validation": 2, "test": 2}
    dataset = materialize_configuration(
        16,
        1,
        split_sizes=split_sizes,
        split_seeds=DEFAULT_SPLIT_SEEDS,
        cache_dir=tmp_path / "cache",
    )
    validate_materialized_dataset(dataset, 16, 1, split_sizes)

    assert dataset["train"].features == features_for_configuration(16, 1)
    assert dataset["train"].features["context_keys"].length == 16
    assert dataset["train"].features["context_keys"].feature.length == 2


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


def test_local_save_removes_cache_only_after_success(tmp_path):
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

    monkeypatch.setattr(upload_pipeline, "materialize_configuration", fake_materialize)
    monkeypatch.setattr(
        upload_pipeline,
        "validate_materialized_dataset",
        lambda *_args, **_kwargs: None,
    )


def test_push_removes_cache_after_success_but_failed_push_preserves_it(
    tmp_path,
    monkeypatch,
):
    completed = _FakeDataset()
    _install_fake_materialization(monkeypatch, completed)
    _process_small_configuration(tmp_path, push=True)
    config_name = configuration_name(8, 2)
    assert completed.push_calls[0][1]["config_name"] == config_name
    assert not (tmp_path / "cache" / config_name).exists()

    failed = _FakeDataset(upload_error=RuntimeError("upload failed"))
    _install_fake_materialization(monkeypatch, failed)
    with pytest.raises(RuntimeError, match="upload failed"):
        _process_small_configuration(tmp_path, push=True)
    assert (tmp_path / "cache" / config_name).is_dir()


def test_failed_local_save_preserves_completed_generation_cache(
    tmp_path,
    monkeypatch,
):
    generated = _FakeDataset()
    _install_fake_materialization(monkeypatch, generated)

    def fail_save(*_args, **_kwargs):
        raise OSError("save failed")

    monkeypatch.setattr(upload_pipeline, "save_configuration", fail_save)
    with pytest.raises(OSError, match="save failed"):
        _process_small_configuration(tmp_path, push=False)
    assert (tmp_path / "cache" / configuration_name(8, 2)).is_dir()


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


def _parquet_files(config_name):
    return [
        f"{config_name}/{split_name}-00000-of-00001.parquet"
        for split_name in ("train", "validation", "test")
    ]


def test_repository_initialization_preserves_existing_metadata(tmp_path):
    card_path = tmp_path / "README.md"
    card_path.write_text("---\nconfigs:\n- config_name: existing\n---\n", encoding="utf-8")
    api = _FakeHubApi(card_path, readme_exists=True)
    initialize_hub_repository("owner/dataset", api=api)
    assert len(api.create_calls) == 1
    assert api.upload_calls == []


def test_repository_initialization_installs_ar_card_when_absent(tmp_path):
    api = _FakeHubApi(tmp_path / "unused", readme_exists=False)
    initialize_hub_repository("owner/dataset", api=api)
    assert api.upload_calls[0]["path_in_repo"] == "README.md"
    assert api.upload_calls[0]["path_or_fileobj"].endswith("ar_multihop_README.md")


def test_resumed_metadata_sync_indexes_all_complete_partial_pushes(tmp_path):
    first = configuration_name(8, 1)
    second = configuration_name(16, 2)
    incomplete = configuration_name(32, 4)
    repo_files = [
        *_parquet_files(first),
        *_parquet_files(second),
        f"{incomplete}/train-00000-of-00001.parquet",
    ]
    configs = metadata_configs_from_repo_files(repo_files)
    assert list(configs) == [first, second]

    card_path = tmp_path / "README.md"
    card_path.write_text(
        "---\npretty_name: Existing card\nconfigs:\n"
        f"- config_name: {first}\n---\n# Existing body\n",
        encoding="utf-8",
    )
    api = _FakeHubApi(card_path, readme_exists=True, repo_files=repo_files)
    indexed = sync_hub_metadata("owner/dataset", api=api)
    assert indexed == (first, second)
    uploaded_card = api.upload_calls[-1]["path_or_fileobj"].decode("utf-8")
    assert "# Existing body" in uploaded_card
    assert f"config_name: {first}" in uploaded_card
    assert f"config_name: {second}" in uploaded_card
