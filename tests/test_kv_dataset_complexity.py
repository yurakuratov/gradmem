from itertools import product

import pytest

import push_dataset
from kv_dataset_utils import ComplexityValueMapper, generate_sequence


def test_complexity_mapper_is_deterministic_and_seeded():
    first = ComplexityValueMapper(2, 6, alphabet="abcd", seed=17)
    same = ComplexityValueMapper(2, 6, alphabet="abcd", seed=17)
    different_seed = ComplexityValueMapper(2, 6, alphabet="abcd", seed=18)

    latent_values = ["".join(chars) for chars in product("abcd", repeat=2)]

    assert [first(value) for value in latent_values] == [same(value) for value in latent_values]
    assert any(first(value) != different_seed(value) for value in latent_values)


@pytest.mark.parametrize("complexity", [1, 2, 3])
def test_reduced_complexity_mapper_caches_repeated_values(monkeypatch, complexity):
    mapper = ComplexityValueMapper(
        complexity,
        v_length=6,
        seed=17,
    )
    latent_value = "a" * complexity
    permutation_calls = 0
    original_permute = mapper._permute

    def counting_permute(integer):
        nonlocal permutation_calls
        permutation_calls += 1
        return original_permute(integer)

    monkeypatch.setattr(mapper, "_permute", counting_permute)

    first = mapper(latent_value)
    second = mapper(latent_value)

    assert first == second
    assert permutation_calls == 1
    assert mapper._cache == {latent_value: first}


def test_complexity_mapper_is_injective_and_uses_full_visible_alphabet():
    alphabet = "abcd"
    mapper = ComplexityValueMapper(2, 5, alphabet=alphabet, seed=42)
    latent_values = ["".join(chars) for chars in product(alphabet, repeat=2)]
    outputs = [mapper(value) for value in latent_values]

    assert len(set(outputs)) == len(latent_values)
    assert all(len(output) == 5 for output in outputs)
    assert all(set(output) <= set(alphabet) for output in outputs)


def test_full_complexity_mapper_permutes_complete_value_space():
    alphabet = "abc"
    mapper = ComplexityValueMapper(2, 2, alphabet=alphabet, seed=42)
    latent_values = ["".join(chars) for chars in product(alphabet, repeat=2)]
    outputs = [mapper(value) for value in latent_values]

    assert all(len(output) == 2 for output in outputs)
    assert len(set(outputs)) == len(latent_values)
    assert set(outputs) == set(latent_values)


def test_complexity_mapper_does_not_expose_a_latent_coordinate():
    alphabet = "abcd"
    mapper = ComplexityValueMapper(2, 5, alphabet=alphabet, seed=42)
    latent_values = ["".join(chars) for chars in product(alphabet, repeat=2)]
    outputs = {value: mapper(value) for value in latent_values}

    for latent_position in range(2):
        for output_position in range(5):
            output_chars_by_latent_char = {
                latent_char: {
                    outputs[value][output_position]
                    for value in latent_values
                    if value[latent_position] == latent_char
                }
                for latent_char in alphabet
            }
            assert any(
                len(output_chars) > 1
                for output_chars in output_chars_by_latent_char.values()
            )


@pytest.mark.parametrize(
    "kwargs,error",
    [
        ({"complexity": 0, "v_length": 4}, "positive"),
        ({"complexity": 5, "v_length": 4}, "greater than or equal to complexity"),
        ({"complexity": 1, "v_length": 4, "alphabet": "aabc"}, "unique"),
    ],
)
def test_complexity_mapper_rejects_invalid_configuration(kwargs, error):
    with pytest.raises(ValueError, match=error):
        ComplexityValueMapper(**kwargs)


def test_generate_sequence_uses_complexity_mapping_for_every_value():
    mapper = ComplexityValueMapper(2, 6, alphabet="abcd", seed=3)

    sample = generate_sequence(
        num_kv_pairs=10,
        k_length=2,
        v_length=6,
        n_segments=2,
        min_segment_len=0,
        max_segment_len=0,
        kv_alphabet="abcd",
        complexity=2,
        complexity_function=mapper,
    )

    visible_values = [pair.split(":", 1)[1][:-1] for pair in sample["kv_pairs"]]
    possible_values = {mapper("".join(chars)) for chars in product("abcd", repeat=2)}
    assert all(len(value) == 6 for value in visible_values)
    assert set(visible_values) <= possible_values


def test_generate_sequence_bypasses_mapper_at_full_complexity():
    mapper_calls = 0

    def forbidden_mapper(value):
        nonlocal mapper_calls
        mapper_calls += 1
        raise AssertionError(f"full-complexity mapper was called for {value!r}")

    sample = generate_sequence(
        num_kv_pairs=4,
        k_length=2,
        v_length=2,
        n_segments=1,
        min_segment_len=0,
        max_segment_len=0,
        kv_alphabet="abcd",
        complexity=2,
        complexity_function=forbidden_mapper,
    )

    visible_values = [pair.split(":", 1)[1][:-1] for pair in sample["kv_pairs"]]
    assert mapper_calls == 0
    assert all(len(value) == 2 for value in visible_values)
    assert all(set(value) <= set("abcd") for value in visible_values)


def test_create_dataset_shares_one_complexity_mapping_across_splits(monkeypatch):
    seen_mappers = []

    def fake_generate_sequence(*args, complexity, complexity_function, **kwargs):
        del args, kwargs
        seen_mappers.append(complexity_function)
        visible_value = complexity_function("a" * complexity)
        return {
            "context": f"!aa:{visible_value}!|",
            "query": "?!aa:",
            "target": f"{visible_value}!|",
        }

    monkeypatch.setattr(push_dataset, "generate_sequence", fake_generate_sequence)

    dataset = push_dataset.create_dataset(
        num_kv_pairs=1,
        k_length=2,
        v_length=5,
        kv_vocab_size=4,
        train_samples=2,
        valid_samples=1,
        test_samples=1,
        complexity=2,
        complexity_seed=9,
    )

    assert set(dataset) == {"train", "valid", "test"}
    assert len(seen_mappers) == 4
    assert len({id(mapper) for mapper in seen_mappers}) == 1
    assert dataset["train"][0]["target"] == dataset["valid"][0]["target"]
    assert dataset["valid"][0]["target"] == dataset["test"][0]["target"]


def test_create_dataset_supports_full_complexity_values_without_mapper(monkeypatch):
    def forbidden_mapper_factory(*args, **kwargs):
        raise AssertionError(f"full-complexity mapper constructed with {args!r}, {kwargs!r}")

    monkeypatch.setattr(push_dataset, "create_complexity_function", forbidden_mapper_factory)

    dataset = push_dataset.create_dataset(
        num_kv_pairs=2,
        k_length=2,
        v_length=2,
        train_samples=2,
        valid_samples=1,
        test_samples=1,
        complexity=2,
        complexity_seed=5,
    )

    assert len(dataset["train"]) == 2
    assert len(dataset["valid"]) == 1
    assert len(dataset["test"]) == 1
    assert all(len(sample["target"]) == 4 for split in dataset.values() for sample in split)


def test_push_dataset_config_name_includes_complexity():
    calls = []

    class FakeDataset:
        def push_to_hub(self, repo_id, config_name):
            calls.append((repo_id, config_name))

    push_dataset.push_dataset_to_hub(
        FakeDataset(),
        repo_id="example/kv",
        num_kv_pairs=8,
        k_length=2,
        v_length=7,
        kv_vocab_size=62,
        complexity=3,
    )

    assert calls == [("example/kv", "N8-K2V7C3-V62")]
