import pytest
import torch

from zoology_mqar_data import (
    PAIR_CLOSE_TOKEN,
    PAIR_OPEN_TOKEN,
    dense_multiquery_ar,
    multiquery_ar,
)


def _make_dense(noise_lvl=0.0, seed=7):
    return dense_multiquery_ar(
        vocab_size=64,
        num_examples=4,
        input_seq_len=12,
        seed=seed,
        num_kv_pairs=4,
        random_non_queries=False,
        mqar_noise_lvl=noise_lvl,
    )


def _extract_noise(dataset):
    context = dataset.inputs[:, :dataset.slices["context_size"]]
    noise = []
    offset = 0
    for pair_index, gap_count in enumerate(dataset.slices["noise_gap_counts"][:-1]):
        noise.append(context[:, offset:offset + gap_count])
        offset += gap_count + 4
    noise.append(context[:, offset:])
    return torch.cat(noise, dim=1)


def _assert_framed_context(dataset):
    context = dataset.inputs[:, :dataset.slices["context_size"]]
    offset = 0
    for gap_count in dataset.slices["noise_gap_counts"][:-1]:
        offset += gap_count
        pair = context[:, offset:offset + 4]
        assert torch.all(pair[:, 0] == PAIR_OPEN_TOKEN)
        assert torch.all(pair[:, 3] == PAIR_CLOSE_TOKEN)
        assert torch.all((pair[:, 1] != PAIR_OPEN_TOKEN) & (pair[:, 1] != PAIR_CLOSE_TOKEN))
        assert torch.all((pair[:, 2] != PAIR_OPEN_TOKEN) & (pair[:, 2] != PAIR_CLOSE_TOKEN))
        offset += 4


def test_context_noise_preserves_signal_and_query_labels():
    clean = _make_dense()
    noisy = _make_dense(noise_lvl=0.5)

    assert clean.inputs.shape == (4, 12)
    assert noisy.inputs.shape == (4, 28)
    assert clean.slices["context_size"] == 8
    assert noisy.slices["clean_context_size"] == 8
    assert noisy.slices["context_size"] == 24
    assert noisy.slices["noise_tokens"] == 8
    assert len(noisy.slices["noise_gap_counts"]) == 5

    noisy_context = noisy.inputs[:, :24]
    noise_tokens = []
    offset = 0
    for pair_index, gap_count in enumerate(noisy.slices["noise_gap_counts"][:-1]):
        noise_tokens.append(noisy_context[:, offset:offset + gap_count])
        offset += gap_count
        pair = noisy_context[:, offset:offset + 4]
        clean_pair = clean.inputs[:, pair_index * 2:pair_index * 2 + 2]
        assert torch.all(pair[:, 0] == PAIR_OPEN_TOKEN)
        assert torch.all(pair[:, 3] == PAIR_CLOSE_TOKEN)
        assert torch.equal(pair[:, 1:3], clean_pair)
        offset += 4
    noise_tokens.append(noisy_context[:, offset:])
    noise_tokens = torch.cat(noise_tokens, dim=1)
    assert noise_tokens.shape[1] == 8
    assert torch.all((noise_tokens != PAIR_OPEN_TOKEN) & (noise_tokens != PAIR_CLOSE_TOKEN))
    assert torch.any(noise_tokens < 32)
    assert torch.any(noise_tokens >= 32)
    assert torch.equal(noisy.inputs[:, 24:], clean.inputs[:, 8:])
    assert torch.equal(noisy.labels[:, 24:], clean.labels[:, 8:])
    assert torch.all(noisy.labels[:, :24] == -100)


@pytest.mark.parametrize("query_sampling", ["uniform", "power_law", "zoology"])
def test_all_dense_generators_frame_pairs_and_reserve_tokens(query_sampling):
    dataset = dense_multiquery_ar(
        vocab_size=64,
        num_examples=3,
        input_seq_len=12,
        seed=11,
        power_a=0.01,
        num_kv_pairs=4,
        random_non_queries=False,
        query_sampling=query_sampling,
        mqar_noise_lvl=0.5,
    )
    _assert_framed_context(dataset)
    noise = _extract_noise(dataset)
    assert torch.all((noise != PAIR_OPEN_TOKEN) & (noise != PAIR_CLOSE_TOKEN))


def test_upstream_generator_reserves_frame_tokens_for_pairs_and_noise():
    dataset = multiquery_ar(
        vocab_size=64,
        num_examples=3,
        input_seq_len=16,
        seed=11,
        num_kv_pairs=4,
        random_non_queries=True,
        mqar_noise_lvl=0.5,
    )
    _assert_framed_context(dataset)
    noise = _extract_noise(dataset)
    assert torch.all((noise != PAIR_OPEN_TOKEN) & (noise != PAIR_CLOSE_TOKEN))
    query_inputs = dataset.inputs[:, dataset.slices["context_size"]:]
    assert torch.all((query_inputs != PAIR_OPEN_TOKEN) & (query_inputs != PAIR_CLOSE_TOKEN))


def test_context_noise_is_deterministic_and_zero_is_unchanged():
    clean = _make_dense()
    zero_noise = _make_dense(noise_lvl=0.0, seed=7)
    noisy_a = _make_dense(noise_lvl=0.5)
    noisy_b = _make_dense(noise_lvl=0.5)

    assert torch.equal(clean.inputs, zero_noise.inputs)
    assert torch.equal(clean.labels, zero_noise.labels)
    assert torch.equal(noisy_a.inputs, noisy_b.inputs)
    assert torch.equal(noisy_a.labels, noisy_b.labels)


@pytest.mark.parametrize("noise_lvl", [-0.01, 1.0, 1.01, float("nan")])
def test_context_noise_level_must_be_in_unit_interval(noise_lvl):
    with pytest.raises(ValueError, match="mqar_noise_lvl"):
        _make_dense(noise_lvl=noise_lvl)
