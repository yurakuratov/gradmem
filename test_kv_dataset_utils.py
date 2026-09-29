"""Tests for synthetic KV query generation.

Run with ``python test_kv_dataset_utils.py`` from ``test_time_gd``; the
plain assertion tests are also pytest-compatible when pytest is installed.
"""

import random
import re

from kv_dataset_utils import generate_sequence


PAIR_RE = re.compile(r'!([^!|:]+):([^!|]+)!')
TWO_KEY_QUERY_RE = re.compile(r'^\?!([^!:|]+):\?!([^!:|]+):$')


def _context_pairs(sample):
    return dict(PAIR_RE.findall(sample['context']))


def test_ordered_pair_concat_target_follows_query_order():
    random.seed(7)
    sample = generate_sequence(
        num_kv_pairs=8,
        k_length=2,
        v_length=3,
        n_segments=4,
        min_segment_len=0,
        max_segment_len=0,
        query_mode='ordered_pair_concat',
    )

    match = TWO_KEY_QUERY_RE.fullmatch(sample['query'])
    assert match is not None
    queried_keys = list(match.groups())
    assert queried_keys == sample['query_keys']
    assert len(set(queried_keys)) == 2

    pair_map = _context_pairs(sample)
    queried_values = [pair_map[key] for key in queried_keys]
    assert queried_values == sample['query_values']
    assert sample['target'] == f'{queried_values[0]}!{queried_values[1]}!|'
    assert sample['input_sequence'] == sample['context'] + sample['query']
    assert len(sample['query_key_indices']) == len(sample['query_segment_ids']) == 2


def test_ordered_pair_concat_enforces_source_segment_relation():
    # Two facts over two segments is the smallest case and would be flaky
    # without the generator's explicit source-segment assignment.
    for relation, expected_same in [('same', True), ('different', False)]:
        random.seed(11)
        sample = generate_sequence(
            num_kv_pairs=2,
            k_length=2,
            v_length=2,
            n_segments=2,
            min_segment_len=0,
            max_segment_len=0,
            query_mode='ordered_pair_concat',
            query_pair_segment_relation=relation,
        )

        assert (sample['query_segment_ids'][0] == sample['query_segment_ids'][1]) is expected_same


def test_single_query_mode_remains_available_and_has_uniform_metadata():
    random.seed(23)
    sample = generate_sequence(
        num_kv_pairs=3,
        k_length=2,
        v_length=2,
        n_segments=2,
        min_segment_len=0,
        max_segment_len=0,
    )

    assert sample['query_mode'] == 'single'
    assert len(sample['query_keys']) == len(sample['query_values']) == 1
    assert sample['query'] == f"?!{sample['query_keys'][0]}:"
    assert sample['target'] == f"{sample['query_values'][0]}!|"


def test_ordered_pair_concat_rejects_impossible_requests():
    try:
        generate_sequence(num_kv_pairs=1, query_mode='ordered_pair_concat')
    except ValueError as exc:
        assert 'num_kv_pairs' in str(exc)
    else:
        raise AssertionError('expected ordered_pair_concat num_kv_pairs validation error')

    try:
        generate_sequence(
            num_kv_pairs=2,
            n_segments=1,
            query_mode='ordered_pair_concat',
            query_pair_segment_relation='different',
        )
    except ValueError as exc:
        assert 'n_segments' in str(exc)
    else:
        raise AssertionError('expected different-segment validation error')


if __name__ == '__main__':
    tests = [value for name, value in sorted(globals().items()) if name.startswith('test_')]
    for test in tests:
        test()
    print(f'{len(tests)} KV dataset generator tests passed')
