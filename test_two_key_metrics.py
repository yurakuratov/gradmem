"""Regression tests for ordered two-key concatenation evaluation metrics.

Run with ``python test_two_key_metrics.py`` from ``test_time_gd``.  The
plain assertion tests are also pytest-compatible when pytest is installed.
"""

import numpy as np

from kv_dataset_utils import create_tokenizer
from run_gradmemgpt_on_kv_retrieval import collate_fn, compute_metrics_fn


def _stats():
    return {key: np.array(0.0) for key in [
        'inner_loss',
        'inner_grad_norm_mean', 'inner_grad_norm_max', 'inner_grad_norm_min',
        'mem_norm_mean', 'mem_norm_max', 'mem_norm_min',
        'delta_mem_norm_mean', 'delta_mem_norm_max', 'delta_mem_norm_min',
    ]}


def test_ordered_pair_metrics_and_reversal_detection():
    tokenizer = create_tokenizer(62)
    # Stored dataset rows need only context/query/target.  Pair-aware chunks
    # make the first query cross-segment (pair 0 + pair 2) and the second
    # query same-segment (pairs 0 + 1), independently of generator metadata.
    batch = [
        {
            'context': '!aa:AB!!bb:CD!|!cc:EF!!dd:GH!|',
            'query': '?!aa:?!cc:',
            'target': 'AB!EF!|',
        },
        {
            'context': '!aa:AB!!bb:CD!|!cc:EF!!dd:GH!|',
            'query': '?!aa:?!bb:',
            'target': 'AB!CD!|',
        },
    ]
    collated = collate_fn(
        batch, tokenizer, hopfield=True,
        pairs_per_segment=2,
    )
    metric_inputs = {key: value.numpy() for key, value in collated['input_ids'].items()}
    labels = collated['labels'].numpy()

    # The second example is perfect.  Make the first one exactly V2,V1 to
    # distinguish a genuine reversal from generic incorrect output.
    joint_preds = np.where(labels == -100, 0, labels)
    value_index = metric_inputs['pair_value_token_index']
    first_mask, second_mask = value_index[0] == 0, value_index[0] == 1
    joint_preds[0, first_mask] = labels[0, second_mask]
    joint_preds[0, second_mask] = labels[0, first_mask]
    joint_preds = np.concatenate(
        [joint_preds, np.zeros((len(batch), 1), dtype=joint_preds.dtype)], axis=1)

    # Both constituent single-key reads are perfect, so the conditional
    # metric's denominator includes both examples.
    single_labels = metric_inputs['single_labels']
    single_preds = np.where(single_labels == -100, 0, single_labels)
    single_preds = np.concatenate(
        [single_preds, np.zeros((len(batch), 2, 1), dtype=single_preds.dtype)], axis=2)

    class EvalPred:
        predictions = (joint_preds, _stats(), single_preds)
        label_ids = labels
        inputs = metric_inputs

    metrics = compute_metrics_fn(
        EvalPred(),
        [tokenizer.convert_tokens_to_ids('!'), tokenizer.convert_tokens_to_ids('|')],
        tokenizer,
    )
    assert metrics['two_key_both_values_exact_match'] == 0.5
    assert metrics['two_key_cross_segment_exact_match'] == 0.0
    assert metrics['two_key_same_segment_exact_match'] == 1.0
    assert metrics['two_key_cross_segment_distance_1_exact_match'] == 0.0
    assert metrics['two_key_reversal_error_rate'] == 0.5
    assert metrics['two_key_both_single_exact_match'] == 1.0
    assert metrics['two_key_joint_exact_match_given_both_single_correct'] == 0.5


if __name__ == '__main__':
    test_ordered_pair_metrics_and_reversal_detection()
    print('ordered two-key metric tests passed')
