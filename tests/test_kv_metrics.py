import numpy as np

from run_gradmemgpt_on_kv_retrieval import fully_answered_value_metrics


def test_fully_answered_value_metrics_averages_complete_values():
    labels = np.array([
        [10, 11, -100, 20, 21, 22, -100],
        [30, 31, -100, 40, 41, 42, -100],
    ])
    predictions = np.array([
        [10, 99, -100, 20, 21, 22, -100],
        [30, 31, -100, 40, 99, 42, -100],
    ])

    metrics = fully_answered_value_metrics(predictions, labels, [])

    assert metrics["value_exact_match"] == 0.5


def test_fully_answered_value_metrics_ignores_delimiters():
    labels = np.array([[10, 11, 1, 2, -100]])
    predictions = np.array([[10, 11, 9, 9, -100]])

    metrics = fully_answered_value_metrics(predictions, labels, [1, 2])

    assert metrics == {}
