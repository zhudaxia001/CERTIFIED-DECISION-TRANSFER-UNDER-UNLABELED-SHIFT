import math

import numpy as np

from risk_controlled_postprocessing import apply_threshold, select_threshold


def test_keeps_feasible_baseline():
    labels = np.array([0, 0, 1, 1], dtype=bool)
    baseline = labels.copy()
    fallback = ~labels
    result = select_threshold(
        np.array([0.1, 0.2, 0.8, 0.9]), baseline, fallback, labels, epsilon=0.25
    )
    assert math.isinf(result.threshold)
    assert result.switch_count == 0


def test_exact_safe_fallback_restores_feasibility():
    labels = np.array([0, 0, 1, 1], dtype=bool)
    baseline = ~labels
    fallback = labels.copy()
    scores = np.array([0.9, 0.8, 0.7, 0.6])
    result = select_threshold(scores, baseline, fallback, labels, epsilon=0.25)
    actions = apply_threshold(scores, baseline, fallback, result.threshold)
    assert not result.empty_feasible
    assert np.array_equal(actions, labels)


def test_empty_feasible_follows_algorithm_zero_threshold_fallback():
    labels = np.array([0, 1], dtype=bool)
    baseline = ~labels
    fallback = ~labels
    scores = np.array([0.2, 0.8])
    result = select_threshold(scores, baseline, fallback, labels, epsilon=0.1)
    assert result.empty_feasible
    assert result.threshold == 0.0
    assert result.switch_count == 2


def test_incremental_selector_matches_brute_force():
    rng = np.random.RandomState(7)
    for size in (5, 20, 100):
        for _ in range(20):
            scores = rng.choice(np.linspace(0.0, 1.0, 11), size=size)
            baseline = rng.rand(size) > 0.5
            fallback = rng.rand(size) > 0.5
            labels = rng.rand(size) > 0.5
            epsilon = float(rng.choice(np.linspace(0.05, 0.95, 19)))
            result = select_threshold(scores, baseline, fallback, labels, epsilon)

            candidates = sorted(set([0.0, *scores.tolist(), math.inf]))
            feasible = []
            for threshold in candidates:
                actions = apply_threshold(scores, baseline, fallback, threshold)
                risk = (int(np.sum(actions != labels)) + 1) / (size + 1)
                if risk <= epsilon:
                    feasible.append(threshold)
            expected_threshold = max(feasible) if feasible else 0.0
            assert result.empty_feasible == (not feasible)
            assert result.threshold == expected_threshold
