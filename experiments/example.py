"""Synthetic smoke test of RM and the direct-cell gate; not an experiment result."""
import numpy as np

from direct_cell_certificate import fully_paired_rate_certificate


def rate_match(reference_decisions, candidate_scores):
    reference_decisions = np.asarray(reference_decisions, dtype=bool)
    candidate_scores = np.asarray(candidate_scores, dtype=float)
    if reference_decisions.ndim != 1 or candidate_scores.shape != reference_decisions.shape:
        raise ValueError("Expected aligned one-dimensional arrays")
    if not np.isfinite(candidate_scores).all():
        raise ValueError("Scores must be finite")
    k = int(reference_decisions.sum())
    selected = np.zeros(len(candidate_scores), dtype=bool)
    # Stable ordering resolves ties by input index without changing the budget.
    selected[np.argsort(-candidate_scores, kind="stable")[:k]] = True
    return selected


def main():
    ref = np.array([True, False, True, False, False])
    selected = rate_match(ref, [.2, .9, .1, .8, .3])
    assert selected.tolist() == [False, True, False, True, False]
    assert selected.sum() == ref.sum()
    assert not rate_match(np.zeros(3), [1, 1, 1]).any()
    assert rate_match(np.ones(3), [1, 1, 1]).all()
    assert rate_match([1, 0, 0], [1, 1, 1]).tolist() == [True, False, False]
    # Large synthetic samples exercise approval. Source entries are positives;
    # target entries are decisions only, without target labels.
    n = 10000
    source_ref = np.r_[np.ones(5000), np.zeros(5000)]
    source_new = np.r_[np.ones(7500), np.zeros(2500)]
    target_ref = np.r_[np.ones(2000), np.zeros(n - 2000)]
    target_new = np.roll(target_ref, 100)
    result = fully_paired_rate_certificate(source_ref, source_new, .5, .5,
                                          target_ref, target_new, .05, 0.)
    assert result["fully_paired_cp_gamma"] > 0
    reversed_result = fully_paired_rate_certificate(source_new, source_ref, .5, .5,
                                                   target_ref, target_new, .05, 0.)
    assert reversed_result["fully_paired_cp_gamma"] <= 0
    empty = fully_paired_rate_certificate(np.array([]), np.array([]), .5, .5,
                                         target_ref, target_new, .05, 0.)
    assert empty["fully_paired_cp_gamma"] <= 0
    print("PASS: exact action count, ties, empty evidence, beneficial/harmful synthetic proposals.")


if __name__ == "__main__":
    main()
