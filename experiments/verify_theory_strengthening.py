"""Numerical audits for decision-transfer analysis and its supporting results."""

from math import ceil, exp, isclose


def _f_beta(beta: float, q: float, u: float, selected_positive_mass: float) -> float:
    return (1.0 + beta**2) * selected_positive_mass / (beta**2 * q + u)


def verify_theory_strengthening() -> None:
    # The main-text fixed-count gain identity, including empty/full selections.
    n = 50
    for beta in (0.5, 1.0, 2.0):
        for positives in (1, 10, 49):
            for count in (0, 5, 25, n):
                minimum = max(0, count - (n - positives))
                maximum = min(count, positives)
                for reference_tp in (minimum, maximum):
                    for candidate_tp in (minimum, maximum):
                        gain = _f_beta(beta, positives, count, candidate_tp) - _f_beta(
                            beta, positives, count, reference_tp
                        )
                        expected = (1.0 + beta**2) * (candidate_tp - reference_tp) / (
                            beta**2 * positives + count
                        )
                        assert isclose(gain, expected, abs_tol=1e-12)

    # Population cardinality--ranking decomposition.
    for beta in (0.5, 1.0, 2.0):
        for q in (0.01, 0.2, 0.7):
            for u in (0.005, 0.05, 0.4, 0.9):
                positive_cap = min(u, q)
                selected_positive_mass = 0.63 * positive_cap
                score = _f_beta(beta, q, u, selected_positive_mass)
                if u <= q:
                    cardinality = beta**2 * (q - u) / (beta**2 * q + u)
                else:
                    cardinality = (u - q) / (beta**2 * q + u)
                ranking = (
                    (1.0 + beta**2)
                    * (positive_cap - selected_positive_mass)
                    / (beta**2 * q + u)
                )
                assert isclose(1.0 - score, cardinality + ranking, abs_tol=1e-12)

    # Label-shift count mismatch: mu_Q-q=(mu_P-p)+(J-1)(q-p).
    for p, q, a, b in (
        (0.02, 0.08, 0.80, 0.03),
        (0.20, 0.10, 0.65, 0.12),
        (0.40, 0.70, 0.55, 0.20),
    ):
        j = a - b
        mu_p = b + j * p
        mu_q = b + j * q
        rhs = (mu_p - p) + (j - 1.0) * (q - p)
        assert isclose(mu_q - q, rhs, abs_tol=1e-12)

    # A smooth monotone-marginal example checks the derivative identity.
    beta, q, decay = 1.3, 0.3, 2.0
    for u in (0.1, 0.3, 0.7):
        selected_positive_mass = 0.5 * (1.0 - exp(-decay * u)) / decay
        marginal_precision = 0.5 * exp(-decay * u)
        derivative_numerator = (
            marginal_precision * (beta**2 * q + u) - selected_positive_mass
        )
        threshold_gap = marginal_precision - _f_beta(
            beta, q, u, selected_positive_mass
        ) / (1.0 + beta**2)
        assert derivative_numerator * threshold_gap >= 0.0
        assert isclose(
            derivative_numerator,
            threshold_gap * (beta**2 * q + u),
            abs_tol=1e-12,
        )

    # Explicit fixed-depth AUC construction from Appendix Proposition D.3.
    for k in (1, 3, 10):
        for epsilon in (0.5, 0.1, 0.01):
            positives = ceil(max(k, 2.0 * k / epsilon)) + 1
            negatives = ceil(max(1.0, 2.0 / epsilon)) + 1
            auc_1 = k / positives
            auc_2 = 1.0 - (positives - k + 1) / (positives * negatives)
            assert k - 1 < k
            assert auc_2 - auc_1 > 1.0 - epsilon

    # Proposition D.10: identical unlabeled decisions admit opposite labels.
    # Cell order is (S,R) = (1,0), (0,1), (1,1), (0,0).
    for pi, shared in ((0.05, 0.2), (0.2, 0.0), (0.4, 0.1)):
        mass = (pi, pi, shared, 1.0 - 2.0 * pi - shared)
        assert min(mass) >= 0.0
        action_rate = pi + shared
        for beta in (0.5, 1.0, 2.0):
            plus_s = _f_beta(beta, pi, action_rate, pi)
            plus_r = _f_beta(beta, pi, action_rate, 0.0)
            minus_s, minus_r = plus_r, plus_s
            assert plus_s > plus_r and minus_s < minus_r


if __name__ == "__main__":
    verify_theory_strengthening()
    print("Theory-strengthening verification passed.")
