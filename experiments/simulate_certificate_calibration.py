"""Monte Carlo calibration and power audit for direct and factorized C-RM."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from analyze_factorized_certificate import cp_lower, cp_upper, factorized_gamma


SOURCE_SIZES = (250, 1000, 5000)
TARGET_SIZES = (1000, 5000, 20000)
SCENARIOS = {
    "harmful": 0.475,
    "null": 0.5,
    "beneficial_001": 0.525,
    "beneficial_005": 0.625,
}


def policy_probabilities(theta: float) -> np.ndarray:
    disagreement = 0.2
    reference_tpr = 0.6
    receiver_only = disagreement * theta
    reference_only = disagreement - receiver_only
    both = reference_tpr - reference_only
    neither = 1.0 - receiver_only - reference_only - both
    probabilities = np.asarray([neither, receiver_only, reference_only, both])
    if np.any(probabilities < 0.0) or not np.isclose(probabilities.sum(), 1.0):
        raise AssertionError(f"Invalid source-positive probabilities: {probabilities}")
    return probabilities


def sample_counts(
    probabilities: np.ndarray,
    trials: int,
    replicates: int,
    rng: np.random.Generator,
) -> np.ndarray:
    return rng.multinomial(trials, probabilities, size=replicates)


def as_gate_counts(source: np.ndarray, target: np.ndarray) -> dict[str, np.ndarray]:
    source_trials = source.sum(axis=1)
    target_trials = target.sum(axis=1)
    return {
        "source_trials": source_trials,
        "source_reference": source[:, 2] + source[:, 3],
        "source_receiver": source[:, 1] + source[:, 3],
        "source_disagreement": source[:, 1] + source[:, 2],
        "source_receiver_only": source[:, 1],
        "source_reference_only": source[:, 2],
        "target_trials": target_trials,
        "target_reference": target[:, 2] + target[:, 3],
        "target_receiver": target[:, 1] + target[:, 3],
        "target_disagreement": target[:, 1] + target[:, 2],
        "target_receiver_only": target[:, 1],
        "target_reference_only": target[:, 2],
    }


def direct_gamma(counts: dict[str, np.ndarray], delta: float) -> np.ndarray:
    alpha = delta / 6.0
    delta_tpr_lower = cp_lower(
        counts["source_receiver_only"], counts["source_trials"], alpha
    ) - cp_upper(counts["source_reference_only"], counts["source_trials"], alpha)
    reference_tpr_upper = cp_upper(
        counts["source_reference"], counts["source_trials"], alpha
    )
    reference_rate_lower = cp_lower(
        counts["target_reference"], counts["target_trials"], alpha
    )
    rate_difference_upper = cp_upper(
        counts["target_receiver_only"], counts["target_trials"], alpha
    ) - cp_lower(counts["target_reference_only"], counts["target_trials"], alpha)
    return np.where(
        delta_tpr_lower > 0.0,
        reference_rate_lower * delta_tpr_lower
        - reference_tpr_upper * np.maximum(0.0, rate_difference_upper),
        -1.0,
    )


def population_f1(theta: float) -> tuple[float, float]:
    target_prior = 0.1
    action_rate = 0.2
    reference_tpr = 0.6
    receiver_tpr = reference_tpr + 0.2 * (2.0 * theta - 1.0)
    reference = 2.0 * target_prior * reference_tpr / (target_prior + action_rate)
    receiver = 2.0 * target_prior * receiver_tpr / (target_prior + action_rate)
    return reference, receiver


def binomial_rate_interval(
    successes: int, trials: int, alpha: float = 0.05
) -> tuple[float, float]:
    success_array = np.asarray([successes], dtype=int)
    trial_array = np.asarray([trials], dtype=int)
    lower = cp_lower(success_array, trial_array, alpha / 2.0)[0]
    upper = cp_upper(success_array, trial_array, alpha / 2.0)[0]
    return float(lower), float(upper)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--replicates", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=20260821)
    parser.add_argument("--delta", type=float, default=0.05)
    parser.add_argument(
        "--json",
        type=Path,
        default=root / "results" / "generated" / "certificate_calibration.json",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=root / "results" / "generated" / "certificate_calibration.csv",
    )
    args = parser.parse_args()

    target_probabilities = np.asarray([0.7, 0.1, 0.1, 0.1])
    rows = []
    for scenario_index, (scenario, theta) in enumerate(SCENARIOS.items()):
        source_probabilities = policy_probabilities(theta)
        reference_f1, receiver_f1 = population_f1(theta)
        for source_size in SOURCE_SIZES:
            for target_size in TARGET_SIZES:
                combo_seed = (
                    args.seed
                    + scenario_index * 10**8
                    + source_size * 10_000
                    + target_size
                )
                rng = np.random.default_rng(combo_seed)
                source = sample_counts(
                    source_probabilities, source_size, args.replicates, rng
                )
                target = sample_counts(
                    target_probabilities, target_size, args.replicates, rng
                )
                counts = as_gate_counts(source, target)
                direct = direct_gamma(counts, args.delta) > 0.0
                factorized = factorized_gamma(counts, args.delta) > 0.0
                direct_switches = int(direct.sum())
                factorized_switches = int(factorized.sum())
                direct_interval = binomial_rate_interval(
                    direct_switches, args.replicates
                )
                factorized_interval = binomial_rate_interval(
                    factorized_switches, args.replicates
                )
                rows.append(
                    {
                        "scenario": scenario,
                        "source_positives": source_size,
                        "target_unlabeled": target_size,
                        "replicates": args.replicates,
                        "true_tpr_gain": 0.2 * (2.0 * theta - 1.0),
                        "true_f1_gain": receiver_f1 - reference_f1,
                        "direct_switches": direct_switches,
                        "direct_rate": float(direct.mean()),
                        "direct_rate_ci95_lower": direct_interval[0],
                        "direct_rate_ci95_upper": direct_interval[1],
                        "factorized_switches": factorized_switches,
                        "factorized_rate": float(factorized.mean()),
                        "factorized_rate_ci95_lower": factorized_interval[0],
                        "factorized_rate_ci95_upper": factorized_interval[1],
                    }
                )

    payload = {
        "protocol": {
            "delta": args.delta,
            "replicates_per_cell": args.replicates,
            "seed": args.seed,
            "source_positive_joint_order": [
                "neither",
                "receiver_only",
                "reference_only",
                "both",
            ],
            "target_joint_probabilities": target_probabilities.tolist(),
            "target_prior": 0.1,
            "reference_tpr": 0.6,
            "reference_action_rate": 0.2,
            "receiver_action_rate": 0.2,
            "note": (
                "All scenarios satisfy policy-specific positive-conditional stability. "
                "Target negative-conditionals exist for every scenario and make the fixed "
                "unlabeled target decision table a valid mixture."
            ),
        },
        "rows": rows,
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    with args.csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
