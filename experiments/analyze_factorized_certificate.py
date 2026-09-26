"""Replay a disagreement-factorized paired certificate from archived counts."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import beta as beta_distribution


DELTA_GRID = (0.005, 0.01, 0.02, 0.03, 0.04, 0.05)
MAGNITUDES = (0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0)
BETAS = (0.5, 1.0, 2.0)


def cp_lower(successes: np.ndarray, trials: np.ndarray, alpha: float) -> np.ndarray:
    result = np.zeros(len(successes), dtype=float)
    mask = successes > 0
    result[mask] = beta_distribution.ppf(
        alpha,
        successes[mask],
        trials[mask] - successes[mask] + 1,
    )
    return result


def cp_upper(successes: np.ndarray, trials: np.ndarray, alpha: float) -> np.ndarray:
    result = np.ones(len(successes), dtype=float)
    mask = successes < trials
    result[mask] = beta_distribution.ppf(
        1.0 - alpha,
        successes[mask] + 1,
        trials[mask] - successes[mask],
    )
    return result


def reconstruct_counts(rows: list[dict], target_trials: int) -> dict[str, np.ndarray]:
    count = len(rows)
    source_trials = np.asarray([row["source_positive_count"] for row in rows], dtype=int)
    source_reference = np.rint(
        [row["reference_tpr"] * trials for row, trials in zip(rows, source_trials)]
    ).astype(int)
    source_receiver = np.rint(
        [row["receiver_tpr"] * trials for row, trials in zip(rows, source_trials)]
    ).astype(int)
    source_disagreement = np.rint(
        [
            row["fully_paired_cp_source_disagreement_rate"] * trials
            for row, trials in zip(rows, source_trials)
        ]
    ).astype(int)
    source_receiver_only_numerator = (
        source_disagreement + source_receiver - source_reference
    )
    if np.any(source_receiver_only_numerator % 2):
        raise AssertionError("Source disagreement cells are not integral")
    source_receiver_only = source_receiver_only_numerator // 2
    source_reference_only = source_disagreement - source_receiver_only

    target_trials_array = np.full(count, target_trials, dtype=int)
    target_reference = np.rint(
        [row["paired_cp_target_reference_rate"] * target_trials for row in rows]
    ).astype(int)
    target_receiver = np.rint(
        [row["paired_cp_target_receiver_rate"] * target_trials for row in rows]
    ).astype(int)
    target_disagreement = np.rint(
        [row["fully_paired_cp_target_disagreement_rate"] * target_trials for row in rows]
    ).astype(int)
    target_receiver_only_numerator = (
        target_disagreement + target_receiver - target_reference
    )
    if np.any(target_receiver_only_numerator % 2):
        raise AssertionError("Target disagreement cells are not integral")
    target_receiver_only = target_receiver_only_numerator // 2
    target_reference_only = target_disagreement - target_receiver_only

    arrays = {
        "source_trials": source_trials,
        "source_reference": source_reference,
        "source_receiver": source_receiver,
        "source_disagreement": source_disagreement,
        "source_receiver_only": source_receiver_only,
        "source_reference_only": source_reference_only,
        "target_trials": target_trials_array,
        "target_reference": target_reference,
        "target_receiver": target_receiver,
        "target_disagreement": target_disagreement,
        "target_receiver_only": target_receiver_only,
        "target_reference_only": target_reference_only,
    }
    for name, values in arrays.items():
        if np.any(values < 0):
            raise AssertionError(f"Negative reconstructed count in {name}")
    return arrays


def archived_gamma_audit(rows: list[dict], counts: dict[str, np.ndarray]) -> None:
    alpha = 0.05 / 6.0
    source_receiver_only_lower = cp_lower(
        counts["source_receiver_only"], counts["source_trials"], alpha
    )
    source_reference_only_upper = cp_upper(
        counts["source_reference_only"], counts["source_trials"], alpha
    )
    delta_lower = source_receiver_only_lower - source_reference_only_upper
    reference_tpr_upper = cp_upper(
        counts["source_reference"], counts["source_trials"], alpha
    )
    reference_rate_lower = cp_lower(
        counts["target_reference"], counts["target_trials"], alpha
    )
    receiver_only_upper = cp_upper(
        counts["target_receiver_only"], counts["target_trials"], alpha
    )
    reference_only_lower = cp_lower(
        counts["target_reference_only"], counts["target_trials"], alpha
    )
    rate_difference_upper = receiver_only_upper - reference_only_lower
    gamma = np.where(
        delta_lower > 0.0,
        reference_rate_lower * delta_lower
        - reference_tpr_upper * np.maximum(0.0, rate_difference_upper),
        -1.0,
    )
    archived = np.asarray([row["fully_paired_cp_gamma"] for row in rows], dtype=float)
    if not np.allclose(gamma, archived, atol=1e-12, rtol=1e-10):
        raise AssertionError("Reconstructed counts do not reproduce archived gamma")


def factorized_gamma(counts: dict[str, np.ndarray], delta: float) -> np.ndarray:
    alpha = delta / 6.0
    source_disagreement_lower = cp_lower(
        counts["source_disagreement"], counts["source_trials"], alpha
    )
    source_orientation_lower = cp_lower(
        counts["source_receiver_only"], counts["source_disagreement"], alpha
    )
    delta_tpr_lower = np.where(
        source_orientation_lower > 0.5,
        source_disagreement_lower * (2.0 * source_orientation_lower - 1.0),
        -1.0,
    )
    reference_tpr_upper = cp_upper(
        counts["source_reference"], counts["source_trials"], alpha
    )

    reference_rate_lower = cp_lower(
        counts["target_reference"], counts["target_trials"], alpha
    )
    target_disagreement_upper = cp_upper(
        counts["target_disagreement"], counts["target_trials"], alpha
    )
    target_orientation_upper = cp_upper(
        counts["target_receiver_only"], counts["target_disagreement"], alpha
    )
    positive_rate_difference_upper = np.maximum(
        0.0,
        target_disagreement_upper * (2.0 * target_orientation_upper - 1.0),
    )
    return np.where(
        delta_tpr_lower > 0.0,
        reference_rate_lower * delta_tpr_lower
        - reference_tpr_upper * positive_rate_difference_upper,
        -1.0,
    )


def population_fbeta(rows: list[dict], policy: str, beta: float) -> np.ndarray:
    beta_sq = beta * beta
    prior = np.asarray([row["target_prior"] for row in rows], dtype=float)
    tpr = np.asarray([row[f"{policy}_tpr"] for row in rows], dtype=float)
    rate = np.asarray([row[f"target_{policy}_rate"] for row in rows], dtype=float)
    return (1.0 + beta_sq) * prior * tpr / (beta_sq * prior + rate)


def summarize(rows: list[dict], counts: dict[str, np.ndarray]) -> dict:
    reference_population = np.asarray(
        [row["reference_population_f1"] for row in rows], dtype=float
    )
    receiver_population = np.asarray(
        [row["receiver_population_f1"] for row in rows], dtype=float
    )
    reference_batch = np.asarray([row["reference_f1"] for row in rows], dtype=float)
    receiver_batch = np.asarray([row["receiver_f1"] for row in rows], dtype=float)
    magnitudes = np.asarray([row["magnitude"] for row in rows], dtype=float)

    output = {"delta_curve": {}, "family_wise": {}}
    previous_switches = None
    for delta in DELTA_GRID:
        switched = factorized_gamma(counts, delta) > 0.0
        if previous_switches is not None and np.any(previous_switches & ~switched):
            raise AssertionError("Switch sets must expand as delta increases")
        previous_switches = switched
        output["delta_curve"][str(delta)] = {
            "switches": int(switched.sum()),
            "population_losses": int(
                np.sum(switched & (receiver_population + 1e-12 < reference_population))
            ),
            "finite_batch_losses": int(
                np.sum(switched & (receiver_batch + 1e-12 < reference_batch))
            ),
            "by_magnitude": {
                str(magnitude): int(np.sum(switched & (magnitudes == magnitude)))
                for magnitude in MAGNITUDES
            },
        }

    for family_size in (30, 90):
        delta = 0.05 / family_size
        switched = factorized_gamma(counts, delta) > 0.0
        output["family_wise"][str(family_size)] = {
            "per_comparison_delta": delta,
            "switches": int(switched.sum()),
            "population_losses": int(
                np.sum(switched & (receiver_population + 1e-12 < reference_population))
            ),
            "finite_batch_losses": int(
                np.sum(switched & (receiver_batch + 1e-12 < reference_batch))
            ),
        }

    primary_switches = factorized_gamma(counts, 0.05) > 0.0
    output["primary_fbeta_losses"] = {}
    for beta in BETAS:
        reference = population_fbeta(rows, "reference", beta)
        receiver = population_fbeta(rows, "receiver", beta)
        output["primary_fbeta_losses"][str(beta)] = int(
            np.sum(primary_switches & (receiver + 1e-12 < reference))
        )
    return output


def verify_primary(summary: dict) -> None:
    expected_curve = {
        "0.005": (169, 0, 0),
        "0.01": (310, 0, 0),
        "0.02": (585, 0, 1),
        "0.03": (821, 0, 2),
        "0.04": (1080, 0, 4),
        "0.05": (1334, 0, 6),
    }
    for delta, expected in expected_curve.items():
        row = summary["delta_curve"][delta]
        observed = (row["switches"], row["population_losses"], row["finite_batch_losses"])
        if observed != expected:
            raise AssertionError(f"Unexpected factorized curve at delta={delta}: {observed}")
    for beta, losses in summary["primary_fbeta_losses"].items():
        if losses != 0:
            raise AssertionError(f"Unexpected population F-beta loss for beta={beta}")
    if summary["family_wise"]["30"]["switches"] != 68:
        raise AssertionError("Unexpected 30-class family-wise coverage")
    if summary["family_wise"]["90"]["switches"] != 26:
        raise AssertionError("Unexpected 90-policy family-wise coverage")


def write_csv(path: Path, summary: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["delta", "switches", "population_losses", "finite_batch_losses"])
        for delta in DELTA_GRID:
            row = summary["delta_curve"][str(delta)]
            writer.writerow(
                [delta, row["switches"], row["population_losses"], row["finite_batch_losses"]]
            )


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=root / "experiments" / "results_a800_20260819" /
        "coco_crm_dinov2_full117k_m25_fixed_iid_s100_d05_pop.npy",
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=root / "experiments" / "results_a800_20260819" /
        "coco_crm_dinov2_full117k_m25_fixed_iid_s100_d05_pop.json",
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=root / "results" / "generated" / "factorized_certificate.json",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=root / "results" / "generated" / "factorized_certificate.csv",
    )
    parser.add_argument(
        "--verify-primary",
        action="store_true",
        help="Require the exact regression counts of the 63k primary archive.",
    )
    parser.add_argument(
        "--max-target-seed",
        type=int,
        default=None,
        help="Optionally retain records whose target_seed is below this value.",
    )
    args = parser.parse_args()

    archive = np.load(args.input, allow_pickle=True).item()
    rows = archive["records"]
    if args.max_target_seed is not None:
        rows = [row for row in rows if int(row["target_seed"]) < args.max_target_seed]
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))["protocol"]
    counts = reconstruct_counts(rows, int(protocol["target_size"]))
    archived_gamma_audit(rows, counts)
    summary = summarize(rows, counts)
    if args.verify_primary:
        if len(rows) != 63_000:
            raise AssertionError(f"Unexpected primary comparison count: {len(rows)}")
        verify_primary(summary)
    summary["protocol"] = {
        "input": str(args.input),
        "comparisons": len(rows),
        "target_trials": int(protocol["target_size"]),
        "risk_allocation": "six one-sided CP bounds at delta/6",
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_csv(args.csv, summary)
    print(json.dumps(summary, indent=2))
    print(f"saved: {args.json}")
    print(f"saved: {args.csv}")


if __name__ == "__main__":
    main()
