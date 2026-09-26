"""Recompute full-COCO decision transfer for multiple F-beta objectives."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


BETAS = (0.5, 1.0, 2.0)
REPORT_MAGNITUDES = (1.0, 8.0)
GATES = {
    "conservative": "paired_cp_gamma",
    "fully_paired": "fully_paired_cp_gamma",
}


def population_fbeta(row: dict, policy: str, beta: float) -> float:
    beta_sq = beta * beta
    prior = float(row["target_prior"])
    tpr = float(row[f"{policy}_tpr"])
    rate = float(row[f"target_{policy}_rate"])
    denominator = beta_sq * prior + rate
    if denominator <= 0.0:
        return 0.0
    return (1.0 + beta_sq) * prior * tpr / denominator


def analyze(rows: list[dict]) -> dict:
    summary: dict[str, dict] = {}
    for beta in BETAS:
        reference = np.asarray(
            [population_fbeta(row, "reference", beta) for row in rows], dtype=float
        )
        receiver = np.asarray(
            [population_fbeta(row, "receiver", beta) for row in rows], dtype=float
        )
        by_magnitude = {}
        magnitudes = np.asarray([float(row["magnitude"]) for row in rows])
        for magnitude in REPORT_MAGNITUDES:
            mask = magnitudes == magnitude
            by_magnitude[str(magnitude)] = {
                "reference": float(reference[mask].mean()),
                "receiver": float(receiver[mask].mean()),
                "gain": float((receiver[mask] - reference[mask]).mean()),
                "comparisons": int(mask.sum()),
            }
        gates = {}
        for label, key in GATES.items():
            switched = np.asarray([float(row[key]) > 0.0 for row in rows])
            losses = switched & (receiver + 1e-12 < reference)
            deployed = np.where(switched, receiver, reference)
            gates[label] = {
                "switches": int(switched.sum()),
                "population_losses": int(losses.sum()),
                "mean_deployed_gain": float(deployed.mean() - reference.mean()),
            }
        summary[str(beta)] = {"magnitudes": by_magnitude, "gates": gates}
    return summary


def verify(summary: dict) -> None:
    expected_switches = {"conservative": 166, "fully_paired": 352}
    for beta in BETAS:
        result = summary[str(beta)]
        for magnitude in REPORT_MAGNITUDES:
            if result["magnitudes"][str(magnitude)]["gain"] <= 0.0:
                raise AssertionError(f"Non-positive gain for beta={beta}, magnitude={magnitude}")
        for gate, expected in expected_switches.items():
            gate_result = result["gates"][gate]
            if gate_result["switches"] != expected:
                raise AssertionError(f"Unexpected {gate} switches for beta={beta}")
            if gate_result["population_losses"] != 0:
                raise AssertionError(f"Unexpected {gate} F-beta loss for beta={beta}")


def write_csv(path: Path, summary: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "beta", "gain_at_1x", "gain_at_8x", "conservative_switches",
                "conservative_losses", "fully_paired_switches", "fully_paired_losses",
            ]
        )
        for beta in BETAS:
            result = summary[str(beta)]
            writer.writerow(
                [
                    beta,
                    result["magnitudes"]["1.0"]["gain"],
                    result["magnitudes"]["8.0"]["gain"],
                    result["gates"]["conservative"]["switches"],
                    result["gates"]["conservative"]["population_losses"],
                    result["gates"]["fully_paired"]["switches"],
                    result["gates"]["fully_paired"]["population_losses"],
                ]
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
        "--json",
        type=Path,
        default=root / "results" / "generated" / "coco_fbeta_sensitivity.json",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=root / "results" / "generated" / "coco_fbeta_sensitivity.csv",
    )
    args = parser.parse_args()

    archive = np.load(args.input, allow_pickle=True).item()
    rows = archive["records"]
    if len(rows) != 63_000:
        raise AssertionError(f"Unexpected comparison count: {len(rows)}")
    summary = analyze(rows)
    verify(summary)
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_csv(args.csv, summary)
    print(json.dumps(summary, indent=2))
    print(f"saved: {args.json}")
    print(f"saved: {args.csv}")


if __name__ == "__main__":
    main()
