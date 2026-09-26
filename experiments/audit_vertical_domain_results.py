"""Recompute vertical-domain summaries from raw per-comparison records."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


GATES = {
    "direct": "direct_gamma",
    "factorized": "factorized_gamma",
    "family_direct": "family_direct_gamma",
    "family_factorized": "family_factorized_gamma",
}


def recompute(records, key):
    switched = np.asarray([row[key] > 0.0 for row in records])
    reference = np.asarray([row["reference_population_f1"] for row in records])
    receiver = np.asarray([row["receiver_population_f1"] for row in records])
    batch_reference = np.asarray([row["reference_batch_f1"] for row in records])
    batch_receiver = np.asarray([row["receiver_batch_f1"] for row in records])
    return {
        "comparisons": len(records),
        "switches": int(switched.sum()),
        "switch_rate": float(switched.mean()),
        "covered_classes": sorted(
            {int(row["class_id"]) for row, switch in zip(records, switched) if switch}
        ),
        "population_losses": int(np.sum(switched & (receiver < reference - 1e-12))),
        "finite_batch_losses": int(
            np.sum(switched & (batch_receiver < batch_reference - 1e-12))
        ),
        "reference_population_f1": float(reference.mean()),
        "receiver_population_f1": float(receiver.mean()),
        "deployed_population_f1": float(
            np.where(switched, receiver, reference).mean()
        ),
        "conditional_population_gain": (
            float((receiver - reference)[switched].mean()) if switched.any() else None
        ),
    }


def compare(expected, actual):
    for key, value in actual.items():
        expected_value = expected[key]
        if isinstance(value, float):
            if not np.isclose(value, expected_value, rtol=0.0, atol=1e-12):
                raise AssertionError(f"{key}: {value} != {expected_value}")
        elif value != expected_value:
            raise AssertionError(f"{key}: {value} != {expected_value}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("prefixes", nargs="+")
    args = parser.parse_args()

    for raw_prefix in args.prefixes:
        prefix = Path(raw_prefix)
        archive = np.load(prefix.with_suffix(".npy"), allow_pickle=True).item()
        stored = json.loads(prefix.with_suffix(".json").read_text(encoding="utf-8"))
        records = archive["records"]
        mismatch = sum(
            row["calibration_reference_count"]
            != row["calibration_receiver_count"]
            for row in records
        )
        if mismatch:
            raise AssertionError(f"{prefix.name}: {mismatch} count mismatches")
        print(f"\n{prefix.name}: comparisons={len(records)} count_mismatch=0")
        for name, key in GATES.items():
            actual = recompute(records, key)
            compare(stored["summary"][name], actual)
            losing_groups = {
                (row["source_seed"], row["target_seed"])
                for row in records
                if row[key] > 0.0
                and row["receiver_population_f1"]
                < row["reference_population_f1"] - 1e-12
            }
            print(
                f"{name:18s} switches={actual['switches']:4d} "
                f"losses={actual['population_losses']:3d} "
                f"groups={len(losing_groups):3d} "
                f"deployed_f1={actual['deployed_population_f1']:.8f}"
            )
        print("RAW_JSON_AUDIT_OK")


if __name__ == "__main__":
    main()
