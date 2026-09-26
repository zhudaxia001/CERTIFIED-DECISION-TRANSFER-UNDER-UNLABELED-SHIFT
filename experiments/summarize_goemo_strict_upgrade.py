"""Recompute and aggregate strict GoEmotions model-upgrade archives."""

from pathlib import Path
import argparse
import json

import numpy as np


GATES = {
    "paired": "paired_cp_gamma",
    "fully_paired": "fully_paired_cp_gamma",
}


def summarize(records, gate_key):
    switched = np.asarray([row[gate_key] > 0.0 for row in records], dtype=bool)
    population_gain = np.asarray(
        [row["receiver_population_f1"] - row["reference_population_f1"] for row in records]
    )
    batch_gain = np.asarray(
        [row["receiver_f1"] - row["reference_f1"] for row in records]
    )
    reference_population = np.asarray(
        [row["reference_population_f1"] for row in records]
    )
    receiver_population = np.asarray(
        [row["receiver_population_f1"] for row in records]
    )
    reference_batch = np.asarray([row["reference_f1"] for row in records])
    receiver_batch = np.asarray([row["receiver_f1"] for row in records])
    harmful = population_gain < -1e-12
    beneficial = population_gain > 1e-12
    return {
        "comparisons": len(records),
        "naive_harmful_population_comparisons": int(harmful.sum()),
        "naive_beneficial_population_comparisons": int(beneficial.sum()),
        "switches": int(switched.sum()),
        "unsafe_population_switches": int(np.sum(switched & harmful)),
        "unsafe_batch_switches": int(np.sum(switched & (batch_gain < -1e-12))),
        "harmful_comparisons_rejected": int(np.sum(~switched & harmful)),
        "reference_population_f1": float(reference_population.mean()),
        "receiver_population_f1": float(receiver_population.mean()),
        "deployed_population_f1": float(
            np.where(switched, receiver_population, reference_population).mean()
        ),
        "reference_batch_f1": float(reference_batch.mean()),
        "receiver_batch_f1": float(receiver_batch.mean()),
        "deployed_batch_f1": float(
            np.where(switched, receiver_batch, reference_batch).mean()
        ),
    }


def compare(stored, recomputed, name):
    for key, actual in recomputed.items():
        expected = stored[key]
        if isinstance(actual, float):
            if not np.isclose(actual, expected, rtol=0.0, atol=1e-12):
                raise AssertionError(f"{name}.{key}: {actual} != {expected}")
        elif actual != expected:
            raise AssertionError(f"{name}.{key}: {actual} != {expected}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--prefix", default="goemo_tfidf_to_qwen_strict")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    runs = []
    for seed in args.seeds:
        stem = args.results_dir / f"{args.prefix}_s{seed}_t50"
        stored = json.loads(stem.with_suffix(".json").read_text(encoding="utf-8"))
        raw = np.load(stem.with_suffix(".npy"), allow_pickle=True).item()
        records = raw["records"]
        count_mismatches = sum(
            row["reference_count"] != row["receiver_count"] for row in records
        )
        if count_mismatches != stored["exact_count_mismatches"]:
            raise AssertionError(
                f"seed {seed}: {count_mismatches} count mismatches, "
                f"stored {stored['exact_count_mismatches']}"
            )
        recomputed = {}
        for gate_name, gate_key in GATES.items():
            recomputed[gate_name] = summarize(records, gate_key)
            compare(stored[gate_name], recomputed[gate_name], f"seed{seed}.{gate_name}")
        runs.append(
            {
                "seed": seed,
                "reference_selector_auc": stored["source_selector"]["reference_macro_auc"],
                "receiver_selector_auc": stored["source_selector"]["receiver_macro_auc"],
                "exact_count_mismatches": count_mismatches,
                **recomputed,
            }
        )

    aggregate = {
        "seeds": args.seeds,
        "runs": runs,
        "reference_selector_auc_mean": float(
            np.mean([row["reference_selector_auc"] for row in runs])
        ),
        "receiver_selector_auc_mean": float(
            np.mean([row["receiver_selector_auc"] for row in runs])
        ),
    }
    for gate_name in GATES:
        aggregate[gate_name] = {
            key: int(sum(row[gate_name][key] for row in runs))
            for key in (
                "comparisons",
                "naive_harmful_population_comparisons",
                "naive_beneficial_population_comparisons",
                "switches",
                "unsafe_population_switches",
                "unsafe_batch_switches",
                "harmful_comparisons_rejected",
            )
        }
        for endpoint in (
            "reference_population_f1",
            "receiver_population_f1",
            "deployed_population_f1",
            "reference_batch_f1",
            "receiver_batch_f1",
            "deployed_batch_f1",
        ):
            aggregate[gate_name][endpoint] = float(
                np.mean([row[gate_name][endpoint] for row in runs])
            )

    output = args.output or (
        args.results_dir / f"{args.prefix}_multiseed_summary.json"
    )
    output.write_text(json.dumps(aggregate, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(aggregate, indent=2))
    print(f"saved={output}")


if __name__ == "__main__":
    main()
