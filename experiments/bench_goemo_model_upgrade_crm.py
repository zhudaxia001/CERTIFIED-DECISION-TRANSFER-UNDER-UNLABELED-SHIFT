"""Audit a deep-model upgrade with target-unlabeled C-RM on GoEmotions."""

from pathlib import Path
import argparse
import json
import time

import numpy as np
from datasets import load_dataset
from sklearn.metrics import roc_auc_score

from bench_coco_crm_certificate import (
    fully_paired_rate_certificate,
    paired_rate_certificate,
    population_f1,
    sample_mixture_batch,
)
from bench_goemo_score_transfer import best_threshold


MAGNITUDES = (0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0)
START = time.time()


def multilabel_matrix(rows, classes=28):
    values = np.zeros((len(rows), classes), dtype=bool)
    for index, labels in enumerate(rows):
        values[index, labels] = True
    return values


def break_ties(valid_scores, test_scores, salt):
    """Apply deterministic sub-gap jitter so every score defines exact Top-k."""
    combined = np.concatenate([valid_scores, test_scores]).astype(np.float64)
    unique = np.unique(combined)
    gaps = np.diff(unique)
    positive_gaps = gaps[gaps > 0.0]
    epsilon = float(positive_gaps.min() / 4.0) if len(positive_gaps) else 1e-9
    token = np.arange(len(combined), dtype=np.float64)
    if salt % 2:
        token = token[::-1]
    jitter = (token + 1.0) / (len(combined) + 1.0)
    adjusted = combined + epsilon * (jitter - 0.5)
    return adjusted[: len(valid_scores)], adjusted[len(valid_scores) :]


def occurrence_adjust(scores, indices, epsilon, salt):
    """Break ties created when target resampling repeats the same example."""
    values = np.asarray(scores[indices], dtype=np.float64).copy()
    token = np.arange(len(values), dtype=np.float64)
    if salt % 2:
        token = token[::-1]
    jitter = (token + 1.0) / (len(values) + 1.0) - 0.5
    return values + epsilon * jitter


def f1_at_threshold(labels, scores, threshold):
    prediction = scores >= threshold
    return float(
        2.0 * np.sum(prediction & labels)
        / max(1, int(prediction.sum() + labels.sum()))
    )


def macro_auc(labels, scores, indices):
    return float(
        np.mean(
            [
                roc_auc_score(labels[indices, class_id], scores[indices, class_id])
                for class_id in range(labels.shape[1])
            ]
        )
    )


def summarize(records, gate_key):
    switched = np.asarray([row[gate_key] > 0.0 for row in records])
    population_gain = np.asarray(
        [row["receiver_population_f1"] - row["reference_population_f1"] for row in records]
    )
    batch_gain = np.asarray(
        [row["receiver_f1"] - row["reference_f1"] for row in records]
    )
    deployed_population = np.where(
        switched,
        [row["receiver_population_f1"] for row in records],
        [row["reference_population_f1"] for row in records],
    )
    deployed_batch = np.where(
        switched,
        [row["receiver_f1"] for row in records],
        [row["reference_f1"] for row in records],
    )
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
        "reference_population_f1": float(
            np.mean([row["reference_population_f1"] for row in records])
        ),
        "receiver_population_f1": float(
            np.mean([row["receiver_population_f1"] for row in records])
        ),
        "deployed_population_f1": float(np.mean(deployed_population)),
        "reference_batch_f1": float(np.mean([row["reference_f1"] for row in records])),
        "receiver_batch_f1": float(np.mean([row["receiver_f1"] for row in records])),
        "deployed_batch_f1": float(np.mean(deployed_batch)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-scores", required=True)
    parser.add_argument("--receiver-scores", required=True)
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--target-seeds", type=int, default=50)
    parser.add_argument("--target-size", type=int, default=5000)
    parser.add_argument("--delta", type=float, default=0.05)
    parser.add_argument("--checkpoint-fraction", type=float, default=0.20)
    parser.add_argument("--checkpoint-split-seed", type=int, default=20260820)
    parser.add_argument("--tpr-drift-budget", type=float, default=0.0)
    args = parser.parse_args()

    raw = load_dataset("google-research-datasets/go_emotions", "simplified")
    labels = {
        split: multilabel_matrix(raw[split]["labels"])
        for split in ("train", "validation", "test")
    }
    reference = np.load(args.reference_scores)
    receiver = np.load(args.receiver_scores)
    for split in ("valid", "test"):
        key = f"{split}_labels"
        expected = labels["validation" if split == "valid" else "test"]
        if not np.array_equal(reference[key], expected):
            raise RuntimeError(f"Reference {split} labels are misaligned")
        if not np.array_equal(receiver[key], expected):
            raise RuntimeError(f"Receiver {split} labels are misaligned")

    permutation = np.random.RandomState(args.checkpoint_split_seed).permutation(
        len(labels["validation"])
    )
    checkpoint_end = max(1, int(round(args.checkpoint_fraction * len(permutation))))
    checkpoint_indices = np.sort(permutation[:checkpoint_end])
    certificate_indices = np.sort(permutation[checkpoint_end:])
    for archive in (reference, receiver):
        if "checkpoint_indices" in archive.files:
            if not np.array_equal(archive["checkpoint_indices"], checkpoint_indices):
                raise RuntimeError("Saved checkpoint split does not match requested protocol")
            if not np.array_equal(archive["certificate_indices"], certificate_indices):
                raise RuntimeError("Saved certificate split does not match requested protocol")

    valid_reference = reference["valid_logits"].astype(np.float64)
    test_reference = reference["test_logits"].astype(np.float64)
    valid_receiver = receiver["valid_logits"].astype(np.float64)
    test_receiver = receiver["test_logits"].astype(np.float64)
    for class_id in range(labels["validation"].shape[1]):
        valid_reference[:, class_id], test_reference[:, class_id] = break_ties(
            valid_reference[:, class_id], test_reference[:, class_id], 2 * class_id
        )
        valid_receiver[:, class_id], test_receiver[:, class_id] = break_ties(
            valid_receiver[:, class_id], test_receiver[:, class_id], 2 * class_id + 1
        )

    reference_auc = macro_auc(labels["validation"], valid_reference, checkpoint_indices)
    receiver_auc = macro_auc(labels["validation"], valid_receiver, checkpoint_indices)
    source_prior = labels["train"].mean(axis=0)
    classes = [
        class_id
        for class_id in range(labels["train"].shape[1])
        if labels["validation"][checkpoint_indices, class_id].sum() >= 3
        and labels["validation"][certificate_indices, class_id].sum() >= 15
    ]

    records = []
    diagnostics = []
    for position, class_id in enumerate(classes):
        reference_threshold = best_threshold(
            labels["validation"][checkpoint_indices, class_id],
            valid_reference[checkpoint_indices, class_id],
        )
        source_positive = labels["validation"][certificate_indices, class_id]
        source_positive_reference = valid_reference[certificate_indices, class_id][source_positive]
        combined_receiver = np.concatenate(
            [valid_receiver[:, class_id], test_receiver[:, class_id]]
        )
        receiver_gaps = np.diff(np.unique(combined_receiver))
        receiver_gaps = receiver_gaps[receiver_gaps > 0.0]
        occurrence_epsilon = (
            float(receiver_gaps.min() / 4.0) if len(receiver_gaps) else 1e-9
        )
        source_positive_indices = certificate_indices[source_positive]
        source_positive_receiver = occurrence_adjust(
            valid_receiver[:, class_id],
            source_positive_indices,
            occurrence_epsilon,
            class_id,
        )
        diagnostics.append(
            {
                "class_id": class_id,
                "source_positive_count": int(source_positive.sum()),
                "reference_selector_auc": float(
                    roc_auc_score(
                        labels["validation"][checkpoint_indices, class_id],
                        valid_reference[checkpoint_indices, class_id],
                    )
                ),
                "receiver_selector_auc": float(
                    roc_auc_score(
                        labels["validation"][checkpoint_indices, class_id],
                        valid_receiver[checkpoint_indices, class_id],
                    )
                ),
            }
        )
        target_labels = labels["test"][:, class_id]
        positive = np.flatnonzero(target_labels)
        negative = np.flatnonzero(~target_labels)
        for magnitude in MAGNITUDES:
            target_prior = min(0.40, float(source_prior[class_id]) * magnitude)
            for target_seed in range(args.target_seeds):
                seed = 10_000_000 * class_id + 10 * target_seed
                calibration_sample = sample_mixture_batch(
                    positive, negative, args.target_size, target_prior, np.random.RandomState(seed)
                )
                rate_sample = sample_mixture_batch(
                    positive, negative, args.target_size, target_prior, np.random.RandomState(seed + 1)
                )
                evaluation_sample = sample_mixture_batch(
                    positive, negative, args.target_size, target_prior, np.random.RandomState(seed + 2)
                )
                reference_count = int(
                    np.sum(test_reference[calibration_sample, class_id] >= reference_threshold)
                )
                calibration_receiver = occurrence_adjust(
                    test_receiver[:, class_id],
                    calibration_sample,
                    occurrence_epsilon,
                    seed,
                )
                if reference_count == 0:
                    receiver_threshold = float("inf")
                elif reference_count == len(calibration_receiver):
                    receiver_threshold = float("-inf")
                else:
                    receiver_threshold = float(
                        np.partition(
                            calibration_receiver, len(calibration_receiver) - reference_count
                        )[len(calibration_receiver) - reference_count]
                    )
                paired = paired_rate_certificate(
                    source_positive_reference,
                    source_positive_receiver,
                    reference_threshold,
                    receiver_threshold,
                    test_reference[rate_sample, class_id],
                    occurrence_adjust(
                        test_receiver[:, class_id],
                        rate_sample,
                        occurrence_epsilon,
                        seed + 1,
                    ),
                    args.delta,
                    args.tpr_drift_budget,
                )
                fully_paired = fully_paired_rate_certificate(
                    source_positive_reference,
                    source_positive_receiver,
                    reference_threshold,
                    receiver_threshold,
                    test_reference[rate_sample, class_id],
                    occurrence_adjust(
                        test_receiver[:, class_id],
                        rate_sample,
                        occurrence_epsilon,
                        seed + 1,
                    ),
                    args.delta,
                    args.tpr_drift_budget,
                )
                reference_f1 = f1_at_threshold(
                    target_labels[evaluation_sample],
                    test_reference[evaluation_sample, class_id],
                    reference_threshold,
                )
                receiver_f1 = f1_at_threshold(
                    target_labels[evaluation_sample],
                    occurrence_adjust(
                        test_receiver[:, class_id],
                        evaluation_sample,
                        occurrence_epsilon,
                        seed + 2,
                    ),
                    receiver_threshold,
                )
                reference_population_f1 = population_f1(
                    test_reference[positive, class_id],
                    test_reference[negative, class_id],
                    reference_threshold,
                    target_prior,
                )
                receiver_population_f1 = population_f1(
                    occurrence_adjust(
                        test_receiver[:, class_id],
                        positive,
                        occurrence_epsilon,
                        class_id + 1000,
                    ),
                    occurrence_adjust(
                        test_receiver[:, class_id],
                        negative,
                        occurrence_epsilon,
                        class_id + 2000,
                    ),
                    receiver_threshold,
                    target_prior,
                )
                records.append(
                    {
                        "class_id": class_id,
                        "magnitude": magnitude,
                        "target_seed": target_seed,
                        "target_prior": target_prior,
                        "reference_count": reference_count,
                        "receiver_count": int(np.sum(calibration_receiver >= receiver_threshold)),
                        "reference_f1": reference_f1,
                        "receiver_f1": receiver_f1,
                        "reference_population_f1": reference_population_f1,
                        "receiver_population_f1": receiver_population_f1,
                        **paired,
                        **fully_paired,
                    }
                )
        print(f"class={position + 1}/{len(classes)} elapsed={time.time() - START:.1f}s", flush=True)

    count_mismatches = int(sum(row["reference_count"] != row["receiver_count"] for row in records))
    summary = {
        "protocol": {
            "reference_scores": args.reference_scores,
            "receiver_scores": args.receiver_scores,
            "target_labels_used_for_selection_or_gate": False,
            "checkpoint_examples": int(len(checkpoint_indices)),
            "certificate_examples": int(len(certificate_indices)),
            "checkpoint_split_seed": args.checkpoint_split_seed,
            "target_seeds": args.target_seeds,
            "target_size": args.target_size,
            "delta": args.delta,
            "classes": classes,
        },
        "source_selector": {
            "reference_macro_auc": reference_auc,
            "receiver_macro_auc": receiver_auc,
            "receiver_selected_by_macro_auc": bool(receiver_auc > reference_auc),
        },
        "exact_count_mismatches": count_mismatches,
        "paired": summarize(records, "paired_cp_gamma"),
        "fully_paired": summarize(records, "fully_paired_cp_gamma"),
        "by_magnitude": {
            str(magnitude): {
                "paired": summarize(
                    [row for row in records if row["magnitude"] == magnitude],
                    "paired_cp_gamma",
                ),
                "fully_paired": summarize(
                    [row for row in records if row["magnitude"] == magnitude],
                    "fully_paired_cp_gamma",
                ),
            }
            for magnitude in MAGNITUDES
        },
        "class_diagnostics": diagnostics,
    }
    prefix = Path(args.output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    prefix.with_suffix(".json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    np.save(prefix.with_suffix(".npy"), {"records": records, "class_diagnostics": diagnostics}, allow_pickle=True)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
