"""Run the frozen factorized C-RM protocol on an independent feature archive."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

from analyze_factorized_certificate import factorized_gamma, reconstruct_counts


MAGNITUDES = (0.5, 1.0, 2.0, 3.0, 5.0)


def hash_order(image_ids: np.ndarray, seed: int) -> np.ndarray:
    keys = [
        hashlib.sha256(f"{seed}:{image_id}".encode("utf-8")).digest()
        for image_id in image_ids
    ]
    return np.asarray(sorted(range(len(keys)), key=keys.__getitem__), dtype=int)


def split_indices(
    dataset: str,
    image_ids: np.ndarray,
    splits: np.ndarray,
    seed: int,
    certificate_extra_size: int | None = None,
) -> dict[str, np.ndarray]:
    if dataset in ("voc2007", "voc2007_aug2012"):
        source = np.where(np.char.startswith(splits.astype(str), "trainval"))[0]
        target = np.where(splits == "test")[0]
        ordered = source[hash_order(image_ids[source], seed)]
        head_end = int(0.50 * len(ordered))
        threshold_end = head_end + int(0.20 * len(ordered))
        return {
            "head": ordered[:head_end],
            "threshold": ordered[head_end:threshold_end],
            "certificate": ordered[threshold_end:],
            "target": target,
        }
    if dataset == "voc2007_certaug2012":
        source07 = np.where(splits == "trainval")[0]
        certificate12 = np.where(splits == "trainval2012")[0]
        target = np.where(splits == "test")[0]
        ordered = source07[hash_order(image_ids[source07], seed)]
        ordered12 = certificate12[
            hash_order(image_ids[certificate12], seed + 1)
        ]
        if certificate_extra_size is not None:
            if certificate_extra_size < 0:
                raise ValueError("certificate_extra_size must be nonnegative")
            ordered12 = ordered12[:certificate_extra_size]
        head_end = int(0.50 * len(ordered))
        threshold_end = head_end + int(0.20 * len(ordered))
        return {
            "head": ordered[:head_end],
            "threshold": ordered[head_end:threshold_end],
            "certificate": np.concatenate(
                [ordered[threshold_end:], ordered12]
            ),
            "target": target,
            "prior_source": source07,
        }
    if dataset == "eurosat":
        ordered = hash_order(image_ids, seed)
        head_end = int(0.40 * len(ordered))
        threshold_end = head_end + int(0.15 * len(ordered))
        certificate_end = threshold_end + int(0.25 * len(ordered))
        return {
            "head": ordered[:head_end],
            "threshold": ordered[head_end:threshold_end],
            "certificate": ordered[threshold_end:certificate_end],
            "target": ordered[certificate_end:],
        }
    if dataset == "cifar10_1":
        source = np.where(splits == "source_train")[0]
        target = np.where(splits == "target")[0]
        ordered = source[hash_order(image_ids[source], seed)]
        head_end = int(0.50 * len(ordered))
        threshold_end = head_end + int(0.20 * len(ordered))
        return {
            "head": ordered[:head_end],
            "threshold": ordered[head_end:threshold_end],
            "certificate": ordered[threshold_end:],
            "target": target,
        }
    raise ValueError(dataset)


def best_threshold(labels: np.ndarray, scores: np.ndarray) -> float:
    thresholds = np.quantile(scores, np.linspace(0.5, 0.9999, 200))
    values = [f1_at_threshold(labels, scores, threshold) for threshold in thresholds]
    return float(thresholds[int(np.argmax(values))])


def threshold_at_count(scores: np.ndarray, count: int) -> float:
    count = min(max(int(count), 0), len(scores))
    if count == 0:
        return float("inf")
    if count == len(scores):
        return float("-inf")
    ordered = np.sort(scores)
    lower = float(ordered[-count - 1])
    upper = float(ordered[-count])
    if lower < upper:
        return (lower + upper) / 2.0
    include_ties = upper
    exclude_ties = float(np.nextafter(upper, np.inf))
    include_count = int(np.sum(scores >= include_ties))
    exclude_count = int(np.sum(scores >= exclude_ties))
    if abs(exclude_count - count) <= abs(include_count - count):
        return exclude_ties
    return include_ties


def f1_at_threshold(labels: np.ndarray, scores: np.ndarray, threshold: float) -> float:
    prediction = scores >= threshold
    true_positive = int(np.sum(prediction & (labels == 1)))
    return 2.0 * true_positive / max(1, int(prediction.sum()) + int(labels.sum()))


def population_f1(
    positive_scores: np.ndarray,
    negative_scores: np.ndarray,
    threshold: float,
    target_prior: float,
) -> float:
    true_positive_rate = float(np.mean(positive_scores >= threshold))
    false_positive_rate = float(np.mean(negative_scores >= threshold))
    action_rate = (
        target_prior * true_positive_rate
        + (1.0 - target_prior) * false_positive_rate
    )
    return (
        2.0 * target_prior * true_positive_rate
        / max(target_prior + action_rate, 1e-15)
    )


def sample_mixture(
    positive_indices: np.ndarray,
    negative_indices: np.ndarray,
    size: int,
    prior: float,
    seed: int,
) -> np.ndarray:
    rng = np.random.RandomState(seed)
    positive_count = int(size * prior)
    negative_count = size - positive_count
    sampled = np.concatenate(
        [
            rng.choice(
                positive_indices,
                positive_count,
                replace=len(positive_indices) < positive_count,
            ),
            rng.choice(
                negative_indices,
                negative_count,
                replace=len(negative_indices) < negative_count,
            ),
        ]
    )
    rng.shuffle(sampled)
    return sampled


def fit_scores(
    features: np.ndarray,
    labels: np.ndarray,
    head_indices: np.ndarray,
) -> np.ndarray:
    scores = np.empty((len(features), labels.shape[1]), dtype=np.float32)
    for class_id in range(labels.shape[1]):
        head = LogisticRegression(
            C=1.0,
            class_weight="balanced",
            max_iter=2000,
            solver="liblinear",
        ).fit(features[head_indices], labels[head_indices, class_id])
        scores[:, class_id] = head.decision_function(features)
    return scores


def summarize(
    rows: list[dict],
    switched: np.ndarray,
    family_switched: np.ndarray,
    class_names: np.ndarray,
) -> dict:
    population_gain = np.asarray(
        [row["receiver_population_f1"] - row["reference_population_f1"] for row in rows]
    )
    batch_gain = np.asarray(
        [row["receiver_f1"] - row["reference_f1"] for row in rows]
    )
    switched_classes = sorted(
        {int(row["class_id"]) for row, use in zip(rows, switched) if use}
    )
    family_classes = sorted(
        {int(row["class_id"]) for row, use in zip(rows, family_switched) if use}
    )
    by_magnitude = {}
    magnitudes = np.asarray([row["magnitude"] for row in rows], dtype=float)
    for magnitude in MAGNITUDES:
        mask = magnitudes == magnitude
        by_magnitude[str(magnitude)] = {
            "comparisons": int(mask.sum()),
            "switches": int(np.sum(mask & switched)),
            "population_losses": int(
                np.sum(mask & switched & (population_gain < -1e-12))
            ),
            "finite_batch_losses": int(
                np.sum(mask & switched & (batch_gain < -1e-12))
            ),
            "mean_reference_population_f1": float(
                np.mean([row["reference_population_f1"] for row, use in zip(rows, mask) if use])
            ),
            "mean_receiver_population_f1": float(
                np.mean([row["receiver_population_f1"] for row, use in zip(rows, mask) if use])
            ),
        }
    return {
        "comparisons": len(rows),
        "switches": int(switched.sum()),
        "switch_rate": float(switched.mean()),
        "covered_classes": len(switched_classes),
        "covered_class_names": [str(class_names[index]) for index in switched_classes],
        "population_losses": int(np.sum(switched & (population_gain < -1e-12))),
        "finite_batch_losses": int(np.sum(switched & (batch_gain < -1e-12))),
        "mean_population_gain_among_switches": (
            float(np.mean(population_gain[switched])) if switched.any() else None
        ),
        "family_wise": {
            "switches": int(family_switched.sum()),
            "covered_classes": len(family_classes),
            "population_losses": int(
                np.sum(family_switched & (population_gain < -1e-12))
            ),
            "finite_batch_losses": int(
                np.sum(family_switched & (batch_gain < -1e-12))
            ),
        },
        "by_magnitude": by_magnitude,
    }


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--target-size", type=int, default=2000)
    parser.add_argument("--target-seeds", type=int, default=100)
    parser.add_argument("--split-seed", type=int, default=20260820)
    parser.add_argument("--certificate-extra-size", type=int)
    parser.add_argument("--delta", type=float, default=0.05)
    parser.add_argument(
        "--analysis-label",
        choices=("confirmatory", "exploratory"),
        default="confirmatory",
    )
    parser.add_argument(
        "--npy",
        type=Path,
        default=root / "results" / "generated" / "independent_factorized.npy",
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=root / "results" / "generated" / "independent_factorized.json",
    )
    args = parser.parse_args()
    started = time.time()

    archive = np.load(args.features, allow_pickle=False)
    reference_features = archive["reference"].astype(np.float32)
    receiver_features = archive["receiver"].astype(np.float32)
    labels = archive["labels"].astype(np.int8)
    image_ids = archive["image_ids"].astype(str)
    splits = archive["splits"].astype(str)
    class_names = archive["class_names"].astype(str)
    dataset = str(archive["dataset"].item())
    indices = split_indices(
        dataset,
        image_ids,
        splits,
        args.split_seed,
        args.certificate_extra_size,
    )

    reference_scores = fit_scores(reference_features, labels, indices["head"])
    receiver_scores = fit_scores(receiver_features, labels, indices["head"])
    source_indices = indices.get(
        "prior_source",
        np.concatenate(
            [indices["head"], indices["threshold"], indices["certificate"]]
        ),
    )
    source_priors = labels[source_indices].mean(axis=0)
    rows: list[dict] = []

    for class_id in range(labels.shape[1]):
        threshold_labels = labels[indices["threshold"], class_id]
        reference_threshold = best_threshold(
            threshold_labels, reference_scores[indices["threshold"], class_id]
        )
        certificate_positive_indices = indices["certificate"][
            labels[indices["certificate"], class_id] == 1
        ]
        target_positive_indices = indices["target"][
            labels[indices["target"], class_id] == 1
        ]
        target_negative_indices = indices["target"][
            labels[indices["target"], class_id] == 0
        ]
        if not len(target_positive_indices) or not len(target_negative_indices):
            raise RuntimeError(f"Class {class_names[class_id]} lacks a target label cell")

        for magnitude_index, magnitude in enumerate(MAGNITUDES):
            target_prior = min(0.5, float(source_priors[class_id]) * magnitude)
            for target_seed in range(args.target_seeds):
                base_seed = (
                    args.split_seed
                    + class_id * 1_000_000
                    + magnitude_index * 10_000
                    + target_seed * 10
                )
                calibration = sample_mixture(
                    target_positive_indices,
                    target_negative_indices,
                    args.target_size,
                    target_prior,
                    base_seed + 1,
                )
                rate = sample_mixture(
                    target_positive_indices,
                    target_negative_indices,
                    args.target_size,
                    target_prior,
                    base_seed + 2,
                )
                evaluation = sample_mixture(
                    target_positive_indices,
                    target_negative_indices,
                    args.target_size,
                    target_prior,
                    base_seed + 3,
                )
                reference_count = int(
                    np.sum(
                        reference_scores[calibration, class_id] >= reference_threshold
                    )
                )
                receiver_threshold = threshold_at_count(
                    receiver_scores[calibration, class_id], reference_count
                )
                receiver_calibration_count = int(
                    np.sum(
                        receiver_scores[calibration, class_id] >= receiver_threshold
                    )
                )

                source_reference_decisions = (
                    reference_scores[certificate_positive_indices, class_id]
                    >= reference_threshold
                )
                source_receiver_decisions = (
                    receiver_scores[certificate_positive_indices, class_id]
                    >= receiver_threshold
                )
                source_positive_count = len(certificate_positive_indices)
                reference_tpr = float(np.mean(source_reference_decisions))
                receiver_tpr = float(np.mean(source_receiver_decisions))
                source_disagreement_rate = float(
                    np.mean(source_reference_decisions != source_receiver_decisions)
                )

                rate_reference_decisions = (
                    reference_scores[rate, class_id] >= reference_threshold
                )
                rate_receiver_decisions = (
                    receiver_scores[rate, class_id] >= receiver_threshold
                )
                target_reference_rate = float(np.mean(rate_reference_decisions))
                target_receiver_rate = float(np.mean(rate_receiver_decisions))
                target_disagreement_rate = float(
                    np.mean(rate_reference_decisions != rate_receiver_decisions)
                )

                evaluation_labels = labels[evaluation, class_id]
                reference_f1 = f1_at_threshold(
                    evaluation_labels,
                    reference_scores[evaluation, class_id],
                    reference_threshold,
                )
                receiver_f1 = f1_at_threshold(
                    evaluation_labels,
                    receiver_scores[evaluation, class_id],
                    receiver_threshold,
                )
                reference_population_f1 = population_f1(
                    reference_scores[target_positive_indices, class_id],
                    reference_scores[target_negative_indices, class_id],
                    reference_threshold,
                    target_prior,
                )
                receiver_population_f1 = population_f1(
                    receiver_scores[target_positive_indices, class_id],
                    receiver_scores[target_negative_indices, class_id],
                    receiver_threshold,
                    target_prior,
                )
                rows.append(
                    {
                        "class_id": class_id,
                        "class_name": str(class_names[class_id]),
                        "head_seed": 0,
                        "target_seed": target_seed,
                        "magnitude": magnitude,
                        "target_prior": target_prior,
                        "source_positive_count": source_positive_count,
                        "reference_tpr": reference_tpr,
                        "receiver_tpr": receiver_tpr,
                        "fully_paired_cp_source_disagreement_rate": source_disagreement_rate,
                        "paired_cp_target_reference_rate": target_reference_rate,
                        "paired_cp_target_receiver_rate": target_receiver_rate,
                        "fully_paired_cp_target_disagreement_rate": target_disagreement_rate,
                        "target_reference_rate": target_reference_rate,
                        "target_receiver_rate": target_receiver_rate,
                        "reference_threshold": reference_threshold,
                        "receiver_threshold": receiver_threshold,
                        "calibration_reference_count": reference_count,
                        "calibration_receiver_count": receiver_calibration_count,
                        "calibration_count_error": (
                            receiver_calibration_count - reference_count
                        ),
                        "reference_f1": reference_f1,
                        "receiver_f1": receiver_f1,
                        "reference_population_f1": reference_population_f1,
                        "receiver_population_f1": receiver_population_f1,
                    }
                )
        print(
            f"class={class_id + 1}/{labels.shape[1]} name={class_names[class_id]} "
            f"source_positives={len(certificate_positive_indices)} elapsed={time.time() - started:.0f}s",
            flush=True,
        )

    counts = reconstruct_counts(rows, args.target_size)
    gamma = factorized_gamma(counts, args.delta)
    switched = gamma > 0.0
    family_switched = factorized_gamma(counts, args.delta / labels.shape[1]) > 0.0
    for row, value, use, family_use in zip(rows, gamma, switched, family_switched):
        row["factorized_gamma"] = float(value)
        row["factorized_switch"] = bool(use)
        row["family_wise_switch"] = bool(family_use)
    summary = summarize(rows, switched, family_switched, class_names)
    useful_threshold = 0.01 if dataset.startswith("voc2007") else 0.05
    minimum_classes = 5
    useful_criterion_met = bool(
        summary["population_losses"] == 0
        and summary["switch_rate"] >= useful_threshold
        and summary["covered_classes"] >= minimum_classes
    )
    if args.analysis_label == "confirmatory":
        summary["pre_registered_useful_replication"] = useful_criterion_met
    else:
        summary["exploratory_useful_criterion_met"] = useful_criterion_met

    payload = {
        "protocol": {
            "dataset": dataset,
            "features": str(args.features),
            "class_count": labels.shape[1],
            "class_names": class_names.tolist(),
            "split_seed": args.split_seed,
            "split_sizes": {name: len(value) for name, value in indices.items()},
            "certificate_extra_size": args.certificate_extra_size,
            "target_size": args.target_size,
            "target_seeds": args.target_seeds,
            "magnitudes": MAGNITUDES,
            "delta": args.delta,
            "analysis_label": args.analysis_label,
            "risk_allocation": "six one-sided exact-binomial bounds at delta/6",
            "target_labels_in_gate": False,
            "tie_policy": (
                "nearest achievable score threshold; ties prefer underfill; "
                "realized target-rate mismatch is certified"
            ),
        },
        "summary": summary,
        "records": rows,
    }
    args.npy.parent.mkdir(parents=True, exist_ok=True)
    args.json.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.npy, payload, allow_pickle=True)
    args.json.write_text(
        json.dumps(
            {"protocol": payload["protocol"], "summary": summary}, indent=2
        ),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    print(f"saved: {args.npy}")
    print(f"saved: {args.json}")


if __name__ == "__main__":
    main()
