"""Evaluate prior-free certified rate matching on saved COCO features."""

from pathlib import Path
import argparse
import json
import math
import time

import numpy as np
from scipy.stats import beta as beta_distribution

from bench_coco_deep_transfer import (
    MAGNITUDES,
    best_threshold,
    f1_at_k,
    f1_at_threshold,
    predict,
    select_classes,
    train_head,
)


START = time.time()


def sample_binary_batch(positive, negative, size, positive_count, rng):
    indices = np.concatenate(
        [
            rng.choice(positive, positive_count, replace=len(positive) < positive_count),
            rng.choice(
                negative,
                size - positive_count,
                replace=len(negative) < size - positive_count,
            ),
        ]
    )
    rng.shuffle(indices)
    return indices


def sample_mixture_batch(positive, negative, size, positive_probability, rng):
    positive_count = int(rng.binomial(size, positive_probability))
    return sample_binary_batch(positive, negative, size, positive_count, rng)


def threshold_at_count(scores, count):
    count = min(max(int(count), 0), len(scores))
    if count == 0:
        return math.inf
    if count == len(scores):
        return -math.inf
    return float(np.partition(scores, len(scores) - count)[len(scores) - count])


def population_f1(positive_scores, negative_scores, threshold, positive_prior):
    true_positive_rate = float(np.mean(positive_scores >= threshold))
    false_positive_rate = float(np.mean(negative_scores >= threshold))
    decision_rate = (
        positive_prior * true_positive_rate
        + (1.0 - positive_prior) * false_positive_rate
    )
    denominator = positive_prior + decision_rate
    if denominator == 0.0:
        return 0.0
    return float(2.0 * positive_prior * true_positive_rate / denominator)


def clopper_pearson_lower(successes, trials, alpha):
    if successes <= 0:
        return 0.0
    return float(beta_distribution.ppf(alpha, successes, trials - successes + 1))


def clopper_pearson_upper(successes, trials, alpha):
    if successes >= trials:
        return 1.0
    return float(beta_distribution.ppf(1.0 - alpha, successes + 1, trials - successes))


def certificate(
    source_positive_reference,
    source_positive_receiver,
    reference_threshold,
    receiver_threshold,
    target_reference_rate,
    target_receiver_rate,
    target_size,
    delta,
):
    trials = len(source_positive_reference)
    if trials == 0:
        return {
            "source_positive_count": 0,
            "reference_tpr": None,
            "receiver_tpr": None,
            "target_reference_rate": target_reference_rate,
            "target_receiver_rate": target_receiver_rate,
            "target_epsilon": math.sqrt(
                math.log(8.0 / delta) / (2.0 * target_size)
            ),
            "tie_error": abs(target_receiver_rate - target_reference_rate),
            "dkw_delta_lower": None,
            "dkw_gamma": -1.0,
            "cp_delta_lower": None,
            "cp_gamma": -1.0,
        }
    reference_successes = int(
        np.sum(source_positive_reference >= reference_threshold)
    )
    receiver_successes = int(
        np.sum(source_positive_receiver >= receiver_threshold)
    )
    reference_tpr = reference_successes / trials
    receiver_tpr = receiver_successes / trials
    target_epsilon = math.sqrt(math.log(8.0 / delta) / (2.0 * target_size))
    tie_error = abs(target_receiver_rate - target_reference_rate)
    rate_error = 2.0 * target_epsilon + tie_error
    rate_lower = max(0.0, target_reference_rate - target_epsilon)

    source_epsilon = math.sqrt(math.log(8.0 / delta) / (2.0 * trials))
    dkw_delta_lower = receiver_tpr - reference_tpr - 2.0 * source_epsilon
    dkw_reference_upper = min(1.0, reference_tpr + source_epsilon)
    dkw_gamma = rate_lower * dkw_delta_lower - dkw_reference_upper * rate_error

    alpha = delta / 4.0
    receiver_lower = clopper_pearson_lower(receiver_successes, trials, alpha)
    reference_upper = clopper_pearson_upper(reference_successes, trials, alpha)
    cp_delta_lower = receiver_lower - reference_upper
    cp_gamma = rate_lower * cp_delta_lower - reference_upper * rate_error
    return {
        "source_positive_count": trials,
        "reference_tpr": reference_tpr,
        "receiver_tpr": receiver_tpr,
        "target_reference_rate": target_reference_rate,
        "target_receiver_rate": target_receiver_rate,
        "target_epsilon": target_epsilon,
        "tie_error": tie_error,
        "dkw_delta_lower": dkw_delta_lower,
        "dkw_gamma": dkw_gamma,
        "cp_delta_lower": cp_delta_lower,
        "cp_gamma": cp_gamma,
    }


def paired_rate_certificate(
    source_positive_reference,
    source_positive_receiver,
    reference_threshold,
    receiver_threshold,
    target_reference_scores,
    target_receiver_scores,
    delta,
    tpr_drift_budget,
):
    source_trials = len(source_positive_reference)
    target_trials = len(target_reference_scores)
    if source_trials == 0 or target_trials == 0:
        return {
            "paired_cp_delta_lower": None,
            "paired_cp_delta_lower_unadjusted": None,
            "paired_cp_reference_tpr_upper": None,
            "paired_cp_reference_rate_lower": None,
            "paired_cp_rate_difference_upper": None,
            "paired_cp_gamma": -1.0,
            "paired_cp_target_reference_rate": None,
            "paired_cp_target_receiver_rate": None,
            "paired_cp_disagreement_rate": None,
        }

    alpha = delta / 5.0
    reference_source_decisions = source_positive_reference >= reference_threshold
    receiver_source_decisions = source_positive_receiver >= receiver_threshold
    reference_successes = int(reference_source_decisions.sum())
    receiver_successes = int(receiver_source_decisions.sum())
    receiver_lower = clopper_pearson_lower(
        receiver_successes, source_trials, alpha
    )
    reference_upper = clopper_pearson_upper(
        reference_successes, source_trials, alpha
    )
    delta_lower_unadjusted = receiver_lower - reference_upper
    delta_lower = delta_lower_unadjusted - tpr_drift_budget

    reference_target_decisions = target_reference_scores >= reference_threshold
    receiver_target_decisions = target_receiver_scores >= receiver_threshold
    reference_target_successes = int(reference_target_decisions.sum())
    receiver_target_successes = int(receiver_target_decisions.sum())
    receiver_only = int(
        np.sum(receiver_target_decisions & ~reference_target_decisions)
    )
    reference_only = int(
        np.sum(reference_target_decisions & ~receiver_target_decisions)
    )
    reference_rate_lower = clopper_pearson_lower(
        reference_target_successes, target_trials, alpha
    )
    receiver_only_upper = clopper_pearson_upper(
        receiver_only, target_trials, alpha
    )
    reference_only_lower = clopper_pearson_lower(
        reference_only, target_trials, alpha
    )
    rate_difference_upper = receiver_only_upper - reference_only_lower
    gamma = -1.0
    if delta_lower > 0.0:
        gamma = (
            reference_rate_lower * delta_lower
            - reference_upper * max(0.0, rate_difference_upper)
        )
    return {
        "paired_cp_delta_lower": delta_lower,
        "paired_cp_delta_lower_unadjusted": delta_lower_unadjusted,
        "paired_cp_reference_tpr_upper": reference_upper,
        "paired_cp_reference_rate_lower": reference_rate_lower,
        "paired_cp_rate_difference_upper": rate_difference_upper,
        "paired_cp_gamma": gamma,
        "paired_cp_target_reference_rate": (
            reference_target_successes / target_trials
        ),
        "paired_cp_target_receiver_rate": (
            receiver_target_successes / target_trials
        ),
        "paired_cp_disagreement_rate": (
            (receiver_only + reference_only) / target_trials
        ),
    }


def fully_paired_rate_certificate(
    source_positive_reference,
    source_positive_receiver,
    reference_threshold,
    receiver_threshold,
    target_reference_scores,
    target_receiver_scores,
    delta,
    tpr_drift_budget,
):
    source_trials = len(source_positive_reference)
    target_trials = len(target_reference_scores)
    if source_trials == 0 or target_trials == 0:
        return {
            "fully_paired_cp_delta_lower": None,
            "fully_paired_cp_delta_lower_unadjusted": None,
            "fully_paired_cp_reference_tpr_upper": None,
            "fully_paired_cp_reference_rate_lower": None,
            "fully_paired_cp_rate_difference_upper": None,
            "fully_paired_cp_gamma": -1.0,
            "fully_paired_cp_source_disagreement_rate": None,
            "fully_paired_cp_target_disagreement_rate": None,
        }

    alpha = delta / 6.0
    reference_source = source_positive_reference >= reference_threshold
    receiver_source = source_positive_receiver >= receiver_threshold
    source_receiver_only = int(np.sum(receiver_source & ~reference_source))
    source_reference_only = int(np.sum(reference_source & ~receiver_source))
    reference_source_successes = int(reference_source.sum())
    source_receiver_only_lower = clopper_pearson_lower(
        source_receiver_only, source_trials, alpha
    )
    source_reference_only_upper = clopper_pearson_upper(
        source_reference_only, source_trials, alpha
    )
    delta_lower_unadjusted = (
        source_receiver_only_lower - source_reference_only_upper
    )
    delta_lower = delta_lower_unadjusted - tpr_drift_budget
    reference_tpr_upper = clopper_pearson_upper(
        reference_source_successes, source_trials, alpha
    )

    reference_target = target_reference_scores >= reference_threshold
    receiver_target = target_receiver_scores >= receiver_threshold
    reference_target_successes = int(reference_target.sum())
    target_receiver_only = int(np.sum(receiver_target & ~reference_target))
    target_reference_only = int(np.sum(reference_target & ~receiver_target))
    reference_rate_lower = clopper_pearson_lower(
        reference_target_successes, target_trials, alpha
    )
    target_receiver_only_upper = clopper_pearson_upper(
        target_receiver_only, target_trials, alpha
    )
    target_reference_only_lower = clopper_pearson_lower(
        target_reference_only, target_trials, alpha
    )
    rate_difference_upper = (
        target_receiver_only_upper - target_reference_only_lower
    )
    gamma = -1.0
    if delta_lower > 0.0:
        gamma = (
            reference_rate_lower * delta_lower
            - reference_tpr_upper * max(0.0, rate_difference_upper)
        )
    return {
        "fully_paired_cp_delta_lower": delta_lower,
        "fully_paired_cp_delta_lower_unadjusted": delta_lower_unadjusted,
        "fully_paired_cp_reference_tpr_upper": reference_tpr_upper,
        "fully_paired_cp_reference_rate_lower": reference_rate_lower,
        "fully_paired_cp_rate_difference_upper": rate_difference_upper,
        "fully_paired_cp_gamma": gamma,
        "fully_paired_cp_source_disagreement_rate": (
            (source_receiver_only + source_reference_only) / source_trials
        ),
        "fully_paired_cp_target_disagreement_rate": (
            (target_receiver_only + target_reference_only) / target_trials
        ),
    }


def summarize(records):
    output = {}
    for magnitude in MAGNITUDES:
        rows = [row for row in records if row["magnitude"] == magnitude]
        magnitude_summary = {}
        for method in (
            "reference_f1",
            "receiver_f1",
            "source_local_selector_f1",
            "crm_dkw_f1",
            "crm_cp_f1",
            "crm_paired_cp_f1",
            "crm_fully_paired_cp_f1",
            "oracle_f1",
            "reference_population_f1",
            "receiver_population_f1",
            "crm_paired_cp_population_f1",
            "crm_fully_paired_cp_population_f1",
            "oracle_population_f1",
        ):
            values = np.asarray([row[method] for row in rows], dtype=np.float64)
            magnitude_summary[method] = float(values.mean())
        for name, gamma_key in (
            ("dkw", "dkw_gamma"),
            ("cp", "cp_gamma"),
            ("paired_cp", "paired_cp_gamma"),
            ("fully_paired_cp", "fully_paired_cp_gamma"),
        ):
            switched = np.asarray([row[gamma_key] > 0.0 for row in rows])
            realized_gain = np.asarray(
                [row["receiver_f1"] - row["reference_f1"] for row in rows]
            )
            population_gain = np.asarray(
                [
                    row["receiver_population_f1"]
                    - row["reference_population_f1"]
                    for row in rows
                ]
            )
            magnitude_summary[f"{name}_switch_rate"] = float(switched.mean())
            magnitude_summary[f"{name}_switch_count"] = int(switched.sum())
            magnitude_summary[f"{name}_false_certificate_rate"] = (
                float(np.mean(population_gain[switched] < 0.0))
                if switched.any()
                else None
            )
            unsafe = switched & (population_gain < 0.0)
            magnitude_summary[f"{name}_unsafe_switch_rate"] = float(unsafe.mean())
            magnitude_summary[f"{name}_unsafe_switch_count"] = int(unsafe.sum())
            magnitude_summary[f"{name}_conditional_gain"] = (
                float(np.mean(population_gain[switched])) if switched.any() else None
            )
            realized_unsafe = switched & (realized_gain < 0.0)
            magnitude_summary[f"{name}_realized_unsafe_switch_rate"] = float(
                realized_unsafe.mean()
            )
            magnitude_summary[f"{name}_realized_unsafe_switch_count"] = int(
                realized_unsafe.sum()
            )
            magnitude_summary[f"{name}_realized_conditional_gain"] = (
                float(np.mean(realized_gain[switched])) if switched.any() else None
            )
        output[str(magnitude)] = magnitude_summary
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True)
    parser.add_argument("--target-features")
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--pos-weight-cap", type=float, default=20.0)
    parser.add_argument("--classes", type=int, default=30)
    parser.add_argument("--class-ids")
    parser.add_argument("--head-seeds", type=int, default=3)
    parser.add_argument("--target-seeds", type=int, default=20)
    parser.add_argument("--target-size", type=int, default=5000)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--threshold-fraction", type=float, default=0.05)
    parser.add_argument("--certificate-fraction", type=float, default=0.05)
    parser.add_argument("--target-fraction", type=float, default=0.20)
    parser.add_argument("--delta", type=float, default=0.10)
    parser.add_argument("--tpr-drift-budget", type=float, default=0.0)
    parser.add_argument("--head-type", choices=("linear", "mlp"), default="linear")
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)
    args = parser.parse_args()

    if not 0.0 < args.delta < 1.0:
        raise ValueError("delta must be in (0, 1)")
    if not 0.0 <= args.tpr_drift_budget < 1.0:
        raise ValueError("tpr-drift-budget must be in [0, 1)")
    fractions = (
        args.train_fraction,
        args.threshold_fraction,
        args.certificate_fraction,
        args.target_fraction,
    )
    if any(value <= 0.0 or value >= 1.0 for value in fractions):
        raise ValueError("All split fractions must be in (0, 1)")
    if sum(fractions) > 1.0 + 1e-12:
        raise ValueError("Split fractions must sum to at most 1")
    data = np.load(args.features, mmap_mode="r")
    reference_features = data["reference"]
    receiver_features = data["receiver"]
    labels = data["labels"].astype(bool)
    target_reference_features = reference_features
    target_receiver_features = receiver_features
    if args.target_features:
        target_data = np.load(args.target_features, mmap_mode="r")
        if not np.array_equal(labels, target_data["labels"].astype(bool)):
            raise RuntimeError("Source and target labels do not align")
        if not np.array_equal(data["image_ids"], target_data["image_ids"]):
            raise RuntimeError("Source and target image IDs do not align")
        target_reference_features = target_data["reference"]
        target_receiver_features = target_data["receiver"]

    permutation = np.random.RandomState(args.split_seed).permutation(len(labels))
    train_end = int(args.train_fraction * len(labels))
    threshold_end = train_end + int(args.threshold_fraction * len(labels))
    certificate_end = threshold_end + int(args.certificate_fraction * len(labels))
    target_start = len(labels) - int(args.target_fraction * len(labels))
    if certificate_end > target_start:
        raise RuntimeError("Certificate and target splits overlap")
    train_indices = permutation[:train_end]
    threshold_indices = permutation[train_end:threshold_end]
    certificate_indices = permutation[threshold_end:certificate_end]
    target_pool_indices = permutation[target_start:]
    if args.class_ids:
        classes = [int(value) for value in args.class_ids.split(",")]
        if len(classes) != len(set(classes)):
            raise ValueError("class-ids contains duplicates")
        if any(class_id < 0 or class_id >= labels.shape[1] for class_id in classes):
            raise ValueError("class-ids contains an out-of-range class")
        for class_id in classes:
            if np.unique(labels[train_indices, class_id]).size < 2:
                raise RuntimeError(f"Class {class_id} is degenerate in the train split")
            if np.unique(labels[threshold_indices, class_id]).size < 2:
                raise RuntimeError(
                    f"Class {class_id} is degenerate in the threshold split"
                )
    else:
        classes = select_classes(
            labels[train_indices], labels[threshold_indices], args.classes
        )
        if len(classes) < args.classes:
            raise RuntimeError(f"Only {len(classes)} classes satisfy source selection")
    source_prior = labels[train_indices].mean(axis=0)
    records = []
    class_diagnostics = []

    print(
        f"images={len(labels)} train={len(train_indices)} "
        f"threshold={len(threshold_indices)} certificate={len(certificate_indices)} "
        f"target_pool={len(target_pool_indices)} classes={len(classes)}",
        flush=True,
    )
    for head_seed in range(args.head_seeds):
        reference_head, reference_auc = train_head(
            reference_features,
            labels,
            train_indices,
            threshold_indices,
            classes,
            args,
            100 + head_seed,
        )
        receiver_head, receiver_auc = train_head(
            receiver_features,
            labels,
            train_indices,
            threshold_indices,
            classes,
            args,
            1000 + head_seed,
        )
        threshold_reference_scores = predict(
            reference_head, reference_features[threshold_indices]
        )
        certificate_reference_scores = predict(
            reference_head, reference_features[certificate_indices]
        )
        certificate_receiver_scores = predict(
            receiver_head, receiver_features[certificate_indices]
        )
        target_reference_scores = predict(
            reference_head, target_reference_features[target_pool_indices]
        )
        target_receiver_scores = predict(
            receiver_head, target_receiver_features[target_pool_indices]
        )

        for class_id in classes:
            reference_threshold = best_threshold(
                labels[threshold_indices, class_id],
                threshold_reference_scores[:, class_id],
            )
            source_labels = labels[certificate_indices, class_id]
            source_positive_reference = certificate_reference_scores[
                source_labels, class_id
            ]
            source_positive_receiver = certificate_receiver_scores[
                source_labels, class_id
            ]
            source_reference_count = int(
                np.sum(
                    certificate_reference_scores[:, class_id]
                    >= reference_threshold
                )
            )
            source_local_gain = f1_at_k(
                source_labels,
                certificate_receiver_scores[:, class_id],
                source_reference_count,
            ) - f1_at_k(
                source_labels,
                certificate_reference_scores[:, class_id],
                source_reference_count,
            )
            class_diagnostics.append(
                {
                    "head_seed": head_seed,
                    "class_id": class_id,
                    "source_prior": float(source_prior[class_id]),
                    "source_positive_count": int(source_labels.sum()),
                    "source_local_gain": float(source_local_gain),
                }
            )
            target_labels_all = labels[target_pool_indices, class_id]
            positive = np.flatnonzero(target_labels_all)
            negative = np.flatnonzero(~target_labels_all)
            population_positive_reference = target_reference_scores[
                positive, class_id
            ]
            population_negative_reference = target_reference_scores[
                negative, class_id
            ]
            population_positive_receiver = target_receiver_scores[
                positive, class_id
            ]
            population_negative_receiver = target_receiver_scores[
                negative, class_id
            ]
            for magnitude in MAGNITUDES:
                target_prior = min(
                    0.40, float(source_prior[class_id]) * magnitude
                )
                for target_seed in range(args.target_seeds):
                    seed = (
                        10_000_000 * class_id
                        + 100_000 * head_seed
                        + 10 * target_seed
                    )
                    calibration_sample = sample_mixture_batch(
                        positive,
                        negative,
                        args.target_size,
                        target_prior,
                        np.random.RandomState(seed),
                    )
                    rate_sample = sample_mixture_batch(
                        positive,
                        negative,
                        args.target_size,
                        target_prior,
                        np.random.RandomState(seed + 1),
                    )
                    evaluation_sample = sample_mixture_batch(
                        positive,
                        negative,
                        args.target_size,
                        target_prior,
                        np.random.RandomState(seed + 2),
                    )
                    calibration_reference = target_reference_scores[
                        calibration_sample, class_id
                    ]
                    calibration_receiver = target_receiver_scores[
                        calibration_sample, class_id
                    ]
                    reference_count = int(
                        np.sum(calibration_reference >= reference_threshold)
                    )
                    receiver_threshold = threshold_at_count(
                        calibration_receiver, reference_count
                    )
                    reference_rate = reference_count / args.target_size
                    receiver_rate = float(
                        np.mean(calibration_receiver >= receiver_threshold)
                    )
                    cert = certificate(
                        source_positive_reference,
                        source_positive_receiver,
                        reference_threshold,
                        receiver_threshold,
                        reference_rate,
                        receiver_rate,
                        args.target_size,
                        args.delta,
                    )
                    paired_cert = paired_rate_certificate(
                        source_positive_reference,
                        source_positive_receiver,
                        reference_threshold,
                        receiver_threshold,
                        target_reference_scores[rate_sample, class_id],
                        target_receiver_scores[rate_sample, class_id],
                        args.delta,
                        args.tpr_drift_budget,
                    )
                    fully_paired_cert = fully_paired_rate_certificate(
                        source_positive_reference,
                        source_positive_receiver,
                        reference_threshold,
                        receiver_threshold,
                        target_reference_scores[rate_sample, class_id],
                        target_receiver_scores[rate_sample, class_id],
                        args.delta,
                        args.tpr_drift_budget,
                    )

                    evaluation_labels = target_labels_all[evaluation_sample]
                    evaluation_reference = target_reference_scores[
                        evaluation_sample, class_id
                    ]
                    evaluation_receiver = target_receiver_scores[
                        evaluation_sample, class_id
                    ]
                    reference_f1 = f1_at_threshold(
                        evaluation_labels,
                        evaluation_reference,
                        reference_threshold,
                    )
                    receiver_f1 = f1_at_threshold(
                        evaluation_labels,
                        evaluation_receiver,
                        receiver_threshold,
                    )
                    reference_population_f1 = population_f1(
                        population_positive_reference,
                        population_negative_reference,
                        reference_threshold,
                        target_prior,
                    )
                    receiver_population_f1 = population_f1(
                        population_positive_receiver,
                        population_negative_receiver,
                        receiver_threshold,
                        target_prior,
                    )
                    paired_switch = paired_cert["paired_cp_gamma"] > 0.0
                    fully_paired_switch = (
                        fully_paired_cert["fully_paired_cp_gamma"] > 0.0
                    )
                    records.append(
                        {
                            "head_seed": head_seed,
                            "target_seed": target_seed,
                            "class_id": class_id,
                            "magnitude": magnitude,
                            "target_prior": target_prior,
                            "calibration_positive_count": int(
                                target_labels_all[calibration_sample].sum()
                            ),
                            "rate_positive_count": int(
                                target_labels_all[rate_sample].sum()
                            ),
                            "evaluation_positive_count": int(
                                evaluation_labels.sum()
                            ),
                            "reference_f1": reference_f1,
                            "receiver_f1": receiver_f1,
                            "source_local_selector_f1": (
                                receiver_f1 if source_local_gain > 0.0 else reference_f1
                            ),
                            "crm_dkw_f1": (
                                receiver_f1 if cert["dkw_gamma"] > 0.0 else reference_f1
                            ),
                            "crm_cp_f1": (
                                receiver_f1 if cert["cp_gamma"] > 0.0 else reference_f1
                            ),
                            "crm_paired_cp_f1": (
                                receiver_f1
                                if paired_switch
                                else reference_f1
                            ),
                            "crm_fully_paired_cp_f1": (
                                receiver_f1
                                if fully_paired_switch
                                else reference_f1
                            ),
                            "oracle_f1": max(reference_f1, receiver_f1),
                            "reference_population_f1": reference_population_f1,
                            "receiver_population_f1": receiver_population_f1,
                            "crm_paired_cp_population_f1": (
                                receiver_population_f1
                                if paired_switch
                                else reference_population_f1
                            ),
                            "crm_fully_paired_cp_population_f1": (
                                receiver_population_f1
                                if fully_paired_switch
                                else reference_population_f1
                            ),
                            "oracle_population_f1": max(
                                reference_population_f1, receiver_population_f1
                            ),
                            "reference_threshold": reference_threshold,
                            "receiver_threshold": receiver_threshold,
                            **cert,
                            **paired_cert,
                            **fully_paired_cert,
                        }
                    )
        print(
            f"head_seed={head_seed} ref_auc={reference_auc:.5f} "
            f"receiver_auc={receiver_auc:.5f} elapsed={time.time()-START:.0f}s",
            flush=True,
        )

    summary = {
        "protocol": {
            "features": args.features,
            "target_features": args.target_features,
            "classes": classes,
            "head_seeds": args.head_seeds,
            "target_seeds": args.target_seeds,
            "target_size": args.target_size,
            "delta": args.delta,
            "tpr_drift_budget": args.tpr_drift_budget,
            "split_seed": args.split_seed,
            "train_fraction": args.train_fraction,
            "threshold_fraction": args.threshold_fraction,
            "certificate_fraction": args.certificate_fraction,
            "target_fraction": args.target_fraction,
            "head_type": args.head_type,
        },
        "magnitudes": summarize(records),
        "class_diagnostics": class_diagnostics,
    }
    prefix = Path(args.output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    prefix.with_suffix(".json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    np.save(
        prefix.with_suffix(".npy"),
        {"records": records, "class_diagnostics": class_diagnostics},
        allow_pickle=True,
    )
    print(json.dumps(summary["magnitudes"], indent=2), flush=True)
    print(f"saved={prefix} elapsed={time.time()-START:.0f}s", flush=True)


if __name__ == "__main__":
    main()
