"""Evaluate C-RM after independent source-only candidate selection."""

from pathlib import Path
import argparse
import json
import time

import numpy as np

from bench_coco_deep_transfer import (
    MAGNITUDES,
    best_threshold,
    f1_at_k,
    f1_at_threshold,
    predict,
    train_head,
)
from bench_coco_crm_certificate import (
    certificate,
    fully_paired_rate_certificate,
    paired_rate_certificate,
    population_f1,
    sample_mixture_batch,
    summarize,
    threshold_at_count,
)


START = time.time()
CANDIDATES = ("dinov2", "convnext")


def load_aligned_features(paths):
    data = {name: np.load(path, mmap_mode="r") for name, path in paths.items()}
    anchor = data[CANDIDATES[0]]
    for name in CANDIDATES[1:]:
        if not np.array_equal(anchor["labels"], data[name]["labels"]):
            raise RuntimeError(f"Labels do not align for {name}")
        if not np.array_equal(anchor["image_ids"], data[name]["image_ids"]):
            raise RuntimeError(f"Image IDs do not align for {name}")
        if not np.array_equal(anchor["reference"], data[name]["reference"]):
            raise RuntimeError(f"Reference features do not align for {name}")
    return data


def validate_classes(labels, train_indices, selector_indices, class_ids):
    classes = [int(value) for value in class_ids.split(",")]
    if len(classes) != len(set(classes)):
        raise ValueError("class-ids contains duplicates")
    if any(class_id < 0 or class_id >= labels.shape[1] for class_id in classes):
        raise ValueError("class-ids contains an out-of-range class")
    for class_id in classes:
        if np.unique(labels[train_indices, class_id]).size < 2:
            raise RuntimeError(f"Class {class_id} is degenerate in the train split")
        if np.unique(labels[selector_indices, class_id]).size < 2:
            raise RuntimeError(f"Class {class_id} is degenerate in the selector split")
    return classes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dinov2-features", required=True)
    parser.add_argument("--convnext-features", required=True)
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--class-ids", required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--pos-weight-cap", type=float, default=20.0)
    parser.add_argument("--head-seeds", type=int, default=3)
    parser.add_argument("--target-seeds", type=int, default=100)
    parser.add_argument("--target-size", type=int, default=5000)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--train-fraction", type=float, default=0.50)
    parser.add_argument("--selector-fraction", type=float, default=0.05)
    parser.add_argument("--certificate-fraction", type=float, default=0.25)
    parser.add_argument("--target-fraction", type=float, default=0.20)
    parser.add_argument("--delta", type=float, default=0.05)
    parser.add_argument("--tpr-drift-budget", type=float, default=0.0)
    parser.add_argument("--head-type", choices=("linear", "mlp"), default="linear")
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)
    args = parser.parse_args()

    if not 0.0 < args.delta < 1.0:
        raise ValueError("delta must be in (0, 1)")
    fractions = (
        args.train_fraction,
        args.selector_fraction,
        args.certificate_fraction,
        args.target_fraction,
    )
    if any(value <= 0.0 or value >= 1.0 for value in fractions):
        raise ValueError("All split fractions must be in (0, 1)")
    if sum(fractions) > 1.0 + 1e-12:
        raise ValueError("Split fractions must sum to at most 1")

    data = load_aligned_features(
        {
            "dinov2": args.dinov2_features,
            "convnext": args.convnext_features,
        }
    )
    labels = data["dinov2"]["labels"].astype(bool)
    reference_features = data["dinov2"]["reference"]
    candidate_features = {name: data[name]["receiver"] for name in CANDIDATES}

    permutation = np.random.RandomState(args.split_seed).permutation(len(labels))
    train_end = int(args.train_fraction * len(labels))
    selector_end = train_end + int(args.selector_fraction * len(labels))
    certificate_end = selector_end + int(args.certificate_fraction * len(labels))
    target_start = len(labels) - int(args.target_fraction * len(labels))
    if certificate_end > target_start:
        raise RuntimeError("Certificate and target splits overlap")
    train_indices = permutation[:train_end]
    selector_indices = permutation[train_end:selector_end]
    certificate_indices = permutation[selector_end:certificate_end]
    target_pool_indices = permutation[target_start:]
    classes = validate_classes(
        labels, train_indices, selector_indices, args.class_ids
    )
    source_prior = labels[train_indices].mean(axis=0)
    records = []
    selection_diagnostics = []

    print(
        f"images={len(labels)} train={len(train_indices)} "
        f"selector={len(selector_indices)} certificate={len(certificate_indices)} "
        f"target_pool={len(target_pool_indices)} classes={len(classes)}",
        flush=True,
    )
    for head_seed in range(args.head_seeds):
        reference_head, reference_auc = train_head(
            reference_features,
            labels,
            train_indices,
            selector_indices,
            classes,
            args,
            100 + head_seed,
        )
        candidate_heads = {}
        candidate_aucs = {}
        for candidate_index, name in enumerate(CANDIDATES):
            candidate_heads[name], candidate_aucs[name] = train_head(
                candidate_features[name],
                labels,
                train_indices,
                selector_indices,
                classes,
                args,
                1000 + 100 * candidate_index + head_seed,
            )

        selector_reference_scores = predict(
            reference_head, reference_features[selector_indices]
        )
        certificate_reference_scores = predict(
            reference_head, reference_features[certificate_indices]
        )
        target_reference_scores = predict(
            reference_head, reference_features[target_pool_indices]
        )
        selector_candidate_scores = {
            name: predict(candidate_heads[name], candidate_features[name][selector_indices])
            for name in CANDIDATES
        }
        certificate_candidate_scores = {
            name: predict(
                candidate_heads[name], candidate_features[name][certificate_indices]
            )
            for name in CANDIDATES
        }
        target_candidate_scores = {
            name: predict(
                candidate_heads[name], candidate_features[name][target_pool_indices]
            )
            for name in CANDIDATES
        }

        for class_id in classes:
            selector_labels = labels[selector_indices, class_id]
            reference_threshold = best_threshold(
                selector_labels, selector_reference_scores[:, class_id]
            )
            selector_reference_count = int(
                np.sum(selector_reference_scores[:, class_id] >= reference_threshold)
            )
            selector_f1 = {
                name: f1_at_k(
                    selector_labels,
                    selector_candidate_scores[name][:, class_id],
                    selector_reference_count,
                )
                for name in CANDIDATES
            }
            selected_candidate = max(CANDIDATES, key=lambda name: selector_f1[name])
            selection_diagnostics.append(
                {
                    "head_seed": head_seed,
                    "class_id": class_id,
                    "selected_candidate": selected_candidate,
                    "selector_reference_count": selector_reference_count,
                    "selector_candidate_f1": selector_f1,
                }
            )

            certificate_receiver_scores = certificate_candidate_scores[
                selected_candidate
            ]
            target_receiver_scores = target_candidate_scores[selected_candidate]
            source_labels = labels[certificate_indices, class_id]
            source_positive_reference = certificate_reference_scores[
                source_labels, class_id
            ]
            source_positive_receiver = certificate_receiver_scores[
                source_labels, class_id
            ]
            source_reference_count = int(
                np.sum(
                    certificate_reference_scores[:, class_id] >= reference_threshold
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

            target_labels_all = labels[target_pool_indices, class_id]
            positive = np.flatnonzero(target_labels_all)
            negative = np.flatnonzero(~target_labels_all)
            population_positive_reference = target_reference_scores[positive, class_id]
            population_negative_reference = target_reference_scores[negative, class_id]
            population_positive_receiver = target_receiver_scores[positive, class_id]
            population_negative_receiver = target_receiver_scores[negative, class_id]

            for magnitude in MAGNITUDES:
                target_prior = min(0.40, float(source_prior[class_id]) * magnitude)
                for target_seed in range(args.target_seeds):
                    seed = 10_000_000 * class_id + 100_000 * head_seed + 10 * target_seed
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
                        evaluation_labels, evaluation_reference, reference_threshold
                    )
                    receiver_f1 = f1_at_threshold(
                        evaluation_labels, evaluation_receiver, receiver_threshold
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
                            "selected_candidate": selected_candidate,
                            "magnitude": magnitude,
                            "target_prior": target_prior,
                            "calibration_positive_count": int(
                                target_labels_all[calibration_sample].sum()
                            ),
                            "rate_positive_count": int(
                                target_labels_all[rate_sample].sum()
                            ),
                            "evaluation_positive_count": int(evaluation_labels.sum()),
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
                                receiver_f1 if paired_switch else reference_f1
                            ),
                            "crm_fully_paired_cp_f1": (
                                receiver_f1 if fully_paired_switch else reference_f1
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
        selected_counts = {
            name: sum(
                row["head_seed"] == head_seed
                and row["selected_candidate"] == name
                for row in selection_diagnostics
            )
            for name in CANDIDATES
        }
        print(
            f"head_seed={head_seed} ref_auc={reference_auc:.5f} "
            f"candidate_auc={candidate_aucs} selected={selected_counts} "
            f"elapsed={time.time()-START:.0f}s",
            flush=True,
        )

    summary = {
        "protocol": {
            "candidate_selection": "source-only selector F1 at reference count",
            "candidates": {
                "dinov2": args.dinov2_features,
                "convnext": args.convnext_features,
            },
            "classes": classes,
            "head_seeds": args.head_seeds,
            "target_seeds": args.target_seeds,
            "target_size": args.target_size,
            "delta": args.delta,
            "split_seed": args.split_seed,
            "train_fraction": args.train_fraction,
            "selector_fraction": args.selector_fraction,
            "certificate_fraction": args.certificate_fraction,
            "target_fraction": args.target_fraction,
        },
        "magnitudes": summarize(records),
        "selection_diagnostics": selection_diagnostics,
    }
    prefix = Path(args.output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    prefix.with_suffix(".json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    np.save(
        prefix.with_suffix(".npy"),
        {"records": records, "selection_diagnostics": selection_diagnostics},
        allow_pickle=True,
    )
    print(json.dumps(summary["magnitudes"], indent=2), flush=True)
    print(f"saved={prefix} elapsed={time.time()-START:.0f}s", flush=True)


if __name__ == "__main__":
    main()
