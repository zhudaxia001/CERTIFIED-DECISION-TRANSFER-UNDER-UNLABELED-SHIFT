"""Evaluate count--rank transfer from ResNet-50 to DINOv2 on COCO."""

from pathlib import Path
import argparse
import json
import random
import time

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


START = time.time()
MAGNITUDES = (0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0)
METHODS = (
    "reference_fixed_count",
    "receiver_fixed_count",
    "source_local_selector",
    "receiver_source_threshold",
    "oracle_rank_selector",
    "receiver_oracle_count",
)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_head(features, labels, train_indices, valid_indices, classes, args, seed):
    seed_everything(seed)
    device = "cuda:0"
    if args.head_type == "linear":
        model = nn.Linear(features.shape[1], labels.shape[1]).to(device)
    else:
        model = nn.Sequential(
            nn.Linear(features.shape[1], args.hidden_dim),
            nn.GELU(),
            nn.Dropout(args.dropout),
            nn.Linear(args.hidden_dim, labels.shape[1]),
        ).to(device)
    positive = labels[train_indices].sum(axis=0).astype(np.float32)
    negative = len(train_indices) - positive
    pos_weight = np.minimum(negative / np.maximum(positive, 1.0), args.pos_weight_cap)
    pos_weight = torch.tensor(pos_weight, dtype=torch.float32, device=device)
    dataset = TensorDataset(
        torch.from_numpy(features[train_indices].astype(np.float32)),
        torch.from_numpy(labels[train_indices].astype(np.float32)),
    )
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
        pin_memory=True,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    best_auc = -np.inf
    best_state = None
    for epoch in range(args.epochs):
        model.train()
        loss_sum = 0.0
        for batch_features, batch_labels in loader:
            batch_features = batch_features.to(device, non_blocking=True)
            batch_labels = batch_labels.to(device, non_blocking=True)
            logits = model(batch_features)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                logits, batch_labels, pos_weight=pos_weight
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach())
        if epoch == 0 or epoch == args.epochs - 1 or (epoch + 1) % 5 == 0:
            valid_logits = predict(model, features[valid_indices], device)
            valid_auc = macro_auc(labels[valid_indices], valid_logits, classes)
            print(
                f"head_seed={seed} epoch={epoch + 1}/{args.epochs} "
                f"loss={loss_sum / len(loader):.5f} valid_auc={valid_auc:.5f} "
                f"elapsed={time.time() - START:.0f}s",
                flush=True,
            )
            if valid_auc > best_auc:
                best_auc = valid_auc
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model, best_auc


def predict(model, features, device="cuda:0"):
    outputs = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(features), 8192):
            batch = torch.from_numpy(features[start : start + 8192].astype(np.float32)).to(device)
            outputs.append(model(batch).cpu().numpy())
    return np.concatenate(outputs)


def macro_auc(labels, scores, classes=None):
    if classes is None:
        classes = range(labels.shape[1])
    values = [
        roc_auc_score(labels[:, class_id], scores[:, class_id])
        for class_id in classes
        if np.unique(labels[:, class_id]).size == 2
    ]
    if not values:
        raise RuntimeError("No non-degenerate classes are available for AUC")
    return float(np.mean(values))


def f1_at_k(labels, scores, count):
    count = min(max(int(count), 0), len(labels))
    if count == 0:
        return 0.0
    selected = np.argpartition(-scores, count - 1)[:count]
    true_positive = int(labels[selected].sum())
    return 2.0 * true_positive / max(1, count + int(labels.sum()))


def f1_at_threshold(labels, scores, threshold):
    prediction = scores >= threshold
    true_positive = int(np.sum(prediction & labels))
    return 2.0 * true_positive / max(1, int(prediction.sum()) + int(labels.sum()))


def best_threshold(labels, scores):
    order = np.argsort(-scores, kind="stable")
    ordered_labels = labels[order].astype(np.int64)
    true_positive = np.cumsum(ordered_labels)
    counts = np.arange(1, len(labels) + 1)
    f1 = 2.0 * true_positive / np.maximum(1, counts + int(labels.sum()))
    best_count = int(np.argmax(f1)) + 1
    return float(scores[order[best_count - 1]])


def oracle_f1(labels, scores):
    order = np.argsort(-scores, kind="stable")
    true_positive = np.cumsum(labels[order].astype(np.int64))
    counts = np.arange(1, len(labels) + 1)
    return float(np.max(2.0 * true_positive / np.maximum(1, counts + int(labels.sum()))))


def select_classes(train_labels, valid_labels, maximum):
    prevalence = train_labels.mean(axis=0)
    classes = [
        class_id
        for class_id in range(train_labels.shape[1])
        if 0.003 < prevalence[class_id] < 0.20
        and valid_labels[:, class_id].sum() >= 20
    ]
    return sorted(classes, key=lambda class_id: prevalence[class_id])[:maximum]


def summarize(records):
    output = {}
    for magnitude in MAGNITUDES:
        rows = [row for row in records if row["magnitude"] == magnitude]
        output[str(magnitude)] = {
            method: float(np.mean([row[method] for row in rows])) for method in METHODS
        }
        output[str(magnitude)]["fixed_count_gain"] = (
            output[str(magnitude)]["receiver_fixed_count"]
            - output[str(magnitude)]["reference_fixed_count"]
        )
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
    parser.add_argument("--head-seeds", type=int, default=3)
    parser.add_argument("--target-seeds", type=int, default=10)
    parser.add_argument("--target-size", type=int, default=5000)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--head-type", choices=("linear", "mlp"), default="linear")
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)
    args = parser.parse_args()

    data = np.load(args.features, mmap_mode="r")
    reference_features = data["reference"]
    receiver_features = data["receiver"]
    labels = data["labels"].astype(bool)
    image_ids = data["image_ids"]
    target_reference_features = reference_features
    target_receiver_features = receiver_features
    if args.target_features:
        target_data = np.load(args.target_features, mmap_mode="r")
        if not np.array_equal(labels, target_data["labels"].astype(bool)):
            raise RuntimeError("Source and target feature labels do not match")
        if not np.array_equal(image_ids, target_data["image_ids"]):
            raise RuntimeError("Source and target feature image IDs do not match")
        target_reference_features = target_data["reference"]
        target_receiver_features = target_data["receiver"]
    permutation = np.random.RandomState(args.split_seed).permutation(len(labels))
    train_end = int(0.70 * len(labels))
    calibration_end = int(0.75 * len(labels))
    selector_end = int(0.80 * len(labels))
    train_indices = permutation[:train_end]
    calibration_indices = permutation[train_end:calibration_end]
    selector_indices = permutation[calibration_end:selector_end]
    test_indices = permutation[selector_end:]
    classes = select_classes(
        labels[train_indices], labels[calibration_indices], args.classes
    )
    if len(classes) < args.classes:
        raise RuntimeError(f"Only {len(classes)} classes satisfy source-only selection")
    print(
        f"images={len(labels)} train={len(train_indices)} "
        f"calibration={len(calibration_indices)} selector={len(selector_indices)} "
        f"test={len(test_indices)} classes={classes}",
        flush=True,
    )

    records = []
    common_records = []
    diagnostics = []
    source_prior = labels[train_indices].mean(axis=0)
    anchor_candidates = [
        class_id
        for class_id in range(labels.shape[1])
        if labels[test_indices, class_id].any()
        and (~labels[test_indices, class_id]).any()
    ]
    anchor_class = min(
        anchor_candidates, key=lambda class_id: abs(source_prior[class_id] - 0.08)
    )
    for head_seed in range(args.head_seeds):
        reference_head, reference_valid_auc = train_head(
            reference_features,
            labels,
            train_indices,
            calibration_indices,
            classes,
            args,
            100 + head_seed,
        )
        receiver_head, receiver_valid_auc = train_head(
            receiver_features,
            labels,
            train_indices,
            calibration_indices,
            classes,
            args,
            1000 + head_seed,
        )
        valid_reference = predict(reference_head, reference_features[calibration_indices])
        valid_receiver = predict(receiver_head, receiver_features[calibration_indices])
        selector_reference = predict(reference_head, reference_features[selector_indices])
        selector_receiver = predict(receiver_head, receiver_features[selector_indices])
        test_reference = predict(
            reference_head, target_reference_features[test_indices]
        )
        test_receiver = predict(
            receiver_head, target_receiver_features[test_indices]
        )
        thresholds_reference = {
            class_id: best_threshold(
                labels[calibration_indices, class_id], valid_reference[:, class_id]
            )
            for class_id in classes
        }
        thresholds_receiver = {
            class_id: best_threshold(
                labels[calibration_indices, class_id], valid_receiver[:, class_id]
            )
            for class_id in classes
        }

        local_gains = {}
        for class_id in classes:
            valid_y = labels[calibration_indices, class_id]
            selector_y = labels[selector_indices, class_id]
            selector_count = int(
                np.sum(selector_reference[:, class_id] >= thresholds_reference[class_id])
            )
            local_gain = f1_at_k(
                selector_y, selector_receiver[:, class_id], selector_count
            ) - f1_at_k(
                selector_y, selector_reference[:, class_id], selector_count
            )
            local_gains[class_id] = local_gain
            diagnostics.append(
                {
                    "head_seed": head_seed,
                    "class_id": class_id,
                    "source_prior": float(source_prior[class_id]),
                    "local_gain": float(local_gain),
                    "reference_auc": float(roc_auc_score(valid_y, valid_reference[:, class_id])),
                    "receiver_auc": float(roc_auc_score(valid_y, valid_receiver[:, class_id])),
                }
            )
            test_y_all = labels[test_indices, class_id]
            positive_indices = np.flatnonzero(test_y_all)
            negative_indices = np.flatnonzero(~test_y_all)
            if not len(positive_indices) or not len(negative_indices):
                raise RuntimeError(f"Class {class_id} is degenerate in the held-out pool")
            for magnitude in MAGNITUDES:
                target_positive = max(
                    1,
                    int(
                        args.target_size
                        * min(0.40, float(source_prior[class_id]) * magnitude)
                    ),
                )
                for target_seed in range(args.target_seeds):
                    rng = np.random.RandomState(
                        1_000_000 * class_id + 10_000 * head_seed + target_seed
                    )
                    sampled = np.concatenate(
                        [
                            rng.choice(
                                positive_indices,
                                target_positive,
                                replace=len(positive_indices) < target_positive,
                            ),
                            rng.choice(
                                negative_indices,
                                args.target_size - target_positive,
                                replace=len(negative_indices) < args.target_size - target_positive,
                            ),
                        ]
                    )
                    rng.shuffle(sampled)
                    target_y = test_y_all[sampled]
                    reference_scores = test_reference[sampled, class_id]
                    receiver_scores = test_receiver[sampled, class_id]
                    target_count = int(
                        np.sum(reference_scores >= thresholds_reference[class_id])
                    )
                    reference_f1 = f1_at_k(target_y, reference_scores, target_count)
                    receiver_f1 = f1_at_k(target_y, receiver_scores, target_count)
                    row = {
                        "head_seed": head_seed,
                        "target_seed": target_seed,
                        "class_id": class_id,
                        "magnitude": magnitude,
                        "target_count": target_count,
                        "target_positives": int(target_y.sum()),
                        "reference_fixed_count": reference_f1,
                        "receiver_fixed_count": receiver_f1,
                        "source_local_selector": receiver_f1 if local_gain > 0 else reference_f1,
                        "receiver_source_threshold": f1_at_threshold(
                            target_y, receiver_scores, thresholds_receiver[class_id]
                        ),
                        "oracle_rank_selector": max(reference_f1, receiver_f1),
                        "receiver_oracle_count": oracle_f1(target_y, receiver_scores),
                    }
                    records.append(row)

        anchor_y = labels[test_indices, anchor_class]
        anchor_positive = np.flatnonzero(anchor_y)
        anchor_negative = np.flatnonzero(~anchor_y)
        for magnitude in MAGNITUDES:
            target_anchor_positive = max(
                1,
                int(
                    args.target_size
                    * min(0.40, float(source_prior[anchor_class]) * magnitude)
                ),
            )
            for target_seed in range(args.target_seeds):
                rng = np.random.RandomState(
                    50_000_000 + 10_000 * head_seed + target_seed
                )
                sampled = np.concatenate(
                    [
                        rng.choice(
                            anchor_positive,
                            target_anchor_positive,
                            replace=len(anchor_positive) < target_anchor_positive,
                        ),
                        rng.choice(
                            anchor_negative,
                            args.target_size - target_anchor_positive,
                            replace=len(anchor_negative)
                            < args.target_size - target_anchor_positive,
                        ),
                    ]
                )
                rng.shuffle(sampled)
                values = {method: [] for method in METHODS}
                for class_id in classes:
                    target_y = labels[test_indices[sampled], class_id]
                    reference_scores = test_reference[sampled, class_id]
                    receiver_scores = test_receiver[sampled, class_id]
                    target_count = int(
                        np.sum(
                            reference_scores >= thresholds_reference[class_id]
                        )
                    )
                    reference_f1 = f1_at_k(
                        target_y, reference_scores, target_count
                    )
                    receiver_f1 = f1_at_k(
                        target_y, receiver_scores, target_count
                    )
                    values["reference_fixed_count"].append(reference_f1)
                    values["receiver_fixed_count"].append(receiver_f1)
                    values["source_local_selector"].append(
                        receiver_f1
                        if local_gains[class_id] > 0
                        else reference_f1
                    )
                    values["receiver_source_threshold"].append(
                        f1_at_threshold(
                            target_y,
                            receiver_scores,
                            thresholds_receiver[class_id],
                        )
                    )
                    values["oracle_rank_selector"].append(
                        max(reference_f1, receiver_f1)
                    )
                    values["receiver_oracle_count"].append(
                        oracle_f1(target_y, receiver_scores)
                    )
                common_records.append(
                    {
                        "head_seed": head_seed,
                        "target_seed": target_seed,
                        "magnitude": magnitude,
                        "anchor_class": anchor_class,
                        "anchor_prevalence": float(anchor_y[sampled].mean()),
                        **{
                            method: float(np.mean(method_values))
                            for method, method_values in values.items()
                        },
                    }
                )
        print(
            f"completed_head_seed={head_seed} ref_valid_auc={reference_valid_auc:.5f} "
            f"recv_valid_auc={receiver_valid_auc:.5f} elapsed={time.time() - START:.0f}s",
            flush=True,
        )

    summary = {
        "protocol": {
            "features": args.features,
            "target_features": args.target_features,
            "num_images": len(labels),
            "train_images": len(train_indices),
            "calibration_images": len(calibration_indices),
            "selector_images": len(selector_indices),
            "test_images": len(test_indices),
            "classes": classes,
            "common_bag_anchor_class": anchor_class,
            "common_bag_anchor_source_prior": float(source_prior[anchor_class]),
            "head_seeds": args.head_seeds,
            "target_seeds": args.target_seeds,
            "target_size": args.target_size,
            "pos_weight_cap": args.pos_weight_cap,
            "split_seed": args.split_seed,
            "head_type": args.head_type,
            "hidden_dim": args.hidden_dim if args.head_type == "mlp" else None,
            "dropout": args.dropout if args.head_type == "mlp" else None,
            "split_image_id_hash": int(np.bitwise_xor.reduce(image_ids.astype(np.int64))),
        },
        "magnitudes": summarize(records),
        "common_bag_magnitudes": summarize(common_records),
        "diagnostics": diagnostics,
    }
    prefix = Path(args.output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    prefix.with_suffix(".json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    np.save(
        prefix.with_suffix(".npy"),
        {
            "records": records,
            "common_records": common_records,
            "diagnostics": diagnostics,
        },
        allow_pickle=True,
    )
    print(json.dumps(summary["magnitudes"], indent=2), flush=True)
    print(json.dumps(summary["common_bag_magnitudes"], indent=2), flush=True)
    print(f"saved={prefix} elapsed={time.time() - START:.0f}s", flush=True)


if __name__ == "__main__":
    main()
