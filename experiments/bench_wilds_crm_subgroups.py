"""Evaluate C-RM on natural CivilComments identity subpopulation shifts."""

from pathlib import Path
import argparse
import json
import time

import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import roc_auc_score

from bench_coco_crm_certificate import (
    fully_paired_rate_certificate,
    paired_rate_certificate,
)


SPLIT_CODES = {"train": 0, "val": 1, "test": 2}


def f1_score(labels, predictions):
    labels = np.asarray(labels, dtype=bool)
    predictions = np.asarray(predictions, dtype=bool)
    true_positive = int(np.sum(labels & predictions))
    return 2.0 * true_positive / max(1, int(labels.sum() + predictions.sum()))


def best_threshold(labels, scores):
    candidates = np.unique(np.quantile(scores, np.linspace(0.5, 0.9995, 400)))
    values = [f1_score(labels, scores >= threshold) for threshold in candidates]
    return float(candidates[int(np.argmax(values))])


def matched_threshold(reference_scores, receiver_scores, reference_threshold):
    target_count = int(np.sum(reference_scores >= reference_threshold))
    if target_count <= 0:
        return float("inf")
    if target_count >= len(receiver_scores):
        return float("-inf")
    ordered = np.sort(receiver_scores)[::-1]
    return float((ordered[target_count - 1] + ordered[target_count]) / 2.0)


def train_receiver(features, labels, train_indices, valid_indices, args):
    device = torch.device(args.device)
    train_x = torch.from_numpy(
        np.asarray(features[train_indices], dtype=np.float32)
    ).to(device)
    train_y = torch.from_numpy(labels[train_indices].astype(np.float32)).to(device)
    valid_x = torch.from_numpy(
        np.asarray(features[valid_indices], dtype=np.float32)
    ).to(device)
    positives = float(train_y.sum().item())
    pos_weight = min(args.pos_weight_cap, (len(train_y) - positives) / max(1.0, positives))
    model = torch.nn.Linear(train_x.shape[1], 1).to(device)
    torch.manual_seed(args.seed)
    torch.nn.init.normal_(model.weight, std=0.01)
    torch.nn.init.zeros_(model.bias)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    criterion = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight], device=device)
    )
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    best_auc = -np.inf
    best_state = None
    for epoch in range(args.epochs):
        order = torch.randperm(len(train_y), generator=generator)
        model.train()
        for start in range(0, len(order), args.batch_size):
            batch = order[start : start + args.batch_size].to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(train_x[batch]).squeeze(1)
            loss = criterion(logits, train_y[batch])
            loss.backward()
            optimizer.step()
        if epoch == 0 or (epoch + 1) % 5 == 0 or epoch + 1 == args.epochs:
            model.eval()
            with torch.inference_mode():
                valid_scores = model(valid_x).squeeze(1).float().cpu().numpy()
            auc = roc_auc_score(labels[valid_indices], valid_scores)
            print(
                f"receiver_epoch={epoch + 1}/{args.epochs} "
                f"loss={loss.item():.5f} valid_auc={auc:.5f}",
                flush=True,
            )
            if auc > best_auc:
                best_auc = float(auc)
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }
    model.load_state_dict(best_state)
    model.eval()
    return model, best_auc


def summarize(records, group_names):
    output = {}
    for group_name in group_names:
        rows = [row for row in records if row["group"] == group_name]
        switched = np.asarray([row["gamma"] > 0.0 for row in rows])
        gains = np.asarray([row["receiver_f1"] - row["reference_f1"] for row in rows])
        output[group_name] = {
            "comparisons": len(rows),
            "reference_f1": float(np.mean([row["reference_f1"] for row in rows])),
            "rm_f1": float(np.mean([row["receiver_f1"] for row in rows])),
            "crm_f1": float(np.mean([row["crm_f1"] for row in rows])),
            "switch_count": int(switched.sum()),
            "unsafe_switch_count": int(np.sum(switched & (gains < 0.0))),
            "conditional_gain": (
                float(np.mean(gains[switched])) if switched.any() else None
            ),
            "source_positive_count": int(rows[0]["source_positive_count"]),
            "target_group_size": int(rows[0]["target_group_size"]),
        }
    all_switched = np.asarray([row["gamma"] > 0.0 for row in records])
    all_gains = np.asarray(
        [row["receiver_f1"] - row["reference_f1"] for row in records]
    )
    output["overall"] = {
        "comparisons": len(records),
        "reference_f1": float(np.mean([row["reference_f1"] for row in records])),
        "rm_f1": float(np.mean([row["receiver_f1"] for row in records])),
        "crm_f1": float(np.mean([row["crm_f1"] for row in records])),
        "switch_count": int(all_switched.sum()),
        "unsafe_switch_count": int(np.sum(all_switched & (all_gains < 0.0))),
        "conditional_gain": (
            float(np.mean(all_gains[all_switched])) if all_switched.any() else None
        ),
    }
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--features")
    parser.add_argument("--receiver-scores")
    parser.add_argument("--receiver-name")
    parser.add_argument(
        "--certificate",
        choices=("paired-rate", "fully-paired"),
        default="paired-rate",
    )
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--target-seeds", type=int, default=50)
    parser.add_argument("--include-all-domain", action="store_true")
    parser.add_argument("--delta", type=float, default=0.05)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--pos-weight-cap", type=float, default=20.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if (args.features is None) == (args.receiver_scores is None):
        raise ValueError("Provide exactly one of --features or --receiver-scores")

    start_time = time.time()
    metadata = np.load(args.metadata, allow_pickle=False)
    labels = metadata["toxicity"].astype(np.int8)
    groups = metadata["groups"].astype(bool)
    split = metadata["split"]
    group_names = [str(value) for value in metadata["identity_columns"]]
    frame = pd.read_csv(
        args.csv, usecols=["comment_text", "split"], low_memory=False
    )
    frame = frame[frame["split"].isin(SPLIT_CODES)].reset_index(drop=True)
    texts = frame["comment_text"].fillna("").astype(str).to_numpy(dtype=object)
    if len(labels) != len(texts):
        raise ValueError("Metadata and text rows are not aligned")

    train_indices = np.flatnonzero(split == 0)
    valid_indices = np.flatnonzero(split == 1)
    test_indices = np.flatnonzero(split == 2)
    split_rng = np.random.RandomState(args.seed)
    shuffled_valid = valid_indices.copy()
    split_rng.shuffle(shuffled_valid)
    threshold_valid = shuffled_valid[: len(shuffled_valid) // 2]
    certificate_valid = shuffled_valid[len(shuffled_valid) // 2 :]

    if args.features is not None:
        features = np.load(args.features, mmap_mode="r")
        if len(features) != len(labels):
            raise ValueError("Feature and metadata rows are not aligned")
        receiver, receiver_valid_auc = train_receiver(
            features, labels, train_indices, valid_indices, args
        )
        receiver_name = (
            args.receiver_name
            or "ModernBERT-base frozen embedding linear head"
        )
    else:
        saved_scores = np.load(args.receiver_scores, allow_pickle=False)
        if not np.array_equal(
            saved_scores["valid_labels"].reshape(-1), labels[valid_indices]
        ) or not np.array_equal(
            saved_scores["test_labels"].reshape(-1), labels[test_indices]
        ):
            raise ValueError("Saved receiver scores are not aligned")
        valid_receiver = saved_scores["valid_logits"].astype(np.float32).reshape(-1)
        test_receiver = saved_scores["test_logits"].astype(np.float32).reshape(-1)
        receiver_valid_auc = float(
            roc_auc_score(labels[valid_indices], valid_receiver)
        )
        receiver_name = (
            args.receiver_name
            or "ModernBERT-base source-fine-tuned classifier"
        )

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=3,
        max_features=100_000,
        sublinear_tf=True,
        dtype=np.float32,
    )
    train_tfidf = vectorizer.fit_transform(texts[train_indices])
    valid_tfidf = vectorizer.transform(texts[valid_indices])
    test_tfidf = vectorizer.transform(texts[test_indices])
    reference = SGDClassifier(
        loss="log_loss",
        alpha=1e-5,
        max_iter=30,
        tol=1e-4,
        class_weight="balanced",
        random_state=args.seed,
        n_jobs=-1,
    ).fit(train_tfidf, labels[train_indices])
    valid_reference = reference.decision_function(valid_tfidf)
    test_reference = reference.decision_function(test_tfidf)

    if args.features is not None:
        device = torch.device(args.device)
        with torch.inference_mode():
            valid_receiver = receiver(
                torch.from_numpy(
                    np.asarray(features[valid_indices], dtype=np.float32)
                ).to(device)
            ).squeeze(1).float().cpu().numpy()
            test_receiver = receiver(
                torch.from_numpy(
                    np.asarray(features[test_indices], dtype=np.float32)
                ).to(device)
            ).squeeze(1).float().cpu().numpy()

    valid_position = {int(index): position for position, index in enumerate(valid_indices)}
    threshold_positions = np.asarray(
        [valid_position[int(index)] for index in threshold_valid], dtype=np.int64
    )
    reference_threshold = best_threshold(
        labels[threshold_valid], valid_reference[threshold_positions]
    )
    test_groups = groups[test_indices]
    valid_groups = groups[valid_indices]
    certificate_positions = np.asarray(
        [valid_position[int(index)] for index in certificate_valid], dtype=np.int64
    )

    records = []
    group_specs = list(enumerate(group_names))
    if args.include_all_domain:
        group_specs.insert(0, (-1, "all_test"))
    for group_index, group_name in group_specs:
        if group_index < 0:
            target_positions = np.arange(len(test_indices), dtype=np.int64)
            source_positive_positions = certificate_positions[
                labels[valid_indices[certificate_positions]] == 1
            ]
        else:
            target_positions = np.flatnonzero(test_groups[:, group_index])
            source_positive_positions = certificate_positions[
                valid_groups[certificate_positions, group_index]
                & (labels[valid_indices[certificate_positions]] == 1)
            ]
        if len(target_positions) < 300 or len(source_positive_positions) < 20:
            print(
                f"skip_group={group_name} target={len(target_positions)} "
                f"source_positives={len(source_positive_positions)}",
                flush=True,
            )
            continue
        for target_seed in range(args.target_seeds):
            seed_offset = 100_000 if group_index < 0 else 1000 * (group_index + 1)
            rng = np.random.RandomState(seed_offset + target_seed)
            order = target_positions.copy()
            rng.shuffle(order)
            threshold_end = max(1, int(0.4 * len(order)))
            certificate_end = max(threshold_end + 1, int(0.7 * len(order)))
            target_threshold_positions = order[:threshold_end]
            target_certificate_positions = order[threshold_end:certificate_end]
            target_evaluation_positions = order[certificate_end:]
            receiver_threshold = matched_threshold(
                test_reference[target_threshold_positions],
                test_receiver[target_threshold_positions],
                reference_threshold,
            )
            certificate_args = (
                valid_reference[source_positive_positions],
                valid_receiver[source_positive_positions],
                reference_threshold,
                receiver_threshold,
                test_reference[target_certificate_positions],
                test_receiver[target_certificate_positions],
                args.delta,
                0.0,
            )
            if args.certificate == "fully-paired":
                certificate = fully_paired_rate_certificate(*certificate_args)
                gamma = certificate["fully_paired_cp_gamma"]
            else:
                certificate = paired_rate_certificate(*certificate_args)
                gamma = certificate["paired_cp_gamma"]
            evaluation_labels = labels[test_indices[target_evaluation_positions]]
            reference_prediction = (
                test_reference[target_evaluation_positions] >= reference_threshold
            )
            receiver_prediction = (
                test_receiver[target_evaluation_positions] >= receiver_threshold
            )
            reference_f1 = f1_score(evaluation_labels, reference_prediction)
            receiver_f1 = f1_score(evaluation_labels, receiver_prediction)
            switched = gamma > 0.0
            records.append(
                {
                    "group": group_name,
                    "target_seed": target_seed,
                    "target_group_size": len(target_positions),
                    "source_positive_count": len(source_positive_positions),
                    "reference_f1": reference_f1,
                    "receiver_f1": receiver_f1,
                    "crm_f1": receiver_f1 if switched else reference_f1,
                    "gamma": gamma,
                    "receiver_threshold": receiver_threshold,
                    "reference_threshold": reference_threshold,
                    **certificate,
                }
            )
        print(
            f"group={group_name} target={len(target_positions)} "
            f"source_positives={len(source_positive_positions)} "
            f"elapsed={time.time() - start_time:.0f}s",
            flush=True,
        )

    summary = summarize(records, sorted({row["group"] for row in records}))
    output_prefix = Path(args.output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    np.save(
        output_prefix.with_suffix(".npy"),
        {"records": records, "summary": summary},
        allow_pickle=True,
    )
    output_prefix.with_suffix(".json").write_text(
        json.dumps(
            {
                "protocol": {
                    "dataset": "WILDS CivilComments",
                    "target_domain": "official test identity subgroups",
                    "reference": "TF-IDF SGD logistic",
                    "receiver": receiver_name,
                    "target_seeds": args.target_seeds,
                    "delta": args.delta,
                    "certificate": args.certificate,
                    "target_labels_used_by_gate": False,
                    "include_all_domain": args.include_all_domain,
                    "receiver_valid_auc": receiver_valid_auc,
                },
                "groups": summary,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2), flush=True)
    print(f"saved={output_prefix} elapsed={time.time() - start_time:.0f}s", flush=True)


if __name__ == "__main__":
    main()
