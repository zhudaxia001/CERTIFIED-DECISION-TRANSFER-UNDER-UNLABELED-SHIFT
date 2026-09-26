"""Evaluate count--rank transfer from TF--IDF to saved deep-model logits."""

from pathlib import Path
import argparse
import json
import time

import numpy as np
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


START = time.time()
MAGS = (1.0, 1.5, 2.0, 3.0, 5.0, 8.0)


def f1_at_k(y, score, k):
    k = min(max(int(k), 0), len(y))
    if k == 0:
        return 0.0
    chosen = np.argpartition(-score, k - 1)[:k]
    return 2.0 * y[chosen].sum() / max(1, k + y.sum())


def best_threshold(y, score):
    candidates = np.unique(np.quantile(score, np.linspace(0.5, 0.9999, 300)))
    values = []
    for threshold in candidates:
        prediction = score >= threshold
        values.append(2.0 * np.sum(prediction & y) / max(1, prediction.sum() + y.sum()))
    return float(candidates[int(np.argmax(values))])


def source_local_gain(y, reference, receiver, rate):
    k = min(max(1, int(round(rate * len(y)))), len(y))
    return f1_at_k(y, receiver, k) - f1_at_k(y, reference, k)


def bootstrap_lcb(y, reference, receiver, rate, seed, repetitions=300):
    rng = np.random.RandomState(seed)
    values = []
    positive = np.flatnonzero(y)
    negative = np.flatnonzero(~y)
    for _ in range(repetitions):
        indices = np.concatenate(
            [
                rng.choice(positive, len(positive), replace=True),
                rng.choice(negative, len(negative), replace=True),
            ]
        )
        values.append(source_local_gain(y[indices], reference[indices], receiver[indices], rate))
    return float(np.quantile(values, 0.05))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--target-size", type=int, default=5000)
    parser.add_argument("--seeds", type=int, default=10)
    args = parser.parse_args()

    raw = load_dataset("google-research-datasets/go_emotions", "simplified")
    texts = {name: np.asarray(raw[name]["text"], dtype=object) for name in ("train", "validation", "test")}
    labels = {}
    for name in ("train", "validation", "test"):
        values = np.zeros((len(raw[name]), 28), dtype=bool)
        for index, row_labels in enumerate(raw[name]["labels"]):
            values[index, row_labels] = True
        labels[name] = values
    deep = np.load(args.scores)
    valid_recv = deep["valid_logits"].astype(np.float32)
    test_recv = deep["test_logits"].astype(np.float32)
    if not np.array_equal(deep["valid_labels"], labels["validation"]):
        raise ValueError("validation logits are misaligned")
    if not np.array_equal(deep["test_labels"], labels["test"]):
        raise ValueError("test logits are misaligned")

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2), min_df=2, max_features=100_000, sublinear_tf=True
    )
    x_train = vectorizer.fit_transform(texts["train"])
    x_valid = vectorizer.transform(texts["validation"])
    x_test = vectorizer.transform(texts["test"])
    prior = labels["train"].mean(axis=0)
    classes = [
        c for c in range(28)
        if 0.001 < prior[c] < 0.08 and labels["validation"][:, c].sum() >= 15
    ]
    classes = sorted(classes, key=lambda c: prior[c])[:20]
    methods = ("reference", "receiver", "local_selector", "lcb_selector", "auc_selector", "oracle_selector")
    results = {method: {mag: [] for mag in MAGS} for method in methods}
    certificate = []

    for position, class_id in enumerate(classes):
        reference = LogisticRegression(
            C=1.0, max_iter=1000, solver="liblinear", random_state=0
        ).fit(x_train, labels["train"][:, class_id])
        valid_ref = reference.decision_function(x_valid)
        test_ref = reference.decision_function(x_test)
        threshold = best_threshold(labels["validation"][:, class_id], valid_ref)
        source_rate = float(np.mean(valid_ref >= threshold))
        local_gain = source_local_gain(
            labels["validation"][:, class_id], valid_ref, valid_recv[:, class_id], source_rate
        )
        lcb = bootstrap_lcb(
            labels["validation"][:, class_id], valid_ref, valid_recv[:, class_id], source_rate, class_id
        )
        auc_gain = roc_auc_score(labels["validation"][:, class_id], valid_recv[:, class_id]) - roc_auc_score(
            labels["validation"][:, class_id], valid_ref
        )
        use_local = bool(local_gain > 0)
        use_lcb = bool(lcb > 0)
        use_auc = bool(auc_gain > 0)
        certificate.append(
            {"class": class_id, "rate": source_rate, "local_gain": local_gain, "lcb": lcb, "auc_gain": auc_gain,
             "use_local": use_local, "use_lcb": use_lcb, "use_auc": use_auc}
        )
        target_y_all = labels["test"][:, class_id]
        positives, negatives = np.flatnonzero(target_y_all), np.flatnonzero(~target_y_all)
        for mag in MAGS:
            n_positive = max(1, int(args.target_size * min(0.4, prior[class_id] * mag)))
            for seed in range(args.seeds):
                rng = np.random.RandomState(10_000 * class_id + seed)
                indices = np.concatenate(
                    [rng.choice(positives, n_positive, replace=len(positives) < n_positive),
                     rng.choice(negatives, args.target_size - n_positive, replace=len(negatives) < args.target_size - n_positive)]
                )
                rng.shuffle(indices)
                y = target_y_all[indices]
                ref_score, recv_score = test_ref[indices], test_recv[indices, class_id]
                k = int(np.sum(ref_score >= threshold))
                ref_f1, recv_f1 = f1_at_k(y, ref_score, k), f1_at_k(y, recv_score, k)
                results["reference"][mag].append(ref_f1)
                results["receiver"][mag].append(recv_f1)
                results["local_selector"][mag].append(recv_f1 if use_local else ref_f1)
                results["lcb_selector"][mag].append(recv_f1 if use_lcb else ref_f1)
                results["auc_selector"][mag].append(recv_f1 if use_auc else ref_f1)
                results["oracle_selector"][mag].append(max(ref_f1, recv_f1))
        print(f"class={position+1}/{len(classes)} elapsed={time.time()-START:.0f}s", flush=True)

    summary = {
        "selected_classes": {
            "local": int(sum(row["use_local"] for row in certificate)),
            "lcb": int(sum(row["use_lcb"] for row in certificate)),
            "auc": int(sum(row["use_auc"] for row in certificate)),
            "total": len(classes),
        },
        "magnitudes": {
            str(mag): {method: float(np.mean(results[method][mag])) for method in methods}
            for mag in MAGS
        },
        "certificate": certificate,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(output, {"results": results, "classes": classes, "certificate": certificate}, allow_pickle=True)
    output.with_suffix(".json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary["selected_classes"], indent=2), flush=True)
    print(json.dumps(summary["magnitudes"], indent=2), flush=True)
    print(f"saved={output} elapsed={time.time()-START:.0f}s", flush=True)


if __name__ == "__main__":
    main()
