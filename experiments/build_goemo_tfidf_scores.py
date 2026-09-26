"""Build a fixed TF-IDF reference score archive for GoEmotions."""

from pathlib import Path
import argparse
import json

import numpy as np
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


NUM_LABELS = 28


def multihot(rows):
    labels = np.zeros((len(rows), NUM_LABELS), dtype=np.int8)
    for index, row in enumerate(rows):
        labels[index, row["labels"]] = 1
    return labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--checkpoint-fraction", type=float, default=0.20)
    parser.add_argument("--checkpoint-split-seed", type=int, default=20260820)
    args = parser.parse_args()

    raw = load_dataset("google-research-datasets/go_emotions", "simplified")
    texts = {
        split: np.asarray(raw[split]["text"], dtype=object)
        for split in ("train", "validation", "test")
    }
    labels = {
        split: multihot(raw[split])
        for split in ("train", "validation", "test")
    }
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_features=100_000,
        sublinear_tf=True,
    )
    train_features = vectorizer.fit_transform(texts["train"])
    valid_features = vectorizer.transform(texts["validation"])
    test_features = vectorizer.transform(texts["test"])
    valid_logits = np.zeros((len(texts["validation"]), NUM_LABELS), dtype=np.float32)
    test_logits = np.zeros((len(texts["test"]), NUM_LABELS), dtype=np.float32)
    for class_id in range(NUM_LABELS):
        model = LogisticRegression(
            C=1.0,
            max_iter=1000,
            solver="liblinear",
            random_state=0,
        ).fit(train_features, labels["train"][:, class_id])
        valid_logits[:, class_id] = model.decision_function(valid_features)
        test_logits[:, class_id] = model.decision_function(test_features)
        print(f"class={class_id + 1}/{NUM_LABELS}", flush=True)

    permutation = np.random.RandomState(args.checkpoint_split_seed).permutation(
        len(texts["validation"])
    )
    checkpoint_end = max(1, int(round(args.checkpoint_fraction * len(permutation))))
    checkpoint_indices = np.sort(permutation[:checkpoint_end])
    certificate_indices = np.sort(permutation[checkpoint_end:])
    selector_auc = float(
        np.mean(
            [
                roc_auc_score(
                    labels["validation"][checkpoint_indices, class_id],
                    valid_logits[checkpoint_indices, class_id],
                )
                for class_id in range(NUM_LABELS)
            ]
        )
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        valid_logits=valid_logits,
        test_logits=test_logits,
        valid_labels=labels["validation"],
        test_labels=labels["test"],
        checkpoint_indices=checkpoint_indices.astype(np.int32),
        certificate_indices=certificate_indices.astype(np.int32),
    )
    output.with_suffix(".json").write_text(
        json.dumps(
            {
                "selector_macro_auc": selector_auc,
                "checkpoint_examples": int(len(checkpoint_indices)),
                "certificate_examples": int(len(certificate_indices)),
                "checkpoint_split_seed": args.checkpoint_split_seed,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"selector_macro_auc={selector_auc:.6f} saved={output}", flush=True)


if __name__ == "__main__":
    main()
