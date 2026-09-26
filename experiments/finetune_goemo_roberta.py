"""Fine-tune a modern score-incompatible receiver on GoEmotions.

The official training split is used for optimization. A fixed subset of the
validation split can be reserved for checkpoint selection so that the
remaining examples stay untouched for downstream certification. No test
labels participate in fitting or selection.
"""

from pathlib import Path
import argparse
import json
import random
import time

import numpy as np
import torch
from datasets import load_dataset
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)


START = time.time()
NUM_LABELS = 28


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def multihot(rows):
    labels = np.zeros((len(rows), NUM_LABELS), dtype=np.float32)
    for index, row in enumerate(rows):
        labels[index, row["labels"]] = 1.0
    return labels


class TextDataset(Dataset):
    def __init__(self, rows):
        self.texts = [row["text"] for row in rows]
        self.labels = multihot(rows)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        return self.texts[index], self.labels[index]


def make_collate(tokenizer, max_length):
    def collate(batch):
        texts, labels = zip(*batch)
        encoded = tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encoded["labels"] = torch.tensor(np.asarray(labels), dtype=torch.float32)
        return encoded

    return collate


def predict(model, loader, device):
    model.eval()
    logits, labels = [], []
    with torch.inference_mode():
        for batch in loader:
            target = batch.pop("labels")
            batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                output = model(**batch).logits
            logits.append(output.float().cpu().numpy())
            labels.append(target.numpy())
    return np.concatenate(logits), np.concatenate(labels)


def macro_auc(labels, logits):
    values = [roc_auc_score(labels[:, c], logits[:, c]) for c in range(NUM_LABELS)]
    return float(np.mean(values)), values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="FacebookAI/roberta-base")
    parser.add_argument("--output", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--pos-weight-cap", type=float, default=20.0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint-fraction", type=float, default=1.0)
    parser.add_argument("--checkpoint-split-seed", type=int, default=0)
    args = parser.parse_args()
    if not 0.0 < args.checkpoint_fraction <= 1.0:
        raise ValueError("checkpoint-fraction must be in (0, 1]")
    seed_everything(args.seed)
    device = "cuda:0"

    raw = load_dataset("google-research-datasets/go_emotions", "simplified")
    train_set = TextDataset(raw["train"])
    valid_set = TextDataset(raw["validation"])
    test_set = TextDataset(raw["test"])
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer must define either a pad token or an EOS token")
        tokenizer.pad_token = tokenizer.eos_token
    collate = make_collate(tokenizer, args.max_length)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
        collate_fn=collate,
    )
    valid_loader = DataLoader(
        valid_set,
        batch_size=args.batch_size * 2,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
        collate_fn=collate,
    )
    validation_permutation = np.random.RandomState(
        args.checkpoint_split_seed
    ).permutation(len(valid_set))
    checkpoint_size = max(
        1, int(round(args.checkpoint_fraction * len(validation_permutation)))
    )
    checkpoint_indices = np.sort(validation_permutation[:checkpoint_size])
    certificate_indices = np.sort(validation_permutation[checkpoint_size:])
    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size * 2,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
        collate_fn=collate,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        num_labels=NUM_LABELS,
        problem_type="multi_label_classification",
    ).to(device)
    model.config.pad_token_id = tokenizer.pad_token_id
    positive_count = train_set.labels.sum(axis=0)
    negative_count = len(train_set) - positive_count
    pos_weight = np.minimum(
        negative_count / np.maximum(positive_count, 1.0), args.pos_weight_cap
    )
    pos_weight = torch.tensor(pos_weight, dtype=torch.float32, device=device)
    print(
        f"pos_weight min={pos_weight.min().item():.2f} "
        f"max={pos_weight.max().item():.2f} cap={args.pos_weight_cap:.2f}",
        flush=True,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    total_steps = args.epochs * len(train_loader)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, int(0.06 * total_steps)),
        num_training_steps=total_steps,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_auc = -np.inf
    history = []

    for epoch in range(args.epochs):
        model.train()
        running = 0.0
        for step, batch in enumerate(train_loader):
            batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
            target = batch.pop("labels")
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(**batch).logits
                loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    logits, target, pos_weight=pos_weight
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            running += float(loss.detach())
            if (step + 1) % 100 == 0:
                print(
                    f"epoch={epoch+1} step={step+1}/{len(train_loader)} "
                    f"loss={running/(step+1):.5f} elapsed={time.time()-START:.0f}s",
                    flush=True,
                )
        valid_logits, valid_labels = predict(model, valid_loader, device)
        valid_auc, class_auc = macro_auc(
            valid_labels[checkpoint_indices], valid_logits[checkpoint_indices]
        )
        record = {
            "epoch": epoch + 1,
            "train_loss": running / len(train_loader),
            "valid_macro_auc": valid_auc,
            "valid_class_auc": class_auc,
        }
        history.append(record)
        print(json.dumps(record), flush=True)
        if valid_auc > best_auc:
            best_auc = valid_auc
            model.save_pretrained(checkpoint_dir, safe_serialization=True)
            tokenizer.save_pretrained(checkpoint_dir)

    del model
    torch.cuda.empty_cache()
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint_dir).to(device)
    valid_logits, valid_labels = predict(model, valid_loader, device)
    test_logits, test_labels = predict(model, test_loader, device)
    test_auc, test_class_auc = macro_auc(test_labels, test_logits)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        valid_logits=valid_logits.astype(np.float16),
        test_logits=test_logits.astype(np.float16),
        valid_labels=valid_labels.astype(np.int8),
        test_labels=test_labels.astype(np.int8),
        checkpoint_indices=checkpoint_indices.astype(np.int32),
        certificate_indices=certificate_indices.astype(np.int32),
    )
    summary = {
        "model": args.model,
        "pos_weight_cap": args.pos_weight_cap,
        "checkpoint_fraction": args.checkpoint_fraction,
        "checkpoint_split_seed": args.checkpoint_split_seed,
        "checkpoint_examples": int(len(checkpoint_indices)),
        "certificate_examples": int(len(certificate_indices)),
        "best_valid_macro_auc": best_auc,
        "test_macro_auc": test_auc,
        "test_class_auc": test_class_auc,
        "history": history,
    }
    output.with_suffix(".json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"best_valid_macro_auc": best_auc, "test_macro_auc": test_auc}), flush=True)
    print(f"saved={output} elapsed={time.time()-START:.0f}s", flush=True)


if __name__ == "__main__":
    main()
