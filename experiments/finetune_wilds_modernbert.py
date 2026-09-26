"""Fine-tune ModernBERT for CivilComments without using target test labels."""

from pathlib import Path
import argparse
import json
import random
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset, Subset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)


SPLIT_CODES = {"train": 0, "val": 1, "test": 2}


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class CivilCommentsDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = list(texts)
        self.labels = np.asarray(labels, dtype=np.float32)

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
        encoded["labels"] = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)
        return encoded

    return collate


def predict(model, loader, device):
    model.eval()
    logits = []
    labels = []
    with torch.inference_mode():
        for batch in loader:
            target = batch.pop("labels")
            batch = {
                key: value.to(device, non_blocking=True)
                for key, value in batch.items()
            }
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                output = model(**batch).logits.squeeze(1)
            logits.append(output.float().cpu().numpy())
            labels.append(target.squeeze(1).numpy())
    return np.concatenate(logits), np.concatenate(labels)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--model", default="answerdotai/ModernBERT-base")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--pos-weight-cap", type=float, default=20.0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--selection-split-seed", type=int, default=0)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    args = parser.parse_args()

    start_time = time.time()
    seed_everything(args.seed)
    device = "cuda:0"
    frame = pd.read_csv(
        args.csv,
        usecols=["comment_text", "toxicity", "split"],
        low_memory=False,
    )
    frame = frame[frame["split"].isin(SPLIT_CODES)].reset_index(drop=True)
    texts = frame["comment_text"].fillna("").astype(str).to_numpy(dtype=object)
    labels = (frame["toxicity"].to_numpy(np.float32) >= 0.5).astype(np.float32)
    split = frame["split"].map(SPLIT_CODES).to_numpy(np.int8)
    train_indices = np.flatnonzero(split == 0)
    valid_indices = np.flatnonzero(split == 1)
    test_indices = np.flatnonzero(split == 2)

    train_set = CivilCommentsDataset(texts[train_indices], labels[train_indices])
    valid_set = CivilCommentsDataset(texts[valid_indices], labels[valid_indices])
    test_set = CivilCommentsDataset(texts[test_indices], labels[test_indices])
    selection_order = np.arange(len(valid_set))
    np.random.RandomState(args.selection_split_seed).shuffle(selection_order)
    selection_set = Subset(valid_set, selection_order[: len(selection_order) // 2])
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, use_fast=True, local_files_only=args.local_files_only
    )
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer must define either a pad token or an EOS token")
        tokenizer.pad_token = tokenizer.eos_token
    collate = make_collate(tokenizer, args.max_length)
    loader_options = {
        "num_workers": args.workers,
        "pin_memory": True,
        "persistent_workers": args.workers > 0,
        "collate_fn": collate,
    }
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
        **loader_options,
    )
    selection_loader = DataLoader(
        selection_set,
        batch_size=args.batch_size * 2,
        shuffle=False,
        **loader_options,
    )
    valid_loader = DataLoader(
        valid_set,
        batch_size=args.batch_size * 2,
        shuffle=False,
        **loader_options,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size * 2,
        shuffle=False,
        **loader_options,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        num_labels=1,
        problem_type="multi_label_classification",
        local_files_only=args.local_files_only,
    ).to(device)
    model.config.pad_token_id = tokenizer.pad_token_id
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
    positive_count = float(labels[train_indices].sum())
    pos_weight = min(
        args.pos_weight_cap,
        (len(train_indices) - positive_count) / max(1.0, positive_count),
    )
    pos_weight_tensor = torch.tensor([pos_weight], dtype=torch.float32, device=device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    total_steps = args.epochs * len(train_loader)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, int(0.06 * total_steps)),
        num_training_steps=total_steps,
    )
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_auc = -np.inf
    history = []
    print(
        f"train={len(train_set)} selection={len(selection_set)} "
        f"valid={len(valid_set)} test={len(test_set)} pos_weight={pos_weight:.3f}",
        flush=True,
    )

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        for step, batch in enumerate(train_loader):
            batch = {
                key: value.to(device, non_blocking=True)
                for key, value in batch.items()
            }
            target = batch.pop("labels")
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(**batch).logits
                loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    logits, target, pos_weight=pos_weight_tensor
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            running_loss += float(loss.detach())
            if (step + 1) % 200 == 0:
                print(
                    f"epoch={epoch + 1} step={step + 1}/{len(train_loader)} "
                    f"loss={running_loss / (step + 1):.5f} "
                    f"elapsed={time.time() - start_time:.0f}s",
                    flush=True,
                )
        selection_logits, selection_labels = predict(
            model, selection_loader, device
        )
        selection_auc = float(roc_auc_score(selection_labels, selection_logits))
        record = {
            "epoch": epoch + 1,
            "train_loss": running_loss / len(train_loader),
            "selection_auc": selection_auc,
        }
        history.append(record)
        print(json.dumps(record), flush=True)
        if selection_auc > best_auc:
            best_auc = selection_auc
            model.save_pretrained(checkpoint_dir, safe_serialization=True)
            tokenizer.save_pretrained(checkpoint_dir)

    del model
    torch.cuda.empty_cache()
    model = AutoModelForSequenceClassification.from_pretrained(
        checkpoint_dir, local_files_only=True
    ).to(device)
    valid_logits, valid_labels = predict(model, valid_loader, device)
    test_logits, test_labels = predict(model, test_loader, device)
    valid_auc = float(roc_auc_score(valid_labels, valid_logits))
    test_auc = float(roc_auc_score(test_labels, test_logits))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        valid_logits=valid_logits.astype(np.float16),
        test_logits=test_logits.astype(np.float16),
        valid_labels=valid_labels.astype(np.int8),
        test_labels=test_labels.astype(np.int8),
    )
    output.with_suffix(".json").write_text(
        json.dumps(
            {
                "model": args.model,
                "best_selection_auc": best_auc,
                "valid_auc": valid_auc,
                "test_auc": test_auc,
                "selection_split_seed": args.selection_split_seed,
                "history": history,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {"best_selection_auc": best_auc, "valid_auc": valid_auc, "test_auc": test_auc}
        ),
        flush=True,
    )
    print(f"saved={output} elapsed={time.time() - start_time:.0f}s", flush=True)


if __name__ == "__main__":
    main()
