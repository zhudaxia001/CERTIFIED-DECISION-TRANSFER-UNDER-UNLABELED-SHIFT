"""Fine-tune ResNet-50 and DINOv2 on COCO, then export paired features."""

from pathlib import Path
import argparse
import json
import random
import time

import numpy as np
import torch
from datasets import load_dataset
from PIL import Image
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet50_Weights, resnet50
from transformers import AutoImageProcessor, AutoModel


START = time.time()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_rows(parquet_dir):
    files = sorted(str(path) for path in Path(parquet_dir).glob("train-*.parquet"))
    if not files:
        raise RuntimeError(f"No train parquet shards found under {parquet_dir}")
    rows = load_dataset("parquet", data_files={"train": files}, split="train")
    categories = set()
    for row in rows.select_columns(["objects"]):
        categories.update(int(value) for value in row["objects"]["categories"])
    category_ids = np.asarray(sorted(categories), dtype=np.int64)
    category_index = {value: index for index, value in enumerate(category_ids)}
    labels = np.zeros((len(rows), len(category_ids)), dtype=np.int8)
    for row_index, row in enumerate(rows.select_columns(["objects"])):
        for value in row["objects"]["categories"]:
            labels[row_index, category_index[int(value)]] = 1
    return rows, labels, category_ids


def select_classes(train_labels, valid_labels, count):
    prevalence = train_labels.mean(axis=0)
    candidates = [
        class_id
        for class_id in np.argsort(prevalence)
        if train_labels[:, class_id].sum() >= 20
        and valid_labels[:, class_id].sum() >= 3
        and (~valid_labels[:, class_id].astype(bool)).sum() >= 3
    ]
    return [int(class_id) for class_id in candidates[:count]]


class CocoFineTuneDataset(Dataset):
    def __init__(self, rows, labels, indices, transform, train):
        self.rows = rows
        self.labels = labels
        self.indices = np.asarray(indices)
        self.transform = transform
        self.train = train

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        index = int(self.indices[item])
        image = self.rows[index]["image"].convert("RGB")
        if self.train and random.random() < 0.5:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        pixels = self.transform(image)
        return pixels, torch.from_numpy(self.labels[index].astype(np.float32)), index


class ProcessorTransform:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, image):
        return self.processor(images=image, return_tensors="pt")["pixel_values"][0]


class ResNetClassifier(nn.Module):
    def __init__(self, classes):
        super().__init__()
        self.backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        dimension = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.head = nn.Linear(dimension, classes)

    def encode(self, pixels):
        return self.backbone(pixels)

    def forward(self, pixels):
        return self.head(self.encode(pixels))


class DinoClassifier(nn.Module):
    def __init__(self, model_path, classes):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_path, local_files_only=True)
        dimension = self.backbone.config.hidden_size
        self.head = nn.Linear(dimension, classes)

    def encode(self, pixels):
        output = self.backbone(pixel_values=pixels)
        pooled = getattr(output, "pooler_output", None)
        return pooled if pooled is not None else output.last_hidden_state[:, 0]

    def forward(self, pixels):
        return self.head(self.encode(pixels))


def macro_auc(labels, scores, classes):
    values = []
    for class_id in classes:
        target = labels[:, class_id]
        if target.any() and (~target.astype(bool)).any():
            values.append(roc_auc_score(target, scores[:, class_id]))
    return float(np.mean(values))


def make_loader(rows, labels, indices, transform, train, args, batch_size):
    dataset = CocoFineTuneDataset(rows, labels, indices, transform, train)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
        drop_last=train,
    )


def predict_scores(model, loader, num_rows):
    scores = np.zeros((num_rows, model.head.out_features), dtype=np.float32)
    model.eval()
    with torch.inference_mode():
        for pixels, _, indices in loader:
            pixels = pixels.cuda(non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(pixels)
            scores[indices.numpy()] = logits.float().cpu().numpy()
    return scores


def train_model(name, model, rows, labels, train_indices, valid_indices, classes,
                transform, args, batch_size, learning_rate):
    model.cuda()
    train_loader = make_loader(
        rows, labels, train_indices, transform, True, args, batch_size
    )
    valid_loader = make_loader(
        rows, labels, valid_indices, transform, False, args, batch_size
    )
    positive = labels[train_indices].sum(axis=0).astype(np.float32)
    negative = len(train_indices) - positive
    pos_weight = np.minimum(
        negative / np.maximum(positive, 1.0), args.pos_weight_cap
    )
    pos_weight = torch.tensor(pos_weight, device="cuda", dtype=torch.float32)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=args.weight_decay
    )
    checkpoint = Path(args.output).with_suffix(f".{name}.pt")
    best_auc = -np.inf
    for epoch in range(args.epochs):
        model.train()
        loss_sum = 0.0
        for pixels, targets, _ in train_loader:
            pixels = pixels.cuda(non_blocking=True)
            targets = targets.cuda(non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(pixels)
                loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    logits, targets, pos_weight=pos_weight
                )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            loss_sum += float(loss.detach())
        valid_scores = predict_scores(model, valid_loader, len(labels))[valid_indices]
        valid_auc = macro_auc(labels[valid_indices], valid_scores, classes)
        print(
            f"model={name} epoch={epoch + 1}/{args.epochs} "
            f"loss={loss_sum / len(train_loader):.5f} valid_auc={valid_auc:.5f} "
            f"elapsed={time.time() - START:.0f}s",
            flush=True,
        )
        if valid_auc > best_auc:
            best_auc = valid_auc
            torch.save(model.state_dict(), checkpoint)
    model.load_state_dict(torch.load(checkpoint, map_location="cuda", weights_only=True))
    checkpoint.unlink()
    return model, best_auc


def extract_features(model, rows, labels, transform, args, batch_size):
    indices = np.arange(len(labels))
    loader = make_loader(rows, labels, indices, transform, False, args, batch_size)
    output = np.zeros((len(labels), model.head.in_features), dtype=np.float16)
    model.eval()
    with torch.inference_mode():
        for step, (pixels, _, batch_indices) in enumerate(loader):
            pixels = pixels.cuda(non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                features = nn.functional.normalize(model.encode(pixels).float(), dim=1)
            output[batch_indices.numpy()] = features.cpu().numpy().astype(np.float16)
            if step % 25 == 0:
                print(
                    f"extract_step={step}/{len(loader)} elapsed={time.time() - START:.0f}s",
                    flush=True,
                )
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet-dir", required=True)
    parser.add_argument("--receiver", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--reference-batch-size", type=int, default=192)
    parser.add_argument("--receiver-batch-size", type=int, default=96)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--classes", type=int, default=30)
    parser.add_argument("--reference-lr", type=float, default=1e-4)
    parser.add_argument("--receiver-lr", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--pos-weight-cap", type=float, default=20.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--split-seed", type=int, default=0)
    args = parser.parse_args()
    seed_everything(args.seed)

    rows, labels, category_ids = load_rows(args.parquet_dir)
    permutation = np.random.RandomState(args.split_seed).permutation(len(labels))
    train_end = int(0.70 * len(labels))
    valid_end = int(0.75 * len(labels))
    train_indices = permutation[:train_end]
    valid_indices = permutation[train_end:valid_end]
    classes = select_classes(
        labels[train_indices], labels[valid_indices], args.classes
    )
    if len(classes) < args.classes:
        raise RuntimeError(f"Only {len(classes)} classes satisfy source-only selection")
    print(
        f"images={len(labels)} train={len(train_indices)} valid={len(valid_indices)} "
        f"classes={classes}",
        flush=True,
    )

    reference_transform = ResNet50_Weights.IMAGENET1K_V2.transforms()
    receiver_processor = AutoImageProcessor.from_pretrained(
        args.receiver, local_files_only=True
    )
    receiver_transform = ProcessorTransform(receiver_processor)

    reference, reference_auc = train_model(
        "resnet50",
        ResNetClassifier(labels.shape[1]),
        rows,
        labels,
        train_indices,
        valid_indices,
        classes,
        reference_transform,
        args,
        args.reference_batch_size,
        args.reference_lr,
    )
    reference_features = extract_features(
        reference, rows, labels, reference_transform, args, args.reference_batch_size
    )
    del reference
    torch.cuda.empty_cache()

    receiver, receiver_auc = train_model(
        "dinov2",
        DinoClassifier(args.receiver, labels.shape[1]),
        rows,
        labels,
        train_indices,
        valid_indices,
        classes,
        receiver_transform,
        args,
        args.receiver_batch_size,
        args.receiver_lr,
    )
    receiver_features = extract_features(
        receiver, rows, labels, receiver_transform, args, args.receiver_batch_size
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output,
        reference=reference_features,
        receiver=receiver_features,
        labels=labels,
        image_ids=np.arange(len(labels), dtype=np.int64),
        category_ids=category_ids,
    )
    output.with_suffix(".json").write_text(
        json.dumps(
            {
                "num_images": len(labels),
                "train_images": len(train_indices),
                "valid_images": len(valid_indices),
                "classes": classes,
                "epochs": args.epochs,
                "seed": args.seed,
                "split_seed": args.split_seed,
                "reference_valid_auc": reference_auc,
                "receiver_valid_auc": receiver_auc,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"saved={output} elapsed={time.time() - START:.0f}s", flush=True)


if __name__ == "__main__":
    main()
