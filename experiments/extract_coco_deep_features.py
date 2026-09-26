"""Extract paired ResNet-50 and DINOv2 features from COCO 2017 images."""

from pathlib import Path
import argparse
import json
import random
import time

import numpy as np
import torch
from datasets import load_dataset
from PIL import Image, ImageEnhance, ImageFilter
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet50_Weights, resnet50
from transformers import AutoImageProcessor, AutoModel


START = time.time()


def apply_corruption(image, corruption, severity, seed):
    if corruption == "none":
        return image
    if corruption == "gaussian_blur":
        radius = (0.5, 1.0, 2.0, 3.0, 4.0)[severity - 1]
        return image.filter(ImageFilter.GaussianBlur(radius=radius))
    if corruption == "brightness":
        factor = (0.80, 0.65, 0.50, 0.35, 0.20)[severity - 1]
        return ImageEnhance.Brightness(image).enhance(factor)
    if corruption == "gaussian_noise":
        sigma = (8.0, 16.0, 24.0, 32.0, 40.0)[severity - 1]
        values = np.asarray(image, dtype=np.float32)
        noise = np.random.RandomState(seed).normal(0.0, sigma, values.shape)
        values = np.clip(values + noise, 0.0, 255.0).astype(np.uint8)
        return Image.fromarray(values, mode="RGB")
    raise ValueError(f"Unknown corruption: {corruption}")


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_records(annotation_path, image_root, max_images):
    with open(annotation_path, "r", encoding="utf-8") as handle:
        coco = json.load(handle)
    category_ids = sorted(category["id"] for category in coco["categories"])
    category_index = {category_id: index for index, category_id in enumerate(category_ids)}
    labels = {image["id"]: np.zeros(len(category_ids), dtype=np.int8) for image in coco["images"]}
    for annotation in coco["annotations"]:
        labels[annotation["image_id"]][category_index[annotation["category_id"]]] = 1
    records = []
    for image in sorted(coco["images"], key=lambda row: row["id"]):
        path = Path(image_root) / image["file_name"]
        if path.is_file():
            records.append((image["id"], path, labels[image["id"]]))
    if max_images:
        records = records[:max_images]
    if not records:
        raise RuntimeError("No COCO images matched the annotation file")
    return records, np.asarray(category_ids, dtype=np.int64)


class CocoFeatureDataset(Dataset):
    def __init__(self, records, reference_transform, receiver_processor, args):
        self.records = records
        self.reference_transform = reference_transform
        self.receiver_processor = receiver_processor
        self.corruption = args.corruption
        self.severity = args.severity
        self.seed = args.seed

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        image_id, path, labels = self.records[index]
        with Image.open(path) as handle:
            image = handle.convert("RGB")
            image = apply_corruption(
                image, self.corruption, self.severity, self.seed + index
            )
            reference_pixels = self.reference_transform(image)
            receiver_pixels = self.receiver_processor(
                images=image, return_tensors="pt"
            )["pixel_values"][0]
        return reference_pixels, receiver_pixels, labels, image_id


class CocoParquetFeatureDataset(Dataset):
    def __init__(self, rows, category_ids, reference_transform, receiver_processor, args):
        self.rows = rows
        self.category_index = {
            category_id: index for index, category_id in enumerate(category_ids)
        }
        self.reference_transform = reference_transform
        self.receiver_processor = receiver_processor
        self.corruption = args.corruption
        self.severity = args.severity
        self.seed = args.seed

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        image = row["image"].convert("RGB")
        image = apply_corruption(
            image, self.corruption, self.severity, self.seed + index
        )
        labels = np.zeros(len(self.category_index), dtype=np.int8)
        for category_id in row["objects"]["categories"]:
            labels[self.category_index[int(category_id)]] = 1
        reference_pixels = self.reference_transform(image)
        receiver_pixels = self.receiver_processor(
            images=image, return_tensors="pt"
        )["pixel_values"][0]
        return reference_pixels, receiver_pixels, labels, index


def load_parquet_dataset(parquet_dir, max_images):
    files = sorted(str(path) for path in Path(parquet_dir).glob("train-*.parquet"))
    if not files:
        raise RuntimeError(f"No train parquet shards found under {parquet_dir}")
    rows = load_dataset("parquet", data_files={"train": files}, split="train")
    if max_images:
        rows = rows.select(range(min(max_images, len(rows))))
    categories = set()
    for row in rows.select_columns(["objects"]):
        categories.update(int(value) for value in row["objects"]["categories"])
    return rows, np.asarray(sorted(categories), dtype=np.int64)


def receiver_pool(output):
    image_embeds = getattr(output, "image_embeds", None)
    if image_embeds is not None:
        return image_embeds
    pooled = getattr(output, "pooler_output", None)
    if pooled is not None:
        return pooled
    return output.last_hidden_state[:, 0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-root")
    parser.add_argument("--annotation")
    parser.add_argument("--parquet-dir")
    parser.add_argument("--output", required=True)
    parser.add_argument("--receiver", default="facebook/dinov2-base")
    parser.add_argument("--batch-size", type=int, default=192)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--corruption",
        choices=("none", "gaussian_blur", "gaussian_noise", "brightness"),
        default="none",
    )
    parser.add_argument("--severity", type=int, choices=range(1, 6), default=3)
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    seed_everything(args.seed)
    device = "cuda:0"

    reference_weights = ResNet50_Weights.IMAGENET1K_V2
    receiver_processor = AutoImageProcessor.from_pretrained(
        args.receiver, local_files_only=args.local_files_only
    )
    if args.parquet_dir:
        rows, category_ids = load_parquet_dataset(args.parquet_dir, args.max_images)
        dataset = CocoParquetFeatureDataset(
            rows, category_ids, reference_weights.transforms(), receiver_processor, args
        )
    else:
        if not args.image_root or not args.annotation:
            parser.error("Use --parquet-dir or provide both --image-root and --annotation")
        records, category_ids = load_records(
            args.annotation, args.image_root, args.max_images
        )
        dataset = CocoFeatureDataset(
            records, reference_weights.transforms(), receiver_processor, args
        )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
    )

    reference = resnet50(weights=reference_weights)
    reference.fc = nn.Identity()
    receiver = AutoModel.from_pretrained(
        args.receiver, local_files_only=args.local_files_only
    )
    if hasattr(receiver, "vision_model"):
        receiver = receiver.vision_model
    reference.eval().to(device)
    receiver.eval().to(device)

    reference_features = []
    receiver_features = []
    labels = []
    image_ids = []
    with torch.inference_mode():
        for step, (ref_pixels, recv_pixels, target, batch_ids) in enumerate(loader):
            ref_pixels = ref_pixels.to(device, non_blocking=True)
            recv_pixels = recv_pixels.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                ref_feature = reference(ref_pixels)
                recv_feature = receiver_pool(receiver(pixel_values=recv_pixels))
            reference_features.append(
                torch.nn.functional.normalize(ref_feature.float(), dim=1).cpu().numpy()
            )
            receiver_features.append(
                torch.nn.functional.normalize(recv_feature.float(), dim=1).cpu().numpy()
            )
            labels.append(target.numpy())
            image_ids.append(batch_ids.numpy())
            if step % 25 == 0:
                print(
                    f"step={step}/{len(loader)} images={min((step + 1) * args.batch_size, len(dataset))} "
                    f"elapsed={time.time() - START:.0f}s",
                    flush=True,
                )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output,
        reference=np.concatenate(reference_features).astype(np.float16),
        receiver=np.concatenate(receiver_features).astype(np.float16),
        labels=np.concatenate(labels).astype(np.int8),
        image_ids=np.concatenate(image_ids).astype(np.int64),
        category_ids=category_ids,
        corruption=np.asarray(args.corruption),
        severity=np.asarray(args.severity, dtype=np.int64),
    )
    print(
        f"saved={output} images={len(dataset)} elapsed={time.time() - START:.0f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
