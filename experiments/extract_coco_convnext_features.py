"""Extract ConvNeXt-Tiny features aligned with an existing COCO feature archive."""

from pathlib import Path
import argparse
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ConvNeXt_Tiny_Weights, convnext_tiny

from extract_coco_deep_features import load_parquet_dataset


START = time.time()


class CocoConvNeXtDataset(Dataset):
    def __init__(self, rows, transform):
        self.rows = rows
        self.transform = transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        image = self.rows[index]["image"].convert("RGB")
        return self.transform(image), index


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet-dir", required=True)
    parser.add_argument("--base-features", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--max-images", type=int, default=0)
    args = parser.parse_args()

    base = np.load(args.base_features, mmap_mode="r")
    expected_images = len(base["labels"])
    max_images = args.max_images or expected_images
    if max_images != expected_images:
        raise ValueError(
            f"max-images={max_images} must match base archive size={expected_images}"
        )

    weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1
    rows, category_ids = load_parquet_dataset(args.parquet_dir, max_images)
    if len(rows) != expected_images:
        raise RuntimeError(
            f"Loaded {len(rows)} images but base archive has {expected_images}"
        )
    if not np.array_equal(category_ids, base["category_ids"]):
        raise RuntimeError("Category IDs do not align with the base archive")

    dataset = CocoConvNeXtDataset(rows, weights.transforms())
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
    )
    model = convnext_tiny(weights=weights)
    model.classifier[2] = nn.Identity()
    model.eval().to("cuda:0")

    receiver_features = []
    observed_indices = []
    with torch.inference_mode():
        for step, (pixels, indices) in enumerate(loader):
            pixels = pixels.to("cuda:0", non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                features = model(pixels)
            receiver_features.append(
                torch.nn.functional.normalize(features.float(), dim=1).cpu().numpy()
            )
            observed_indices.append(indices.numpy())
            if step % 25 == 0:
                print(
                    f"step={step}/{len(loader)} "
                    f"images={min((step + 1) * args.batch_size, len(dataset))} "
                    f"elapsed={time.time() - START:.0f}s",
                    flush=True,
                )

    observed_indices = np.concatenate(observed_indices)
    if not np.array_equal(observed_indices, np.arange(expected_images)):
        raise RuntimeError("DataLoader order changed during extraction")
    receiver_features = np.concatenate(receiver_features).astype(np.float16)
    if receiver_features.shape != (expected_images, 768):
        raise RuntimeError(f"Unexpected ConvNeXt feature shape {receiver_features.shape}")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output,
        reference=np.asarray(base["reference"]),
        receiver=receiver_features,
        labels=np.asarray(base["labels"]),
        image_ids=np.asarray(base["image_ids"]),
        category_ids=np.asarray(base["category_ids"]),
        corruption=np.asarray("none"),
        severity=np.asarray(0, dtype=np.int64),
        receiver_name=np.asarray("torchvision/convnext_tiny_imagenet1k_v1"),
    )
    print(
        f"saved={output} shape={receiver_features.shape} "
        f"elapsed={time.time() - START:.0f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
