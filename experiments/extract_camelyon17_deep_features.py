"""Extract paired ResNet-50 and DINOv2 features for Camelyon17 splits."""

from pathlib import Path
import argparse
import time

import numpy as np
import torch
from datasets import load_dataset
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet50_Weights, resnet50
from transformers import AutoImageProcessor, AutoModel


class CamelyonDataset(Dataset):
    def __init__(self, rows, reference_transform, receiver_processor):
        self.rows = rows
        self.reference_transform = reference_transform
        self.receiver_processor = receiver_processor

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        image = row["image"].convert("RGB")
        return (
            self.reference_transform(image),
            self.receiver_processor(images=image, return_tensors="pt")[
                "pixel_values"
            ][0],
            int(row["label"]),
            index,
        )


def receiver_pool(output):
    image_embeds = getattr(output, "image_embeds", None)
    if image_embeds is not None:
        return image_embeds
    pooled = getattr(output, "pooler_output", None)
    if pooled is not None:
        return pooled
    return output.last_hidden_state[:, 0]


def split_files(data_dir, split):
    files = sorted(Path(data_dir).glob(f"{split}-*.parquet"))
    if not files:
        raise RuntimeError(f"No parquet files found for {split}")
    return [str(path) for path in files]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--receiver", default="facebook/dinov2-base")
    parser.add_argument("--output-tag", default="dinov2")
    parser.add_argument(
        "--splits", default="id_train,id_val,ood_val,ood_test"
    )
    parser.add_argument("--batch-size", type=int, default=192)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--max-images-per-split", type=int, default=0)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    device = "cuda:0"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    reference_weights = ResNet50_Weights.IMAGENET1K_V2
    receiver_processor = AutoImageProcessor.from_pretrained(
        args.receiver, local_files_only=args.local_files_only
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

    for split in args.splits.split(","):
        split = split.strip()
        output = (
            output_dir
            / f"camelyon17_{split}_resnet50_{args.output_tag}_features.npz"
        )
        if output.is_file() and not args.overwrite:
            print(f"skip={output}", flush=True)
            continue
        rows = load_dataset(
            "parquet",
            data_files={split: split_files(args.data_dir, split)},
            split=split,
        )
        if args.max_images_per_split:
            rows = rows.select(range(min(args.max_images_per_split, len(rows))))
        dataset = CamelyonDataset(
            rows, reference_weights.transforms(), receiver_processor
        )
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=True,
            persistent_workers=args.workers > 0,
        )
        started = time.time()
        reference_features = []
        receiver_features = []
        labels = []
        indices = []
        with torch.inference_mode():
            for step, (ref_pixels, recv_pixels, target, batch_indices) in enumerate(
                loader
            ):
                ref_pixels = ref_pixels.to(device, non_blocking=True)
                recv_pixels = recv_pixels.to(device, non_blocking=True)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    ref_feature = reference(ref_pixels)
                    recv_feature = receiver_pool(receiver(pixel_values=recv_pixels))
                reference_features.append(
                    torch.nn.functional.normalize(ref_feature.float(), dim=1)
                    .cpu()
                    .numpy()
                )
                receiver_features.append(
                    torch.nn.functional.normalize(recv_feature.float(), dim=1)
                    .cpu()
                    .numpy()
                )
                labels.append(target.numpy())
                indices.append(batch_indices.numpy())
                if step % 25 == 0:
                    print(
                        f"split={split} step={step}/{len(loader)} "
                        f"images={min((step + 1) * args.batch_size, len(dataset))} "
                        f"elapsed={time.time() - started:.0f}s",
                        flush=True,
                    )
        np.savez(
            output,
            reference=np.concatenate(reference_features).astype(np.float16),
            receiver=np.concatenate(receiver_features).astype(np.float16),
            labels=np.concatenate(labels).astype(np.int8),
            indices=np.concatenate(indices).astype(np.int64),
            split=np.asarray(split),
        )
        print(
            f"saved={output} images={len(dataset)} elapsed={time.time() - started:.0f}s",
            flush=True,
        )


if __name__ == "__main__":
    main()
