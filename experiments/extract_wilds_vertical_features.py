"""Extract paired frozen features for WILDS vertical-domain replications."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet50_Weights, resnet50
from transformers import AutoImageProcessor, AutoModel
from wilds import get_dataset


class WILDSImageSubset(Dataset):
    def __init__(self, dataset, split, reference_transform, receiver_processor):
        subset = dataset.get_subset(split)
        self.dataset = dataset
        self.indices = np.asarray(subset.indices, dtype=np.int64)
        self.reference_transform = reference_transform
        self.receiver_processor = receiver_processor

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        original_index = int(self.indices[index])
        image, label, metadata = self.dataset[original_index]
        image = image.convert("RGB")
        reference_pixels = self.reference_transform(image)
        receiver_pixels = self.receiver_processor(
            images=image, return_tensors="pt"
        )["pixel_values"][0]
        return (
            reference_pixels,
            receiver_pixels,
            int(label),
            torch.as_tensor(metadata, dtype=torch.long),
            original_index,
        )


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
    parser.add_argument("--dataset", choices=("iwildcam", "rxrx1", "fmow"), required=True)
    parser.add_argument("--root-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--receiver", default="facebook/dinov2-base")
    parser.add_argument("--output-tag", default="dinov2")
    parser.add_argument("--splits", default="train,id_val,id_test,val,test")
    parser.add_argument("--batch-size", type=int, default=192)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--max-images-per-split", type=int, default=0)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda:0")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = get_dataset(
        dataset=args.dataset,
        root_dir=args.root_dir,
        download=args.download,
    )

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

    for split in [value.strip() for value in args.splits.split(",") if value.strip()]:
        output = output_dir / (
            f"{args.dataset}_{split}_resnet50_{args.output_tag}_features.npz"
        )
        if output.is_file() and not args.overwrite:
            print(f"skip={output}", flush=True)
            continue
        data = WILDSImageSubset(
            dataset,
            split,
            reference_weights.transforms(),
            receiver_processor,
        )
        if args.max_images_per_split:
            data.indices = data.indices[: args.max_images_per_split]
        loader = DataLoader(
            data,
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
        metadata = []
        indices = []
        with torch.inference_mode():
            for step, batch in enumerate(loader):
                ref_pixels, recv_pixels, target, meta, original_indices = batch
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
                metadata.append(meta.numpy())
                indices.append(original_indices.numpy())
                if step % 25 == 0:
                    processed = min((step + 1) * args.batch_size, len(data))
                    print(
                        f"dataset={args.dataset} split={split} step={step}/{len(loader)} "
                        f"images={processed}/{len(data)} elapsed={time.time() - started:.0f}s",
                        flush=True,
                    )
        np.savez(
            output,
            reference=np.concatenate(reference_features).astype(np.float16),
            receiver=np.concatenate(receiver_features).astype(np.float16),
            labels=np.concatenate(labels).astype(np.int16),
            metadata=np.concatenate(metadata).astype(np.int64),
            indices=np.concatenate(indices).astype(np.int64),
            split=np.asarray(split),
            dataset=np.asarray(args.dataset),
        )
        print(
            f"saved={output} images={len(data)} elapsed={time.time() - started:.0f}s",
            flush=True,
        )


if __name__ == "__main__":
    main()
