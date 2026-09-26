"""Extract frozen ResNet-50 and SigLIP2 features for independent replications."""

from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import CIFAR10, EuroSAT, VOCDetection
from torchvision.models import ResNet50_Weights, resnet50
from transformers import AutoImageProcessor, AutoModel


VOC_CLASSES = (
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def receiver_pool(output) -> torch.Tensor:
    image_embeds = getattr(output, "image_embeds", None)
    if image_embeds is not None:
        return image_embeds
    pooled = getattr(output, "pooler_output", None)
    if pooled is not None:
        return pooled
    return output.last_hidden_state[:, 0]


def voc_labels(target: dict) -> tuple[np.ndarray, str]:
    annotation = target["annotation"]
    objects = annotation.get("object", [])
    if isinstance(objects, dict):
        objects = [objects]
    present = {row["name"] for row in objects}
    labels = np.asarray([name in present for name in VOC_CLASSES], dtype=np.int8)
    return labels, Path(annotation["filename"]).stem


class IndependentFeatureDataset(Dataset):
    def __init__(self, root: Path, dataset_name: str, reference_transform, receiver_processor):
        self.dataset_name = dataset_name
        self.reference_transform = reference_transform
        self.receiver_processor = receiver_processor
        if dataset_name == "voc2007":
            self.parts = (
                (
                    "trainval",
                    VOCDetection(
                        root=str(root), year="2007", image_set="trainval", download=True
                    ),
                ),
                (
                    "test",
                    VOCDetection(
                        root=str(root), year="2007", image_set="test", download=True
                    ),
                ),
            )
            self.offsets = np.cumsum([0] + [len(dataset) for _, dataset in self.parts])
            self.class_names = VOC_CLASSES
        elif dataset_name == "voc2012":
            self.parts = (
                (
                    "trainval2012",
                    VOCDetection(
                        root=str(root), year="2012", image_set="trainval", download=True
                    ),
                ),
            )
            self.offsets = np.asarray([0, len(self.parts[0][1])])
            self.class_names = VOC_CLASSES
        elif dataset_name == "eurosat":
            dataset = EuroSAT(root=str(root), download=True)
            self.parts = (("all", dataset),)
            self.offsets = np.asarray([0, len(dataset)])
            self.class_names = tuple(dataset.classes)
        elif dataset_name == "cifar10_1":
            source = CIFAR10(root=str(root), train=True, download=True)
            target = CIFAR101Dataset(
                root / "cifar10.1_v6_data.npy",
                root / "cifar10.1_v6_labels.npy",
            )
            self.parts = (("source_train", source), ("target", target))
            self.offsets = np.cumsum([0, len(source), len(target)])
            self.class_names = tuple(source.classes)
        else:
            raise ValueError(dataset_name)

    def __len__(self) -> int:
        return int(self.offsets[-1])

    def __getitem__(self, index: int):
        part_index = int(np.searchsorted(self.offsets[1:], index, side="right"))
        split_name, dataset = self.parts[part_index]
        local_index = index - int(self.offsets[part_index])
        image, target = dataset[local_index]
        image = image.convert("RGB")
        if self.dataset_name in ("voc2007", "voc2012"):
            labels, image_id = voc_labels(target)
        elif self.dataset_name == "cifar10_1":
            labels = np.zeros(len(self.class_names), dtype=np.int8)
            labels[int(target)] = 1
            prefix = "cifar10-source" if split_name == "source_train" else "cifar10.1-target"
            image_id = f"{prefix}-{local_index:06d}"
        else:
            labels = np.zeros(len(self.class_names), dtype=np.int8)
            labels[int(target)] = 1
            image_id = f"eurosat-{local_index:06d}"
        reference_pixels = self.reference_transform(image)
        receiver_pixels = self.receiver_processor(images=image, return_tensors="pt")[
            "pixel_values"
        ][0]
        return reference_pixels, receiver_pixels, labels, image_id, split_name


class CIFAR101Dataset(Dataset):
    def __init__(self, data_path: Path, labels_path: Path):
        self.data = np.load(data_path, allow_pickle=False)
        self.labels = np.load(labels_path, allow_pickle=False)
        if len(self.data) != len(self.labels):
            raise ValueError("CIFAR-10.1 data and labels have different lengths")

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int):
        return Image.fromarray(self.data[index]), int(self.labels[index])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        choices=("voc2007", "voc2012", "eurosat", "cifar10_1"),
        required=True,
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receiver", default="google/siglip2-base-patch16-224")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20260820)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    started = time.time()
    seed_everything(args.seed)

    reference_weights = ResNet50_Weights.IMAGENET1K_V2
    receiver_processor = AutoImageProcessor.from_pretrained(
        args.receiver, local_files_only=args.local_files_only
    )
    dataset = IndependentFeatureDataset(
        args.data_root,
        args.dataset,
        reference_weights.transforms(),
        receiver_processor,
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
    reference.eval().to(args.device)
    receiver.eval().to(args.device)

    reference_features = []
    receiver_features = []
    labels = []
    image_ids = []
    splits = []
    with torch.inference_mode():
        for step, (ref_pixels, recv_pixels, target, batch_ids, batch_splits) in enumerate(loader):
            ref_pixels = ref_pixels.to(args.device, non_blocking=True)
            recv_pixels = recv_pixels.to(args.device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                reference_feature = reference(ref_pixels)
                receiver_feature = receiver_pool(receiver(pixel_values=recv_pixels))
            reference_features.append(
                torch.nn.functional.normalize(reference_feature.float(), dim=1)
                .cpu()
                .numpy()
            )
            receiver_features.append(
                torch.nn.functional.normalize(receiver_feature.float(), dim=1)
                .cpu()
                .numpy()
            )
            labels.append(target.numpy())
            image_ids.extend(batch_ids)
            splits.extend(batch_splits)
            if step % 25 == 0:
                print(
                    f"step={step}/{len(loader)} images={min((step + 1) * args.batch_size, len(dataset))} "
                    f"elapsed={time.time() - started:.0f}s",
                    flush=True,
                )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output,
        reference=np.concatenate(reference_features).astype(np.float16),
        receiver=np.concatenate(receiver_features).astype(np.float16),
        labels=np.concatenate(labels).astype(np.int8),
        image_ids=np.asarray(image_ids),
        splits=np.asarray(splits),
        class_names=np.asarray(dataset.class_names),
        dataset=np.asarray(args.dataset),
        receiver_name=np.asarray(args.receiver),
    )
    print(
        f"saved={args.output} images={len(dataset)} elapsed={time.time() - started:.0f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
