"""Dataset preparation, splitting, and DataLoader utilities."""

from __future__ import annotations

import copy
import re
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from tqdm.auto import tqdm

from src.utils import seed_worker

CATEGORY_PATTERN = re.compile(r"^\d{3}\.")
CLUTTER_PREFIX = "257."


def preprocess_caltech256(
    raw_dir: Path,
    output_dir: Path,
    image_size: int,
    force: bool = False,
) -> tuple[Path, Path]:
    """Convert raw Caltech-256 JPEG files into memory-mapped NumPy arrays."""
    images_path = output_dir / "images.npy"
    labels_path = output_dir / "labels.npy"

    if not force and images_path.exists() and labels_path.exists():
        print(f"Preprocessed arrays already exist: {output_dir}")
        return images_path, labels_path

    if not raw_dir.is_dir():
        raise FileNotFoundError(f"Raw dataset directory not found: {raw_dir}")

    # Only numeric category directories are used; 257.clutter is excluded
    category_dirs = sorted(
        path
        for path in raw_dir.iterdir()
        if path.is_dir()
        and CATEGORY_PATTERN.match(path.name)
        and not path.name.startswith(CLUTTER_PREFIX)
    )

    if len(category_dirs) != 256:
        raise ValueError(
            f"Expected 256 category directories, found {len(category_dirs)}"
        )

    image_records: list[tuple[Path, int]] = []

    for category_dir in category_dirs:
        label = int(category_dir.name[:3]) - 1
        image_paths = sorted(category_dir.glob("*.jpg"))
        image_records.extend((image_path, label) for image_path in image_paths)

    output_dir.mkdir(parents=True, exist_ok=True)

    preprocess = transforms.Compose(
        [
            transforms.Resize(
                256,
                interpolation=InterpolationMode.BILINEAR,
            ),
            transforms.CenterCrop(image_size),
        ]
    )

    # Create the output array directly on disk
    images = np.lib.format.open_memmap(
        str(images_path),
        mode="w+",
        dtype=np.uint8,
        shape=(len(image_records), image_size, image_size, 3),
    )
    labels = np.empty(len(image_records), dtype=np.int64)

    for index, (image_path, label) in enumerate(
        tqdm(image_records, desc="Preprocessing")
    ):
        with Image.open(image_path) as image:
            processed = preprocess(image.convert("RGB"))

        # Convert the PIL image to a uint8 HWC array
        images[index] = np.asarray(processed, dtype=np.uint8)
        labels[index] = label

    # Flush the memory map and save labels
    images.flush()
    np.save(labels_path, labels)
    print(f"Saved {len(image_records)} images to {output_dir}")

    return images_path, labels_path


def load_preprocessed_data(
    images_path: Path,
    labels_path: Path,
) -> tuple[np.ndarray, np.ndarray]:
    """Load preprocessed images as a read-only memory map and labels."""
    if not images_path.exists():
        raise FileNotFoundError(f"Images array not found: {images_path}")

    if not labels_path.exists():
        raise FileNotFoundError(f"Labels array not found: {labels_path}")

    images = np.load(images_path, mmap_mode="r")
    labels = np.load(labels_path)

    return images, labels


def load_or_create_splits(
    labels: np.ndarray,
    splits_path: Path,
    seed: int,
    train_fraction: float,
    val_fraction: float,
    test_fraction: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load a saved split or create a stratified train/val/test split."""
    if abs(train_fraction + val_fraction + test_fraction - 1.0) > 1e-9:
        raise ValueError("Split fractions must sum to 1.0")

    if splits_path.exists():
        with np.load(splits_path) as splits:
            train_indices = splits["train"]
            val_indices = splits["val"]
            test_indices = splits["test"]

        # Reuse a saved split only if it covers the dataset without
        # duplicates or out-of-range indices
        all_indices = np.concatenate(
            [train_indices, val_indices, test_indices]
        )
        is_valid = (
            len(all_indices) == len(labels)
            and len(np.unique(all_indices)) == len(labels)
            and all_indices.size > 0
            and all_indices.min() >= 0
            and all_indices.max() < len(labels)
        )

        if is_valid:
            return train_indices, val_indices, test_indices

        print("Saved split does not match the dataset; recreating it.")

    all_indices = np.arange(len(labels))
    train_indices, temp_indices = train_test_split(
        all_indices,
        test_size=1.0 - train_fraction,
        random_state=seed,
        stratify=labels,
    )
    val_indices, test_indices = train_test_split(
        temp_indices,
        test_size=test_fraction / (val_fraction + test_fraction),
        random_state=seed,
        stratify=labels[temp_indices],
    )

    splits_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        splits_path,
        train=train_indices,
        val=val_indices,
        test=test_indices,
    )
    print(f"Saved split to {splits_path}")

    return train_indices, val_indices, test_indices


class CaltechDataset(Dataset):
    """Dataset."""

    def __init__(
        self,
        images_path: Path,
        labels_path: Path,
        indices: Sequence[int],
        transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ) -> None:
        self.indices: np.ndarray = np.asarray(indices, dtype=np.int64)
        self.transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = (
            transform
        )
        self.labels: np.ndarray = np.load(labels_path)
        self.images: np.ndarray = np.load(images_path, mmap_mode="r")
        self.normalize: transforms.Normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        real_index = int(self.indices[index])
        image = torch.from_numpy(self.images[real_index].copy())
        # HWC -> CHW and uint8 -> float32.
        image = image.permute(2, 0, 1).float() / 255.0
        label = int(self.labels[real_index])

        # Augment before ImageNet normalization.
        if self.transform is not None:
            image = self.transform(image)

        image = self.normalize(image)

        return image, label


class PseudoLabelDataset(Dataset):
    """Selected images with model-generated labels.
    Only teacher predictions are used as targets.
    """

    def __init__(
        self,
        base_dataset: CaltechDataset,
        sample_indices: Sequence[int],
        pseudo_labels: Sequence[int],
    ) -> None:
        if len(sample_indices) != len(pseudo_labels):
            raise ValueError(
                "sample_indices and pseudo_labels must have equal length"
            )

        self.base_dataset: CaltechDataset = base_dataset
        self.sample_indices: np.ndarray = np.asarray(
            sample_indices,
            dtype=np.int64,
        )
        self.pseudo_labels: np.ndarray = np.asarray(
            pseudo_labels,
            dtype=np.int64,
        )

    def __len__(self) -> int:
        return len(self.sample_indices)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        base_index = int(self.sample_indices[index])
        # The true label returned by the base dataset is intentionally ignored
        image, _ = self.base_dataset[base_index]
        label = int(self.pseudo_labels[index])

        return image, label


def build_datasets(
    images_path: Path,
    labels_path: Path,
    train_indices: Sequence[int],
    val_indices: Sequence[int],
    test_indices: Sequence[int],
) -> tuple[CaltechDataset, CaltechDataset, CaltechDataset]:
    """Create train, validation, and test datasets with matching transforms."""
    train_transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(p=0.5),
        ]
    )

    train_dataset = CaltechDataset(
        images_path,
        labels_path,
        train_indices,
        transform=train_transform,
    )
    val_dataset = CaltechDataset(images_path, labels_path, val_indices)
    test_dataset = CaltechDataset(images_path, labels_path, test_indices)

    return train_dataset, val_dataset, test_dataset


def build_loaders(
    train_dataset: Dataset,
    val_dataset: Dataset,
    test_dataset: Dataset,
    batch_size: int,
    num_workers: int,
    seed: int,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Build DataLoaders with a deterministic training shuffle order."""
    generator = torch.Generator()
    generator.manual_seed(seed)

    loader_kwargs: dict[str, object] = {
        "num_workers": num_workers,
        "pin_memory": False,
    }

    if num_workers > 0:
        loader_kwargs["worker_init_fn"] = seed_worker

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        **loader_kwargs,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        **loader_kwargs,
    )

    return train_loader, val_loader, test_loader


def build_train_transform(
    level: int, image_size: int = 160
) -> transforms.Compose:
    """Train augmentation of increasing strength.

    level 0: horizontal flip only (identical to the baseline)
    level 1: + moderate random crop and color jitter
    level 2: + strong crop, color jitter, rotation, random erasing
    Works on float CHW tensors in [0, 1], before ImageNet normalization.
    """
    if level not in (0, 1, 2):
        raise ValueError(f"Unknown augmentation level: {level}")

    ops: list = [transforms.RandomHorizontalFlip(p=0.5)]

    if level == 1:
        ops += [
            transforms.RandomResizedCrop(
                image_size, scale=(0.6, 1.0), ratio=(0.8, 1.25), antialias=True
            ),
            transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2
            ),
        ]
    elif level == 2:
        ops += [
            transforms.RandomResizedCrop(
                image_size,
                scale=(0.35, 1.0),
                ratio=(0.75, 1.33),
                antialias=True,
            ),
            transforms.ColorJitter(
                brightness=0.4, contrast=0.4, saturation=0.4, hue=0.05
            ),
            transforms.RandomRotation(15),
            transforms.RandomErasing(p=0.25, scale=(0.02, 0.15)),
        ]

    return transforms.Compose(ops)


def clone_with_transform(
    dataset: CaltechDataset,
    transform: Callable[[torch.Tensor], torch.Tensor],
) -> CaltechDataset:
    """Shallow copy of a dataset with a different transform.

    The memmap and labels are shared, and the original dataset (used by the
    baseline and other experiments) is not mutated.
    """
    clone = copy.copy(dataset)
    clone.transform = transform
    return clone
