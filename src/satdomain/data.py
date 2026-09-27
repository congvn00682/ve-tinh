"""Manifest-backed datasets and transforms."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Callable

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import functional as TF

from .constants import IMAGENET_MEAN, IMAGENET_STD


class RandomRotate90:
    """Rotate by a random multiple of 90 degrees without interpolation artifacts."""

    def __call__(self, image: Image.Image) -> Image.Image:
        return TF.rotate(image, 90 * random.randrange(4))


class ManifestDataset(Dataset):
    """Load RGB images described by a manifest DataFrame."""

    def __init__(
        self,
        frame: pd.DataFrame,
        root: str | Path,
        transform: Callable | None = None,
    ) -> None:
        self.frame = frame.reset_index(drop=True).copy()
        self.root = Path(root)
        self.transform = transform

        required = {"path", "class_index", "label"}
        missing = required.difference(self.frame.columns)
        if missing:
            raise ValueError(f"Manifest is missing columns: {sorted(missing)}")

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, str]:
        row = self.frame.iloc[index]
        image_path = self.root / str(row["path"])
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            if self.transform is not None:
                image = self.transform(image)
        return image, int(row["class_index"]), str(row["path"])


def build_transforms(image_size: int = 224) -> tuple[Callable, Callable]:
    """Return train and deterministic evaluation transforms."""

    train_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.RandomResizedCrop(image_size, scale=(0.80, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            RandomRotate90(),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    return train_transform, eval_transform


def read_manifest(path: str | Path, domain: str | None = None) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if domain is not None:
        frame = frame.loc[frame["domain"] == domain].copy()
    if frame.empty:
        raise ValueError(f"No samples found in {path} for domain={domain!r}")
    return frame


def source_train_validation_split(
    frame: pd.DataFrame,
    validation_fold: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Hold out one source fold for model selection, never for reporting."""

    validation = frame.loc[frame["fold"] == validation_fold].copy()
    train = frame.loc[frame["fold"] != validation_fold].copy()

    if train.empty or validation.empty:
        raise ValueError("A source train/validation partition is empty")
    return train, validation
