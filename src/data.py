"""Dataset loading, splitting, and preprocessing.

Expected on-disk layout (one folder per species, ``ImageFolder`` style)::

    data/night_wildlife/
        bobcat/    img001.jpg ...
        coyote/    ...
        raccoon/   ...
        deer/      ...

The night-vision camera-trap frames are infrared (single channel). This loader
treats every image the same way: it is read, optionally forced through a
grayscale->RGB path (so infrared frames match the 3-channel backbone), resized,
and normalised with ImageNet statistics so the pretrained backbone sees inputs in
the distribution it expects.

The split is stratified and deterministic given the seed, so the same image
never leaks between train / val / test across runs.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

# ImageNet normalisation constants (the backbone was pretrained with these).
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_transforms(image_size: int, grayscale_to_rgb: bool, train: bool):
    """Return a torchvision transform pipeline.

    Training augmentations (flips, small rotations, colour jitter, random
    resized crop) improve robustness to the pose/brightness variation typical of
    camera-trap frames. Validation/test use a deterministic resize + centre crop.
    """
    from torchvision import transforms

    steps = []
    # Infrared frames are effectively single-channel. Converting to grayscale and
    # back to 3 channels gives the pretrained (3-channel) backbone a consistent
    # input regardless of whether a given frame was captured in colour or IR.
    if grayscale_to_rgb:
        steps.append(transforms.Grayscale(num_output_channels=3))

    if train:
        steps += [
            transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
        ]
    else:
        steps += [
            transforms.Resize(int(image_size * 1.15)),
            transforms.CenterCrop(image_size),
        ]

    steps += [
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]
    return transforms.Compose(steps)


@dataclass
class Datasets:
    train: object
    val: object
    test: object
    class_names: List[str]


def _stratified_indices(targets: List[int], n_classes: int,
                        val_fraction: float, test_fraction: float,
                        seed: int) -> Tuple[List[int], List[int], List[int]]:
    """Split indices per-class so every class is represented in each split."""
    rng = np.random.default_rng(seed)
    train_idx, val_idx, test_idx = [], [], []
    targets_arr = np.asarray(targets)
    for c in range(n_classes):
        idx = np.where(targets_arr == c)[0]
        rng.shuffle(idx)
        n = len(idx)
        n_test = int(round(n * test_fraction))
        n_val = int(round(n * val_fraction))
        test_idx.extend(idx[:n_test].tolist())
        val_idx.extend(idx[n_test:n_test + n_val].tolist())
        train_idx.extend(idx[n_test + n_val:].tolist())
    return train_idx, val_idx, test_idx


def load_datasets(cfg) -> Datasets:
    """Load an ImageFolder dataset and produce stratified train/val/test subsets.

    Val and test wrap the *same* underlying images but with evaluation-time
    (non-augmenting) transforms, while train uses the augmenting pipeline.
    """
    from torch.utils.data import Subset
    from torchvision.datasets import ImageFolder

    base = ImageFolder(cfg.data_dir)  # no transform yet; we attach per-split ones
    class_names = base.classes
    targets = [label for _, label in base.samples]

    train_idx, val_idx, test_idx = _stratified_indices(
        targets, len(class_names), cfg.val_fraction, cfg.test_fraction, cfg.seed)

    train_tf = build_transforms(cfg.image_size, cfg.grayscale_to_rgb, train=True)
    eval_tf = build_transforms(cfg.image_size, cfg.grayscale_to_rgb, train=False)

    train_ds = ImageFolder(cfg.data_dir, transform=train_tf)
    eval_ds = ImageFolder(cfg.data_dir, transform=eval_tf)

    return Datasets(
        train=Subset(train_ds, train_idx),
        val=Subset(eval_ds, val_idx),
        test=Subset(eval_ds, test_idx),
        class_names=class_names,
    )


def make_loaders(cfg, datasets: Datasets):
    """Wrap the subsets in DataLoaders."""
    from torch.utils.data import DataLoader

    common = dict(batch_size=cfg.batch_size, num_workers=cfg.num_workers,
                  pin_memory=(cfg.resolved_device() == "cuda"))
    return (
        DataLoader(datasets.train, shuffle=True, **common),
        DataLoader(datasets.val, shuffle=False, **common),
        DataLoader(datasets.test, shuffle=False, **common),
    )


def class_weights(datasets: Datasets, n_classes: int):
    """Inverse-frequency class weights from the training split (for imbalance)."""
    import torch

    counts = np.zeros(n_classes, dtype=np.float64)
    for _, label in datasets.train:
        counts[label] += 1
    counts = np.maximum(counts, 1.0)
    weights = counts.sum() / (n_classes * counts)
    return torch.tensor(weights, dtype=torch.float32)
