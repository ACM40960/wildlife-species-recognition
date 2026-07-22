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

import os
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
    test: object                 # unseen-location test (the honest held-out set)
    class_names: List[str]
    seen_test: object = None     # held-out images from SEEN (training) locations


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


def read_manifest(data_dir, manifest_name):
    """Read manifest rows, or None if there is no manifest."""
    import csv

    path = os.path.join(data_dir, manifest_name)
    if not os.path.exists(path):
        return None
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def _parse_bbox(raw):
    if not raw:
        return None
    try:
        x, y, w, h = (int(float(v)) for v in raw.split(";"))
        return (x, y, w, h)
    except Exception:
        return None


def _carve_seen_test(train_rows, fraction, seed):
    """Deterministically hold out a fraction of train rows as a seen-location test.

    Assignment is by a stable hash of ``image_id`` (not Python's salted ``hash``),
    so the same images are held out on every run and across processes. Returns
    (kept_train_rows, seen_test_rows). Stratified per class so each species is
    represented in the seen-location test.
    """
    import hashlib
    from collections import defaultdict

    by_class = defaultdict(list)
    for r in train_rows:
        by_class[r["class"]].append(r)

    kept, seen = [], []
    for cls, rows_c in by_class.items():
        def key(r):
            h = hashlib.md5(f"{seed}:{r['image_id']}".encode()).hexdigest()
            return int(h[:8], 16)
        rows_sorted = sorted(rows_c, key=key)
        n_seen = int(round(len(rows_sorted) * fraction))
        seen.extend(rows_sorted[:n_seen])
        kept.extend(rows_sorted[n_seen:])
    return kept, seen


class ManifestDataset:
    """Dataset driven by the manifest, cropping to the animal box at *load* time.

    Stored frames are uncropped (see ``scripts/build_night_wildlife.py``); the
    bounding box lives in the manifest and the crop is applied here, so nothing is
    baked into the files and the crop strategy is a runtime choice.
    """

    def __init__(self, rows, data_dir, class_to_idx, transform,
                 crop_to_bbox=True, bbox_pad=0.15):
        from torch.utils.data import Dataset  # noqa: F401  (documents the interface)
        self.rows = rows
        self.data_dir = data_dir
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.crop_to_bbox = crop_to_bbox
        self.bbox_pad = bbox_pad

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        from PIL import Image

        row = self.rows[i]
        img = Image.open(os.path.join(self.data_dir, row["filename"])).convert("RGB")
        box = _parse_bbox(row.get("bbox", "")) if self.crop_to_bbox else None
        if box is not None:
            x, y, w, h = box
            px, py = w * self.bbox_pad, h * self.bbox_pad
            W, H = img.size
            crop = (max(0, int(x - px)), max(0, int(y - py)),
                    min(W, int(x + w + px)), min(H, int(y + h + py)))
            if crop[2] - crop[0] > 5 and crop[3] - crop[1] > 5:
                img = img.crop(crop)
        return self.transform(img), self.class_to_idx[row["class"]]


def load_datasets(cfg) -> Datasets:
    """Produce train/val/test datasets.

    If ``cfg.split_by == "location"`` and a manifest is present, images are loaded
    via :class:`ManifestDataset` with the location-grouped split from the manifest
    (whole camera locations held out — no shared backgrounds) and cropped to the
    animal box at load time. Otherwise a plain stratified random ImageFolder split
    is used. Val/test use evaluation transforms; train uses augmentation.
    """
    rows = None
    if getattr(cfg, "split_by", "location") == "location":
        rows = read_manifest(cfg.data_dir, getattr(cfg, "manifest_name", "manifest.csv"))

    train_tf = build_transforms(cfg.image_size, cfg.grayscale_to_rgb, train=True)
    eval_tf = build_transforms(cfg.image_size, cfg.grayscale_to_rgb, train=False)

    if rows is not None:
        print("[data] location-grouped split from manifest; crop_to_bbox="
              f"{getattr(cfg, 'crop_to_bbox', True)}")
        class_names = sorted({r["class"] for r in rows})
        class_to_idx = {c: i for i, c in enumerate(class_names)}
        by_split = {"train": [], "val": [], "test": []}
        for r in rows:
            if r.get("split") in by_split:
                by_split[r["split"]].append(r)

        # Carve a SEEN-location test set: hold out a fraction of the train-location
        # images (deterministically, so they are never trained on) to measure
        # accuracy on locations the model HAS seen, alongside the unseen test set.
        seen_frac = getattr(cfg, "seen_test_fraction", 0.0)
        seen_rows = []
        if seen_frac > 0:
            keep, seen_rows = _carve_seen_test(by_split["train"], seen_frac, cfg.seed)
            by_split["train"] = keep
        crop = getattr(cfg, "crop_to_bbox", True)

        def ds(split_rows, tf):
            return ManifestDataset(split_rows, cfg.data_dir, class_to_idx, tf, crop)

        return Datasets(
            train=ds(by_split["train"], train_tf),
            val=ds(by_split["val"], eval_tf),
            test=ds(by_split["test"], eval_tf),
            class_names=class_names,
            seen_test=ds(seen_rows, eval_tf) if seen_rows else None,
        )

    # Fallback: stratified random split over a plain ImageFolder.
    print("[data] stratified random split")
    from torch.utils.data import Subset
    from torchvision.datasets import ImageFolder

    class_names = ImageFolder(cfg.data_dir).classes
    targets = [label for _, label in ImageFolder(cfg.data_dir).samples]
    train_idx, val_idx, test_idx = _stratified_indices(
        targets, len(class_names), cfg.val_fraction, cfg.test_fraction, cfg.seed)
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


def _train_labels(train_ds):
    """Training-split labels without decoding any images where possible."""
    if isinstance(train_ds, ManifestDataset):
        return [train_ds.class_to_idx[r["class"]] for r in train_ds.rows]
    # torch Subset over an ImageFolder: read labels from .samples via indices.
    dataset = getattr(train_ds, "dataset", None)
    indices = getattr(train_ds, "indices", None)
    if dataset is not None and indices is not None and hasattr(dataset, "samples"):
        return [dataset.samples[i][1] for i in indices]
    return [label for _, label in train_ds]  # last resort (decodes images)


def class_weights(datasets: Datasets, n_classes: int):
    """Inverse-frequency class weights from the training split (for imbalance)."""
    import torch

    counts = np.zeros(n_classes, dtype=np.float64)
    for label in _train_labels(datasets.train):
        counts[label] += 1
    counts = np.maximum(counts, 1.0)
    weights = counts.sum() / (n_classes * counts)
    return torch.tensor(weights, dtype=torch.float32)
