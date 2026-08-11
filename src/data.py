"""dataset loading, splitting, and preprocessing."""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

# ImageNet normalisation constants (the backbone was pretrained with these)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class Letterbox:
    """resize preserving aspect ratio and pad to a square — nothing is cut off."""

    def __init__(self, size: int, fill: int = 114):
        self.size = size
        self.fill = fill

    def __call__(self, img):
        from PIL import Image

        w, h = img.size
        scale = self.size / max(w, h)
        nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
        img = img.resize((nw, nh), Image.BILINEAR)
        canvas = Image.new(img.mode, (self.size, self.size),
                           self.fill if img.mode == "L" else (self.fill,) * 3)
        canvas.paste(img, ((self.size - nw) // 2, (self.size - nh) // 2))
        return canvas

    def __repr__(self):
        return f"{type(self).__name__}(size={self.size})"


def build_transforms(image_size: int, grayscale_to_rgb: bool, train: bool,
                     pad_to_square: bool = True, ir_augment: bool = True):
    """return a torchvision transform pipeline."""
    from torchvision import transforms

    steps = []
    # infrared frames are effectively single-channel
    if grayscale_to_rgb:
        steps.append(transforms.Grayscale(num_output_channels=3))

    if train:
        if pad_to_square:
            # letterbox first so scale augmentation samples from the whole frame
            steps += [Letterbox(image_size),
                      transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0))]
        else:
            steps.append(transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0)))
        steps += [
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
        ]
        if ir_augment:
            steps += [
                # gamma/sharpness stand in for IR exposure and focus variation
                transforms.RandomApply(
                    [transforms.ColorJitter(brightness=(0.6, 1.6))], p=0.3),
                transforms.RandomAdjustSharpness(sharpness_factor=0.4, p=0.2),
            ]
    elif pad_to_square:
        steps.append(Letterbox(image_size))
    else:
        steps += [
            transforms.Resize(int(image_size * 1.15)),
            transforms.CenterCrop(image_size),
        ]

    steps += [
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]
    if train and ir_augment:
        # random erasing operates on tensors, so it comes after ToTensor
        steps.append(transforms.RandomErasing(p=0.25, scale=(0.02, 0.15)))
    return transforms.Compose(steps)


@dataclass
class Datasets:
    train: object
    val: object
    test: object  # unseen-location test (the honest held-out set)
    class_names: List[str]
    seen_test: object = None  # held-out images from SEEN (training) locations


def _stratified_indices(targets: List[int], n_classes: int,
                        val_fraction: float, test_fraction: float,
                        seed: int) -> Tuple[List[int], List[int], List[int]]:
    """split indices per-class so every class is represented in each split."""
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
    """read manifest rows, or None if there is no manifest."""
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
    """deterministically hold out a fraction of train rows as a seen-location test."""
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


BBOX_PAD = 0.15  # fraction of box size added as padding on each side


def lookup_bbox(image_path: str, manifest_name: str = "manifest.csv"):
    """find an image's recorded bounding box by searching parent dirs for a manifest."""
    path = os.path.abspath(image_path)
    directory = os.path.dirname(path)
    while True:
        candidate = os.path.join(directory, manifest_name)
        if os.path.exists(candidate):
            rel = os.path.relpath(path, directory).replace(os.sep, "/")
            for row in read_manifest(directory, manifest_name) or []:
                if row.get("filename") == rel:
                    return _parse_bbox(row.get("bbox", ""))
            return None
        parent = os.path.dirname(directory)
        if parent == directory:  # reached the filesystem root
            return None
        directory = parent


def crop_to_box(img, box, pad: float = BBOX_PAD):
    """crop a PIL image to a padded ``(x, y, w, h)`` box."""
    if box is None:
        return img
    x, y, w, h = box
    px, py = w * pad, h * pad
    W, H = img.size
    left, top = max(0, int(x - px)), max(0, int(y - py))
    right, bottom = min(W, int(x + w + px)), min(H, int(y + h + py))
    if right - left <= 5 or bottom - top <= 5:
        return img
    return img.crop((left, top, right, bottom))


class ManifestDataset:
    """dataset driven by the manifest, cropping to the animal box at *load* time."""

    def __init__(self, rows, data_dir, class_to_idx, transform,
                 crop_to_bbox=True, bbox_pad=BBOX_PAD):
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
        img = crop_to_box(img, box, self.bbox_pad)
        return self.transform(img), self.class_to_idx[row["class"]]


def load_datasets(cfg) -> Datasets:
    """produce train/val/test datasets."""
    rows = read_manifest(cfg.data_dir, getattr(cfg, "manifest_name", "manifest.csv"))
    split_by = getattr(cfg, "split_by", "location")

    pad = getattr(cfg, "pad_to_square", True)
    ir_aug = getattr(cfg, "ir_augment", True)
    train_tf = build_transforms(cfg.image_size, cfg.grayscale_to_rgb, train=True,
                                pad_to_square=pad, ir_augment=ir_aug)
    eval_tf = build_transforms(cfg.image_size, cfg.grayscale_to_rgb, train=False,
                               pad_to_square=pad, ir_augment=ir_aug)

    if rows is not None:
        print(f"[data] {split_by} split from manifest; crop_to_bbox="
              f"{getattr(cfg, 'crop_to_bbox', True)}")
        class_names = sorted({r["class"] for r in rows})
        class_to_idx = {c: i for i, c in enumerate(class_names)}
        by_split = {"train": [], "val": [], "test": []}
        if split_by == "stratified":
            # random per-class split over the same rows, so the location-vs-random comparison
            labels = [class_to_idx[r["class"]] for r in rows]
            tr, va, te = _stratified_indices(labels, len(class_names),
                                             cfg.val_fraction, cfg.test_fraction,
                                             cfg.seed)
            for name, idxs in (("train", tr), ("val", va), ("test", te)):
                by_split[name] = [rows[i] for i in idxs]
        else:
            for r in rows:
                if r.get("split") in by_split:
                    by_split[r["split"]].append(r)

        # carve a SEEN-location test set: hold out a fraction of the train-location images
        seen_frac = getattr(cfg, "seen_test_fraction", 0.0)
        seen_rows = []
        if seen_frac > 0 and split_by == "location":
            keep, seen_rows = _carve_seen_test(by_split["train"], seen_frac, cfg.seed)
            by_split["train"] = keep
        if not by_split["train"] or not by_split["test"] or not by_split["val"]:
            raise ValueError(
                f"manifest at {cfg.data_dir} produced an empty train or test split "
                f"(train={len(by_split['train'])}, val={len(by_split['val'])}, "
                f"test={len(by_split['test'])}). Does it have a 'split' column with "
                "train/val/test values? Rebuild it with scripts/build_night_wildlife.py.")
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

    # fallback: stratified random split over a plain ImageFolder
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


def safe_num_workers(requested: int, min_shm_mb: int = 256) -> int:
    """drop to 0 workers when /dev/shm is too small to be safe."""
    if requested <= 0:
        return 0
    try:
        st = os.statvfs("/dev/shm")
        shm_mb = st.f_blocks * st.f_frsize / (1024 * 1024)
    except (OSError, AttributeError):
        return requested  # not Linux / can't tell: respect it
    if shm_mb < min_shm_mb:
        print(f"[data] /dev/shm is only {shm_mb:.0f}MB; using num_workers=0 "
              f"(requested {requested}) to avoid shared-memory errors.")
        return 0
    return requested


def make_loaders(cfg, datasets: Datasets):
    """wrap the subsets in DataLoaders."""
    from torch.utils.data import DataLoader

    workers = safe_num_workers(cfg.num_workers)
    common = dict(batch_size=cfg.batch_size, num_workers=workers,
                  pin_memory=(cfg.resolved_device() == "cuda"))
    return (
        DataLoader(datasets.train, shuffle=True, **common),
        DataLoader(datasets.val, shuffle=False, **common),
        DataLoader(datasets.test, shuffle=False, **common),
    )


def _train_labels(train_ds):
    """training-split labels without decoding any images where possible."""
    if isinstance(train_ds, ManifestDataset):
        return [train_ds.class_to_idx[r["class"]] for r in train_ds.rows]
    # torch Subset over an ImageFolder: read labels from .samples via indices
    dataset = getattr(train_ds, "dataset", None)
    indices = getattr(train_ds, "indices", None)
    if dataset is not None and indices is not None and hasattr(dataset, "samples"):
        return [dataset.samples[i][1] for i in indices]
    return [label for _, label in train_ds]  # last resort (decodes images)


def class_weights(datasets: Datasets, n_classes: int):
    """inverse-frequency class weights from the training split (for imbalance)."""
    import torch

    counts = np.zeros(n_classes, dtype=np.float64)
    for label in _train_labels(datasets.train):
        counts[label] += 1
    counts = np.maximum(counts, 1.0)
    weights = counts.sum() / (n_classes * counts)
    return torch.tensor(weights, dtype=torch.float32)
