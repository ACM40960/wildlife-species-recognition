"""experiment configuration."""
from __future__ import annotations

from dataclasses import dataclass, asdict
import json


@dataclass
class Config:
    # --- Data ---
    data_dir: str = "data/night_wildlife"  # root with one sub-folder per class
    image_size: int = 224  # square input size fed to the CNN
    val_fraction: float = 0.15  # split fractions (train = remainder)
    test_fraction: float = 0.15
    split_by: str = "location"  # "location" (manifest, held-out sites)
                                           # or "stratified" (random per class)
    manifest_name: str = "manifest.csv"
    crop_to_bbox: bool = True  # crop to the animal box at load time
                                           # (frames are stored uncropped)
    seen_test_fraction: float = 0.15  # of train-location images, held out
                                           # (not trained) to measure accuracy on SEEN
    grayscale_to_rgb: bool = True  # IR frames are 1-channel; repeat to 3
    pad_to_square: bool = True  # letterbox-pad instead of centre-crop,
                                           # so no part of the frame is discarded
    ir_augment: bool = True  # gamma/sharpness jitter + random
                                           # erasing, suited to infrared frames
    num_workers: int = 0  # 0 = load in-process. DataLoader workers
                                           # pass tensors through /dev/shm, which is tiny

    # --- Model ---
    backbone: str = "resnet18"  # resnet18 | resnet34 | resnet50
    pretrained: bool = True  # start from ImageNet weights
    freeze_until: str = "layer2"  # freeze up to & incl. this block, then
                                           # retrain layer3/layer4/head

    # --- Training ---
    epochs: int = 16  # matches the reported/methodology setup
    batch_size: int = 32
    learning_rate: float = 3e-4  # for the newly added / unfrozen params
    weight_decay: float = 1e-4
    label_smoothing: float = 0.05
    early_stop_patience: int = 5  # stop if val acc stalls this many epochs

    # --- Evaluation ---
    tta: bool = False  # average over the horizontal mirror.
                                           # measured on this dataset it costs 2x inference and
    temperature_scaling: bool = True  # calibrate confidence on the val split

    # --- Runtime ---
    seed: int = 42
    deterministic: bool = True  # enable deterministic torch algorithms
    device: str = "auto"  # auto | cpu | cuda
    output_dir: str = "results"
    checkpoint_name: str = "best_model.pt"
    experiment_name: str = "night_wildlife_resnet18"

    def resolved_device(self) -> str:
        if self.device != "auto":
            return self.device
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"

    def to_json(self, path: str) -> None:
        with open(path, "w") as fh:
            json.dump(asdict(self), fh, indent=2)

    @classmethod
    def from_json(cls, path: str) -> "Config":
        with open(path) as fh:
            data = json.load(fh)
        # ignore unknown keys so a config.json written by a newer/older version (with fields
        known = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**known)
