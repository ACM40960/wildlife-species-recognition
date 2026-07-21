"""Experiment configuration.

A single dataclass holds every knob the pipeline needs so that training and
evaluation read from one place. Values can be overridden on the command line by
the scripts in ``scripts/``.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional
import json


@dataclass
class Config:
    # --- Data ---
    data_dir: str = "data/night_wildlife"  # root with one sub-folder per class
    image_size: int = 224                  # square input size fed to the CNN
    val_fraction: float = 0.15             # split fractions (train = remainder)
    test_fraction: float = 0.15
    split_by: str = "location"             # "location" (manifest, held-out sites)
                                           # or "stratified" (random per class)
    manifest_name: str = "manifest.csv"
    grayscale_to_rgb: bool = True          # IR frames are 1-channel; repeat to 3
    num_workers: int = 2

    # --- Model ---
    backbone: str = "resnet18"             # resnet18 | resnet34 | resnet50
    pretrained: bool = True                # start from ImageNet weights
    freeze_until: str = "layer4"           # freeze everything up to this block;
                                           # "" trains all layers, "all" freezes
                                           # the whole backbone (linear probe)

    # --- Training ---
    epochs: int = 15
    batch_size: int = 32
    learning_rate: float = 1e-3            # for the newly added / unfrozen params
    weight_decay: float = 1e-4
    label_smoothing: float = 0.05
    early_stop_patience: int = 5           # stop if val acc stalls this many epochs

    # --- Runtime ---
    seed: int = 42
    device: str = "auto"                   # auto | cpu | cuda
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
            return cls(**json.load(fh))
