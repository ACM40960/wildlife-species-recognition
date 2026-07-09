#!/usr/bin/env python3
"""Train the transfer-learning model.

Examples:
    # Train on the real NTLNP data laid out under data/ntlnp/<species>/
    python scripts/run_training.py --data-dir data/ntlnp --epochs 15

    # Quick demo on synthetic data (few epochs, CPU-friendly)
    python scripts/run_training.py --data-dir data/synthetic --epochs 5 \
        --image-size 128 --output-dir results/demo
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import Config  # noqa: E402
from src.train import train  # noqa: E402


def build_config() -> Config:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--backbone", default=None, choices=["resnet18", "resnet34", "resnet50"])
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--learning-rate", type=float, default=None)
    ap.add_argument("--image-size", type=int, default=None)
    ap.add_argument("--freeze-until", default=None,
                    help="'', 'all', or a resnet block name like layer4")
    ap.add_argument("--pretrained", dest="pretrained", action="store_true", default=None,
                    help="start from ImageNet weights (default; needs internet)")
    ap.add_argument("--no-pretrained", dest="pretrained", action="store_false",
                    help="train the backbone from scratch (offline/CI)")
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--device", default=None, choices=["auto", "cpu", "cuda"])
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    cfg = Config()
    for key in ("data_dir", "backbone", "epochs", "batch_size", "learning_rate",
                "image_size", "freeze_until", "pretrained", "output_dir", "device", "seed"):
        val = getattr(args, key, None)
        if val is not None:
            setattr(cfg, key, val)
    return cfg


if __name__ == "__main__":
    train(build_config())
