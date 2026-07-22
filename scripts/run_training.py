#!/usr/bin/env python3
"""Train the transfer-learning model.

Examples:
    # Train on the infrared night-vision data under data/night_wildlife/<species>/
    python scripts/run_training.py --data-dir data/night_wildlife --epochs 16 \
        --pretrained --grayscale --freeze-until layer2

    # Real infrared night-vision demo dataset (Caltech Camera Traps subset)
    python scripts/run_training.py --data-dir data/night_wildlife --epochs 16 \
        --image-size 224 --pretrained --grayscale --freeze-until layer2 \
        --output-dir results/demo
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
    ap.add_argument("--grayscale", dest="grayscale_to_rgb", action="store_true", default=None,
                    help="force grayscale->RGB (the infrared path; default)")
    ap.add_argument("--no-grayscale", dest="grayscale_to_rgb", action="store_false",
                    help="keep colour input (for ordinary colour photo datasets)")
    ap.add_argument("--split-by", dest="split_by", default=None,
                    choices=["location", "stratified"],
                    help="location = held-out camera sites (manifest); "
                         "stratified = random per class (same-location)")
    ap.add_argument("--crop-to-bbox", dest="crop_to_bbox", action="store_true", default=None,
                    help="crop to the detected animal box (default)")
    ap.add_argument("--no-crop-to-bbox", dest="crop_to_bbox", action="store_false",
                    help="use the full frame (for the full-vs-detected comparison)")
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--device", default=None, choices=["auto", "cpu", "cuda"])
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    cfg = Config()
    for key in ("data_dir", "backbone", "epochs", "batch_size", "learning_rate",
                "image_size", "freeze_until", "pretrained", "grayscale_to_rgb",
                "split_by", "crop_to_bbox", "output_dir", "device", "seed"):
        val = getattr(args, key, None)
        if val is not None:
            setattr(cfg, key, val)
    return cfg


if __name__ == "__main__":
    train(build_config())
