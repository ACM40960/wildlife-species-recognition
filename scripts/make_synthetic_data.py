#!/usr/bin/env python3
"""Generate a synthetic night-vision dataset for demos / CI.

Usage:
    python scripts/make_synthetic_data.py --out data/synthetic --per-class 120
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.synthetic import generate  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/synthetic")
    ap.add_argument("--per-class", type=int, default=120)
    ap.add_argument("--image-size", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    generate(args.out, per_class=args.per_class, image_size=args.image_size, seed=args.seed)


if __name__ == "__main__":
    main()
