#!/usr/bin/env python3
"""Build a committed real-image dataset from the CIFAR-10-images repository.

The real NTLNP night-vision dataset (25,657 images) lives on Hugging Face, which
is blocked by the build sandbox's egress policy. So that the repo ships with a
*real* (not synthetic) image dataset that the whole pipeline can train and
evaluate on, this script samples the six animal classes from the openly
available CIFAR-10-images repository:

    https://github.com/YoongiKim/CIFAR-10-images   (real 32x32 photographs)

These are genuine photographs (not fabricated), organised one folder per class
in the ImageFolder layout the pipeline expects. They are ordinary daylight
photos rather than infrared camera-trap frames, so they are a *substitute for
demonstrating the classifier on real data*, not the NTLNP data itself — use
scripts/download_ntlnp.sh for the real thing on an unrestricted machine.

Usage:
    # 1. clone the source (once):
    git clone --depth 1 https://github.com/YoongiKim/CIFAR-10-images
    # 2. build the committed subset:
    python scripts/prepare_real_dataset.py --src CIFAR-10-images \
        --out data/real_animals --per-class 500
"""
import argparse
import os
import shutil

# CIFAR-10 classes that are animals (the wildlife-relevant subset).
ANIMAL_CLASSES = ["bird", "cat", "deer", "dog", "frog", "horse"]


def build(src: str, out: str, per_class: int) -> None:
    if not os.path.isdir(src):
        raise SystemExit(
            f"source '{src}' not found. Clone it first:\n"
            f"  git clone --depth 1 https://github.com/YoongiKim/CIFAR-10-images")

    total = 0
    for cls in ANIMAL_CLASSES:
        dst_dir = os.path.join(out, cls)
        os.makedirs(dst_dir, exist_ok=True)
        # Pool train + test so we get some variety; take a deterministic slice.
        pool = []
        for split in ("train", "test"):
            d = os.path.join(src, split, cls)
            if os.path.isdir(d):
                pool += [os.path.join(d, f) for f in sorted(os.listdir(d))
                         if f.lower().endswith((".png", ".jpg", ".jpeg"))]
        pool = sorted(pool)[:per_class]
        for i, path in enumerate(pool):
            shutil.copy(path, os.path.join(dst_dir, f"{cls}_{i:04d}.png"))
        total += len(pool)
        print(f"  {cls}: {len(pool)} images")
    print(f"[real-dataset] {len(ANIMAL_CLASSES)} classes, {total} images -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default="CIFAR-10-images",
                    help="path to a clone of YoongiKim/CIFAR-10-images")
    ap.add_argument("--out", default="data/real_animals")
    ap.add_argument("--per-class", type=int, default=500)
    args = ap.parse_args()
    build(args.src, args.out, args.per_class)
