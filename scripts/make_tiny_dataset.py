#!/usr/bin/env python3
"""Generate a tiny synthetic dataset for CI smoke tests (no network needed).

Creates a handful of solid-ish grayscale images per class across several camera
"locations", with a manifest identical in shape to the real one (location-grouped
split, bounding boxes, checksums), so `run_training.py` / `run_evaluation.py` can
be exercised end-to-end in seconds. This is NOT real data — it only checks the
pipeline runs and the outputs are produced.

Usage:
    python scripts/make_tiny_dataset.py --out data/_tiny
"""
import argparse
import csv
import hashlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.split import location_grouped_split  # noqa: E402


def make(out_dir, classes=("aa", "bb", "cc"), n_locations=6, per_loc=3,
         image_size=64, seed=0):
    import numpy as np
    from PIL import Image

    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(seed)
    records = []
    for ci, cls in enumerate(classes):
        cdir = os.path.join(out_dir, cls)
        os.makedirs(cdir, exist_ok=True)
        idx = 0
        for loc in range(n_locations):
            for _ in range(per_loc):
                # class-distinct base brightness + noise so it is learnable
                base = 40 + ci * 60
                arr = np.clip(base + rng.normal(0, 15, (image_size, image_size)),
                              0, 255).astype("uint8")
                # a bright block whose position depends on the class (a fake "animal")
                y0 = 8 + ci * 10
                arr[y0:y0 + 20, 8:40] = 220
                fname = f"{cls}_{idx:03d}.jpg"
                path = os.path.join(cdir, fname)
                Image.fromarray(arr, mode="L").save(path)
                checksum = hashlib.sha256(open(path, "rb").read()).hexdigest()
                records.append({
                    "class": cls, "filename": f"{cls}/{fname}",
                    "image_id": f"{cls}-{loc}-{idx}", "orig_filename": fname,
                    "location": f"L{loc}", "seq_id": f"{cls}-{loc}-{idx}",
                    "timestamp": "2013-01-01 22:00:00", "month": "2013-01",
                    "season": "winter", "bbox": "8;{};32;20".format(y0),
                    "has_bbox": True, "box_source": "gt", "checksum": checksum,
                })
                idx += 1

    split = location_grouped_split(records, 0.2, 0.2, seed)
    for r in records:
        r["split"] = split[r["location"]]

    cols = ["split", "class", "filename", "image_id", "orig_filename", "location",
            "seq_id", "timestamp", "month", "season", "bbox", "has_bbox",
            "box_source", "checksum"]
    with open(os.path.join(out_dir, "manifest.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(sorted(records, key=lambda r: r["filename"]))
    print(f"[tiny] {len(classes)} classes, {len(records)} images -> {out_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/_tiny")
    ap.add_argument("--image-size", type=int, default=64)
    args = ap.parse_args()
    make(args.out, image_size=args.image_size)
