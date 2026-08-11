#!/usr/bin/env python3
"""write a manifest whose boxes come only from a detector, never from ground truth."""
import argparse
import csv
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def build(data_dir, manifest_name, out_name, splits, detector, weights, conf):
    src = os.path.join(data_dir, manifest_name)
    rows = list(csv.DictReader(open(src, newline="")))
    if not rows:
        raise SystemExit(f"empty manifest: {src}")
    fieldnames = list(rows[0].keys())
    for required in ("split", "bbox", "has_bbox", "box_source"):
        if required not in fieldnames:
            raise SystemExit(f"{src} has no '{required}' column")

    if detector == "megadetector":
        from src import megadetector as md
        if not md.available():
            raise SystemExit('yolov5 is not installed: pip install yolov5 "setuptools<81"')
        model = md.load_detector(weights)
        def find(path):  # noqa: E306
            return md.best_animal_box(model, path, conf=conf)
    else:
        from src.detect import load_detector, best_animal_box, yolo_available
        if not yolo_available():
            raise SystemExit("ultralytics is not installed: pip install ultralytics")
        model = load_detector(weights)
        def find(path):  # noqa: E306
            return best_animal_box(model, path, conf=conf)

    targets = [r for r in rows if r["split"] in splits]
    print(f"[detector-manifest] re-detecting {len(targets)} rows in splits={sorted(splits)}")

    counts, replaced_gt = Counter(), 0
    for i, r in enumerate(targets, 1):
        was_gt = r.get("box_source") == "gt"
        box = find(os.path.join(data_dir, r["filename"]))
        if box is not None:
            r["bbox"] = ";".join(map(str, box))
            r["has_bbox"] = True
            r["box_source"] = detector
            counts[detector] += 1
        else:
            # no detection means no crop, exactly as it would go live
            r["bbox"] = ""
            r["has_bbox"] = False
            r["box_source"] = "none"
            counts["none"] += 1
        replaced_gt += was_gt
        if i % 25 == 0 or i == len(targets):
            print(f"  {i}/{len(targets)}")

    dst = os.path.join(data_dir, out_name)
    with open(dst, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    covered = len(targets) - counts["none"]
    print(f"[detector-manifest] detector coverage on those rows: "
          f"{covered}/{len(targets)} = {covered/max(len(targets),1):.0%} ({dict(counts)})")
    print(f"[detector-manifest] ground-truth boxes replaced: {replaced_gt}")
    print(f"[detector-manifest] wrote {dst}")
    return dst


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default="data/night_wildlife")
    ap.add_argument("--manifest-name", default="manifest.csv")
    ap.add_argument("--out-name", default="manifest_detector.csv")
    ap.add_argument("--splits", nargs="+", default=["test"],
                    choices=["train", "val", "test"],
                    help="splits to re-detect; others are copied unchanged")
    ap.add_argument("--detector", default="megadetector",
                    choices=["megadetector", "yolov8"])
    ap.add_argument("--weights", default=None)
    ap.add_argument("--conf", type=float, default=0.2)
    args = ap.parse_args()
    build(args.data_dir, args.manifest_name, args.out_name, set(args.splits),
          args.detector, args.weights, args.conf)
