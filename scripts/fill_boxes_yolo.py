#!/usr/bin/env python3
"""Fill missing animal boxes in the manifest using YOLOv8.

About half of the Caltech Camera Traps frames carry a ground-truth bounding box;
the rest are classified from the whole frame. This script runs a COCO-pretrained
YOLOv8 detector over the frames that lack a box and writes the detected box into
the manifest, so the load-time crop (``crop_to_bbox``) covers more images.

It operates on the already-stored frames (boxes are in stored-image coordinates),
so no re-download is needed. It adds a ``box_source`` column recording where each
box came from: ``gt`` (dataset ground truth), ``yolo`` (detected here), or
``none`` (no box — the frame is used whole). Checksums are unchanged because the
image files are not modified.

Usage:
    python scripts/fetch_yolo_weights.py        # once, if offline
    python scripts/fill_boxes_yolo.py --data-dir data/night_wildlife --conf 0.2
"""
import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.detect import load_detector, best_animal_box, yolo_available  # noqa: E402


def fill_manifest_boxes(data_dir, manifest_name="manifest.csv",
                        weights=None, conf=0.2):
    if not yolo_available():
        raise SystemExit("ultralytics is not installed: pip install ultralytics")
    path = os.path.join(data_dir, manifest_name)
    rows = list(csv.DictReader(open(path, newline="")))
    if not rows:
        raise SystemExit(f"empty manifest: {path}")

    model = load_detector(weights)
    fieldnames = list(rows[0].keys())
    if "box_source" not in fieldnames:
        fieldnames.insert(fieldnames.index("has_bbox") + 1, "box_source")

    counts = {"gt": 0, "yolo": 0, "none": 0}
    for r in rows:
        has_gt = str(r.get("has_bbox")).lower() == "true" and r.get("bbox")
        if has_gt:
            r["box_source"] = "gt"
            counts["gt"] += 1
            continue
        box = best_animal_box(model, os.path.join(data_dir, r["filename"]), conf=conf)
        if box is not None:
            r["bbox"] = ";".join(map(str, box))
            r["has_bbox"] = True
            r["box_source"] = "yolo"
            counts["yolo"] += 1
        else:
            r["box_source"] = "none"
            counts["none"] += 1

    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    total = len(rows)
    covered = counts["gt"] + counts["yolo"]
    print(f"[fill] box coverage {covered}/{total} = {covered/total:.0%} "
          f"(gt={counts['gt']}, yolo={counts['yolo']}, none={counts['none']})")
    print(f"[fill] updated {path}")
    return counts


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default="data/night_wildlife")
    ap.add_argument("--manifest-name", default="manifest.csv")
    ap.add_argument("--weights", default=None,
                    help="path to yolov8 weights (default: .cct_cache/yolov8n.pt)")
    ap.add_argument("--conf", type=float, default=0.2)
    args = ap.parse_args()
    fill_manifest_boxes(args.data_dir, args.manifest_name, args.weights, args.conf)
