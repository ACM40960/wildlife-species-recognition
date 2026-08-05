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
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.detect import load_detector, best_animal_box, yolo_available  # noqa: E402


def fill_manifest_boxes(data_dir, manifest_name="manifest.csv",
                        weights=None, conf=0.2, detector="megadetector",
                        refresh=False):
    """Fill missing boxes using MegaDetector (default) or COCO YOLOv8.

    MegaDetector is trained on camera-trap imagery (including infrared) and finds
    animals in far more of these frames than a COCO-pretrained detector.
    """
    if detector == "megadetector":
        from src import megadetector as md
        if not md.available():
            raise SystemExit('yolov5 is not installed: pip install yolov5 "setuptools<81"')
        model = md.load_detector(weights)
        find_box = lambda path: md.best_animal_box(model, path, conf=conf)  # noqa: E731
    else:
        if not yolo_available():
            raise SystemExit("ultralytics is not installed: pip install ultralytics")
        model = load_detector(weights)
        find_box = lambda path: best_animal_box(model, path, conf=conf)  # noqa: E731
    path = os.path.join(data_dir, manifest_name)
    rows = list(csv.DictReader(open(path, newline="")))
    if not rows:
        raise SystemExit(f"empty manifest: {path}")

    fieldnames = list(rows[0].keys())
    if "box_source" not in fieldnames:
        fieldnames.insert(fieldnames.index("has_bbox") + 1, "box_source")

    counts = Counter()
    for r in rows:
        existing = (r.get("box_source") or "").strip()
        has_box = str(r.get("has_bbox")).lower() == "true" and r.get("bbox")
        # Never relabel a box we already have. Re-running used to mark every
        # boxed row as "gt", which destroyed the record of which detector (or the
        # dataset itself) supplied it. Only rows with no box are detected now.
        # Ground truth is never re-detected, even with --refresh.
        if has_box and (existing == "gt" or not refresh):
            if not existing or existing == "none":
                # Legacy manifest with no provenance column: a pre-existing box
                # can only have come from the dataset's ground truth.
                existing = "gt"
            r["box_source"] = existing
            counts[existing] += 1
            continue
        box = find_box(os.path.join(data_dir, r["filename"]))
        if box is not None:
            r["bbox"] = ";".join(map(str, box))
            r["has_bbox"] = True
            r["box_source"] = detector
            counts[detector] += 1
        else:
            r["box_source"] = "none"
            counts["none"] += 1

    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    total = len(rows)
    covered = total - counts["none"]
    print(f"[fill] box coverage {covered}/{total} = {covered/total:.0%} "
          f"({dict(counts)})")
    print(f"[fill] updated {path}")
    return counts


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default="data/night_wildlife")
    ap.add_argument("--manifest-name", default="manifest.csv")
    ap.add_argument("--weights", default=None,
                    help="detector weights (default: .cct_cache/md_v5a.0.0.pt for "
                         "megadetector, .cct_cache/yolov8n.pt for yolov8)")
    ap.add_argument("--conf", type=float, default=0.2)
    ap.add_argument("--refresh", action="store_true",
                    help="re-detect every frame, replacing existing detector boxes "
                         "(ground-truth rows are still preserved)")
    ap.add_argument("--detector", default="megadetector",
                    choices=["megadetector", "yolov8"],
                    help="megadetector = camera-trap detector (recommended)")
    args = ap.parse_args()
    fill_manifest_boxes(args.data_dir, args.manifest_name, args.weights,
                        args.conf, args.detector, args.refresh)
