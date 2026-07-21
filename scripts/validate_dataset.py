#!/usr/bin/env python3
"""Validate the night-vision dataset before training.

Checks, and fails (non-zero exit) on any problem:

  1. manifest <-> files agree (every manifest row has a file, no orphan files);
  2. file integrity — each image opens and its SHA-256 matches the manifest;
  3. class balance — every class present, counts within tolerance;
  4. visible-animal annotation — how many frames carry a bounding box;
  5. split overlap — no image assigned to more than one split;
  6. location overlap — no camera location shared between splits (the whole point
     of the location-held-out split);
  7. every class appears in every split.

Usage:
    python scripts/validate_dataset.py --data-dir data/night_wildlife
"""
import argparse
import csv
import hashlib
import os
import sys
from collections import Counter, defaultdict


def _fail(msg, problems):
    problems.append(msg)
    print(f"  FAIL  {msg}")


def _ok(msg):
    print(f"  ok    {msg}")


def validate(data_dir, manifest_name="manifest.csv", tolerance=0.0):
    problems = []
    manifest_path = os.path.join(data_dir, manifest_name)
    if not os.path.exists(manifest_path):
        print(f"no manifest at {manifest_path}")
        return 1
    rows = list(csv.DictReader(open(manifest_path, newline="")))
    print(f"manifest: {len(rows)} rows\n")

    # 1. manifest <-> files
    manifest_files = {r["filename"] for r in rows}
    disk_files = set()
    for root, _dirs, files in os.walk(data_dir):
        for f in files:
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                rel = os.path.relpath(os.path.join(root, f), data_dir).replace(os.sep, "/")
                disk_files.add(rel)
    missing = manifest_files - disk_files
    orphan = disk_files - manifest_files
    if missing:
        _fail(f"{len(missing)} manifest rows have no file (e.g. {sorted(missing)[:3]})", problems)
    if orphan:
        _fail(f"{len(orphan)} files not in manifest — possible stale files "
              f"(e.g. {sorted(orphan)[:3]})", problems)
    if not missing and not orphan:
        _ok(f"manifest matches {len(disk_files)} files on disk")

    # 2. integrity (open + checksum)
    bad = 0
    try:
        from PIL import Image
    except Exception:
        Image = None
    for r in rows:
        p = os.path.join(data_dir, r["filename"])
        if not os.path.exists(p):
            continue
        data = open(p, "rb").read()
        if hashlib.sha256(data).hexdigest() != r.get("checksum"):
            bad += 1
            continue
        if Image is not None:
            try:
                Image.open(p).verify()
            except Exception:
                bad += 1
    if bad:
        _fail(f"{bad} files failed checksum or would not open", problems)
    else:
        _ok("all files pass checksum and open cleanly")

    # 3. class balance
    per_class = Counter(r["class"] for r in rows)
    counts = list(per_class.values())
    spread = (max(counts) - min(counts)) / max(counts) if counts else 1
    if spread > tolerance + 1e-9:
        _fail(f"class counts imbalanced (spread {spread:.2f}): {dict(per_class)}", problems)
    else:
        _ok(f"class balance ok: {dict(per_class)}")

    # 4. visible-animal annotation (informational, not a hard fail)
    with_box = sum(1 for r in rows if str(r.get("has_bbox")).lower() == "true")
    _ok(f"frames with a bounding box: {with_box}/{len(rows)} "
        f"({100*with_box/max(len(rows),1):.0f}%); rest use whole-frame letterbox")

    # 5. split overlap (each filename in exactly one split — trivially true per row,
    #    but check no filename is duplicated across rows)
    fn_counts = Counter(r["filename"] for r in rows)
    dupes = [f for f, c in fn_counts.items() if c > 1]
    if dupes:
        _fail(f"{len(dupes)} filenames appear in multiple manifest rows", problems)
    else:
        _ok("no duplicate filenames across splits")

    # 6. location overlap between splits
    loc_splits = defaultdict(set)
    for r in rows:
        loc_splits[r["location"]].add(r["split"])
    leaked = {loc: s for loc, s in loc_splits.items() if len(s) > 1}
    if leaked:
        _fail(f"{len(leaked)} camera locations span >1 split (background leakage): "
              f"{dict(list(leaked.items())[:3])}", problems)
    else:
        _ok(f"no location shared between splits ({len(loc_splits)} locations)")

    # 7. every class in every split
    cls_split = defaultdict(set)
    for r in rows:
        cls_split[r["class"]].add(r["split"])
    for cls, splits in cls_split.items():
        missing_splits = {"train", "val", "test"} - splits
        if missing_splits:
            _fail(f"class '{cls}' missing from split(s) {missing_splits}", problems)
    if all({"train", "val", "test"} <= s for s in cls_split.values()):
        _ok("every class present in train/val/test")

    print()
    if problems:
        print(f"VALIDATION FAILED: {len(problems)} problem(s)")
        return 1
    print("VALIDATION PASSED")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default="data/night_wildlife")
    ap.add_argument("--manifest-name", default="manifest.csv")
    ap.add_argument("--tolerance", type=float, default=0.0,
                    help="allowed relative spread in per-class counts (0 = exact)")
    args = ap.parse_args()
    sys.exit(validate(args.data_dir, args.manifest_name, args.tolerance))
