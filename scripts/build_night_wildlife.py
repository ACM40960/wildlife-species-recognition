#!/usr/bin/env python3
"""Build a real night-vision (infrared) wildlife dataset from Caltech Camera Traps.

The project targets night-vision / infrared camera-trap imagery. The LILA BC
repository hosts the **Caltech Camera Traps (CCT)** dataset on Google Cloud
Storage (storage.googleapis.com), so this script assembles a genuine infrared
night-vision wildlife dataset from it.

CCT contains ~243k camera-trap frames from the U.S. Southwest with species
labels, bounding boxes, capture timestamps and a camera-location id. At night the
traps switch to an infrared flash, producing grayscale, low-contrast frames —
precisely the imagery this project is about. This script:

  1. downloads the CCT label + bounding-box metadata (COCO format) from LILA's GCS mirror,
  2. selects a set of well-represented wild species, spread across many camera
     locations (a per-location cap stops any single site dominating a class, so a
     location-held-out split has material to work with),
  3. keeps only NIGHT captures (by timestamp) that are verified to be grayscale
     (mean HSV saturation below a threshold => genuine infrared frame),
  4. de-duplicates by capture sequence so near-identical burst frames do not leak
     between splits,
  5. **crops to the animal's bounding box** (with padding) when one exists, and
     otherwise uses an aspect-preserving *letterbox* resize — never a blind
     centre-crop that can slice the animal out of frame,
  6. assigns a **location-grouped split** (whole sites go to one split) and writes
     a full **manifest.csv** (source image id, original filename, class, location,
     sequence id, timestamp, split, sha256 checksum) so the dataset is verifiable
     and the experiment reproducible.

Data source / credit: Caltech Camera Traps, Beery et al., "Recognition in Terra
Incognita" (ECCV 2018); distributed via https://lila.science/datasets/caltech-camera-traps .
Licensed Community Data License Agreement (permissive variant).

Usage:
    python scripts/build_night_wildlife.py --out data/night_wildlife \
        --per-class 200 --per-location-cap 20 --image-size 224
"""
import argparse
import csv
import hashlib
import io
import json
import os
import sys
import urllib.request
import zipfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.split import location_grouped_split  # noqa: E402

GCS = "https://storage.googleapis.com/public-datasets-lila"
LABELS_URL = f"{GCS}/caltechcameratraps/labels/caltech_camera_traps.json.zip"
BBOX_URL = f"{GCS}/caltechcameratraps/labels/caltech_bboxes_20200316.json"
IMAGES_BASE = f"{GCS}/caltech-unzipped/cct_images/"

DEFAULT_SPECIES = ["bobcat", "coyote", "raccoon", "opossum", "rabbit", "deer"]
SAT_THRESHOLD = 6.0    # mean HSV saturation below this => grayscale / infrared
NIGHT_HOURS = set(range(19, 24)) | set(range(0, 7))


def _is_night(date_captured: str) -> bool:
    try:
        return int(date_captured[11:13]) in NIGHT_HOURS
    except Exception:
        return False


def load_metadata(cache_dir: str):
    os.makedirs(cache_dir, exist_ok=True)
    zpath = os.path.join(cache_dir, "cct_labels.json.zip")
    if not os.path.exists(zpath):
        print("[meta] downloading CCT labels...")
        urllib.request.urlretrieve(LABELS_URL, zpath)
    data = json.loads(zipfile.ZipFile(zpath).read(zipfile.ZipFile(zpath).namelist()[0]))
    cats = {c["id"]: c["name"] for c in data["categories"]}
    images = {i["id"]: i for i in data["images"]}
    img_cat = {}
    for a in data["annotations"]:
        img_cat.setdefault(a["image_id"], a["category_id"])

    bpath = os.path.join(cache_dir, "cct_bboxes.json")
    if not os.path.exists(bpath):
        print("[meta] downloading CCT bounding boxes...")
        urllib.request.urlretrieve(BBOX_URL, bpath)
    bdata = json.load(open(bpath))
    # image_id -> largest bounding box [x, y, w, h]
    boxes = {}
    for a in bdata["annotations"]:
        bb = a.get("bbox")
        if not bb:
            continue
        area = bb[2] * bb[3]
        cur = boxes.get(a["image_id"])
        if cur is None or area > cur[2] * cur[3]:
            boxes[a["image_id"]] = bb
    return cats, images, img_cat, boxes


def select_records(species_list, per_class, per_location_cap, cats, images, img_cat):
    """Choose night infrared candidates per species, spread across locations."""
    name_to_id = {v: k for k, v in cats.items()}
    selected = []
    for species in species_list:
        cid = name_to_id[species]
        per_loc = defaultdict(int)
        seen_seq = set()
        chosen = []
        # deterministic order; UUID ids give a location-mixed ordering
        night = []
        for iid, c in img_cat.items():
            if c != cid:
                continue
            rec = images.get(iid)
            if rec and _is_night(rec.get("date_captured", "")):
                night.append(rec)
        night.sort(key=lambda r: r["id"])
        for rec in night:
            seq = rec.get("seq_id", rec["id"])
            loc = rec.get("location", "?")
            if seq in seen_seq or per_loc[loc] >= per_location_cap:
                continue
            seen_seq.add(seq)
            per_loc[loc] += 1
            chosen.append(rec)
            if len(chosen) >= per_class * 2:   # over-select; some rejected as colour
                break
        selected.append((species, chosen))
    return selected


def letterbox(img, size, fill=0):
    """Aspect-preserving resize to a square canvas, padding the short side."""
    from PIL import Image
    w, h = img.size
    scale = size / max(w, h)
    nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
    img = img.resize((nw, nh))
    canvas = Image.new("L", (size, size), fill)
    canvas.paste(img, ((size - nw) // 2, (size - nh) // 2))
    return canvas


def fetch_one(rec, box, image_size, pad=0.15):
    """Download, verify grayscale, crop to bbox (padded) or letterbox whole frame."""
    from PIL import Image
    import numpy as np

    try:
        raw = urllib.request.urlopen(IMAGES_BASE + rec["file_name"], timeout=40).read()
        im = Image.open(io.BytesIO(raw))
        hsv = np.asarray(im.convert("HSV"), dtype=np.float32)
        if hsv[:, :, 1].mean() > SAT_THRESHOLD:
            return None                     # colour => daytime, skip
        g = im.convert("L")
        if box is not None:
            W, H = g.size
            x, y, bw, bh = box
            px, py = bw * pad, bh * pad
            left = max(0, int(x - px)); top = max(0, int(y - py))
            right = min(W, int(x + bw + px)); bottom = min(H, int(y + bh + py))
            if right - left > 5 and bottom - top > 5:
                g = g.crop((left, top, right, bottom))
        return letterbox(g, image_size)     # preserves aspect, no animal cut off
    except Exception:
        return None


def build(out_dir, species_list, per_class, per_location_cap, image_size,
          cache_dir, workers, val_fraction, test_fraction, seed):
    cats, images, img_cat, boxes = load_metadata(cache_dir)
    selected = select_records(species_list, per_class, per_location_cap,
                              cats, images, img_cat)

    saved_records = []          # rows for the manifest
    for species, cands in selected:
        cls_dir = os.path.join(out_dir, species)
        os.makedirs(cls_dir, exist_ok=True)
        saved = 0
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(fetch_one, rec, boxes.get(rec["id"]), image_size): rec
                    for rec in cands}
            for fut in as_completed(futs):
                if saved >= per_class:
                    break
                img = fut.result()
                if img is None:
                    continue
                rec = futs[fut]
                fname = f"{species}_{saved:04d}.jpg"
                path = os.path.join(cls_dir, fname)
                img.save(path, quality=88)
                checksum = hashlib.sha256(open(path, "rb").read()).hexdigest()
                saved_records.append({
                    "class": species,
                    "filename": f"{species}/{fname}",
                    "image_id": rec["id"],
                    "orig_filename": rec["file_name"],
                    "location": rec.get("location", "?"),
                    "seq_id": rec.get("seq_id", ""),
                    "timestamp": rec.get("date_captured", ""),
                    "has_bbox": boxes.get(rec["id"]) is not None,
                    "checksum": checksum,
                })
                saved += 1
        n_loc = len({r["location"] for r in saved_records if r["class"] == species})
        print(f"  {species:10s} {saved} infrared night images across {n_loc} locations")

    # Location-grouped split, then write the manifest.
    split_by_loc = location_grouped_split(saved_records, val_fraction, test_fraction, seed)
    for r in saved_records:
        r["split"] = split_by_loc[r["location"]]

    manifest_path = os.path.join(out_dir, "manifest.csv")
    cols = ["split", "class", "filename", "image_id", "orig_filename",
            "location", "seq_id", "timestamp", "has_bbox", "checksum"]
    with open(manifest_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in sorted(saved_records, key=lambda r: r["filename"]):
            w.writerow(r)

    from collections import Counter
    per_split = Counter(r["split"] for r in saved_records)
    print(f"[night-wildlife] {len(species_list)} classes, {len(saved_records)} images -> {out_dir}")
    print(f"[split] location-grouped: {dict(per_split)}")
    print(f"[manifest] {manifest_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data/night_wildlife")
    ap.add_argument("--species", nargs="*", default=DEFAULT_SPECIES)
    ap.add_argument("--per-class", type=int, default=200)
    ap.add_argument("--per-location-cap", type=int, default=20)
    ap.add_argument("--image-size", type=int, default=224)
    ap.add_argument("--cache-dir", default=".cct_cache")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--val-fraction", type=float, default=0.15)
    ap.add_argument("--test-fraction", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    build(args.out, args.species, args.per_class, args.per_location_cap,
          args.image_size, args.cache_dir, args.workers,
          args.val_fraction, args.test_fraction, args.seed)
