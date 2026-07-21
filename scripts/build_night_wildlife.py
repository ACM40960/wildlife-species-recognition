#!/usr/bin/env python3
"""Build a real night-vision (infrared) wildlife dataset from Caltech Camera Traps.

Source: the LILA BC Google-Cloud mirror of the Caltech Camera Traps (CCT) dataset
(Beery et al., ECCV 2018), ~243k camera-trap frames with species labels, bounding
boxes, timestamps and a camera-location id. At night the traps switch to an
infrared flash, producing the grayscale, low-contrast frames this project targets.

Design notes (why the pipeline is shaped this way):

  * Deterministic. Records are selected in a fixed, stratified order *before* any
    download, and filenames / retention follow that order — not whichever
    concurrent download happens to finish first. Re-running with the same seed
    produces byte-identical file assignments.
  * Stratified sampling. Candidates are spread across camera locations *and* time
    (round-robin over locations; evenly spaced over each location's date range)
    to reduce selection bias, with a per-location cap.
  * Non-destructive. The downloaded frame is stored **uncropped** (only downscaled
    to --store-size for repo size); the animal bounding box is recorded in the
    manifest and the crop is applied at *load* time (see src/data.py). Nothing is
    permanently cropped, so the crop strategy can change without re-downloading.
  * Single-species only. Frames annotated with more than one species are excluded
    (this is single-label classification); the count is reported.
  * Transparent failures. Every rejected/failed image is logged with its id and
    the reason, and a summary is written to build_report.txt.
  * Clean rebuilds. The output directory is wiped first (guarded to a path under
    data/) so a smaller re-run cannot leave stale files behind.

The location-grouped train/val/test split and a full manifest.csv (with SHA-256
checksums) are written so the dataset is verifiable and reproducible; run
`python scripts/validate_dataset.py` afterwards to check it.

Usage:
    python scripts/build_night_wildlife.py --out data/night_wildlife \
        --per-class 200 --per-location-cap 20 --store-size 384
"""
import argparse
import csv
import hashlib
import io
import itertools
import json
import os
import shutil
import sys
import urllib.request
import zipfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.split import location_grouped_split  # noqa: E402

GCS = "https://storage.googleapis.com/public-datasets-lila"
LABELS_URL = f"{GCS}/caltechcameratraps/labels/caltech_camera_traps.json.zip"
BBOX_URL = f"{GCS}/caltechcameratraps/labels/caltech_bboxes_20200316.json"
IMAGES_BASE = f"{GCS}/caltech-unzipped/cct_images/"

DEFAULT_SPECIES = ["bobcat", "coyote", "raccoon", "opossum", "rabbit", "deer"]
SAT_THRESHOLD = 6.0                       # mean HSV saturation below => infrared
NIGHT_HOURS = set(range(19, 24)) | set(range(0, 7))
NON_ANIMAL_CATS = {30, 33}                # empty, car — not a "second species"


def _is_night(date_captured: str) -> bool:
    try:
        return int(date_captured[11:13]) in NIGHT_HOURS
    except Exception:
        return False


def _season(month: int) -> str:
    return {12: "winter", 1: "winter", 2: "winter", 3: "spring", 4: "spring",
            5: "spring", 6: "summer", 7: "summer", 8: "summer", 9: "autumn",
            10: "autumn", 11: "autumn"}.get(month, "?")


def load_metadata(cache_dir: str):
    os.makedirs(cache_dir, exist_ok=True)
    zpath = os.path.join(cache_dir, "cct_labels.json.zip")
    if not os.path.exists(zpath):
        print("[meta] downloading CCT labels...")
        urllib.request.urlretrieve(LABELS_URL, zpath)
    zf = zipfile.ZipFile(zpath)
    data = json.loads(zf.read(zf.namelist()[0]))
    cats = {c["id"]: c["name"] for c in data["categories"]}
    images = {i["id"]: i for i in data["images"]}

    # ALL species per image, so we can exclude multi-species frames.
    img_species = defaultdict(set)
    for a in data["annotations"]:
        if a["category_id"] not in NON_ANIMAL_CATS:
            img_species[a["image_id"]].add(a["category_id"])

    bpath = os.path.join(cache_dir, "cct_bboxes.json")
    if not os.path.exists(bpath):
        print("[meta] downloading CCT bounding boxes...")
        urllib.request.urlretrieve(BBOX_URL, bpath)
    bdata = json.load(open(bpath))
    boxes = {}
    for a in bdata["annotations"]:
        bb = a.get("bbox")
        if not bb:
            continue
        cur = boxes.get(a["image_id"])
        if cur is None or bb[2] * bb[3] > cur[2] * cur[3]:
            boxes[a["image_id"]] = bb            # largest box per image
    return cats, images, img_species, boxes


def _even_spread(items, k):
    """Pick up to k items evenly spaced across a list (spans the whole range)."""
    n = len(items)
    if n <= k:
        return items
    return [items[round(i * (n - 1) / (k - 1))] for i in range(k)]


def select_records(species, per_class, per_location_cap, cats, images, img_species):
    """Deterministic, location- and time-stratified candidate order for one species.

    Excludes multi-species frames and sequence duplicates. Returns a list of image
    records ordered so that consecutive picks come from different locations and
    span each location's date range.
    """
    cid = {v: k for k, v in cats.items()}[species]
    by_loc = defaultdict(list)
    seen_seq = set()
    for iid, sset in img_species.items():
        if sset != {cid}:                        # not exactly this one species
            continue
        rec = images.get(iid)
        if not rec or not _is_night(rec.get("date_captured", "")):
            continue
        seq = rec.get("seq_id", iid)
        if seq in seen_seq:
            continue
        seen_seq.add(seq)
        by_loc[rec.get("location", "?")].append(rec)

    # Within each location: sort by time, then take an even spread across dates.
    per_loc_ordered = {}
    for loc in sorted(by_loc):
        recs = sorted(by_loc[loc], key=lambda r: r.get("date_captured", ""))
        per_loc_ordered[loc] = _even_spread(recs, per_location_cap)

    # Round-robin across locations (sorted) => location-balanced, deterministic.
    columns = [per_loc_ordered[loc] for loc in sorted(per_loc_ordered)]
    ordered = [r for r in itertools.chain.from_iterable(
        itertools.zip_longest(*columns)) if r is not None]
    return ordered[: per_class * 3]              # oversample; some rejected later


def _process_frame(raw_bytes, store_size, box):
    """Return (grayscale PIL frame downscaled to store_size, scaled bbox or None).

    The frame is NOT cropped — only downscaled — so the original framing is kept.
    """
    from PIL import Image
    import numpy as np

    im = Image.open(io.BytesIO(raw_bytes))
    hsv = np.asarray(im.convert("HSV"), dtype=np.float32)
    if hsv[:, :, 1].mean() > SAT_THRESHOLD:
        raise ValueError("not-grayscale (daytime colour frame)")
    g = im.convert("L")
    W, H = g.size
    scale = store_size / max(W, H)
    g = g.resize((max(1, round(W * scale)), max(1, round(H * scale))))
    scaled_box = None
    if box is not None:
        x, y, bw, bh = box
        scaled_box = [round(x * scale), round(y * scale),
                      round(bw * scale), round(bh * scale)]
    return g, scaled_box


def fetch(rec, box, store_size):
    """Download and process one record. Returns (rec, image, scaled_box, error)."""
    try:
        raw = urllib.request.urlopen(IMAGES_BASE + rec["file_name"], timeout=45).read()
        img, scaled_box = _process_frame(raw, store_size, box)
        return rec, img, scaled_box, None
    except Exception as e:                        # report, don't hide (issue #10)
        return rec, None, None, f"{type(e).__name__}: {e}"


def build(out_dir, species_list, per_class, per_location_cap, store_size,
          cache_dir, workers, val_fraction, test_fraction, seed):
    cats, images, img_species, boxes = load_metadata(cache_dir)

    # Clean rebuild (issue #11): wipe the dataset dir, guarded to a data/ path.
    norm = os.path.normpath(out_dir)
    if os.path.exists(norm):
        assert "data" in norm.split(os.sep) and norm not in (".", "/"), \
            f"refusing to wipe unexpected path: {norm}"
        shutil.rmtree(norm)
    os.makedirs(norm, exist_ok=True)

    saved_records = []
    rejects = []                                  # (image_id, class, reason)
    for species in species_list:
        cls_dir = os.path.join(out_dir, species)
        os.makedirs(cls_dir, exist_ok=True)
        candidates = select_records(species, per_class, per_location_cap,
                                    cats, images, img_species)

        # Download concurrently, but map every result back to its record so that
        # retention + filenames follow the deterministic candidate order, not the
        # order downloads finish in (issue #6).
        results = {}
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(fetch, rec, boxes.get(rec["id"]), store_size)
                    for rec in candidates]
            for fut in as_completed(futs):
                rec, img, box, err = fut.result()
                results[rec["id"]] = (img, box, err)

        saved = 0
        for rec in candidates:                    # deterministic order
            if saved >= per_class:
                break
            img, box, err = results.get(rec["id"], (None, None, "no-result"))
            if img is None:
                rejects.append((rec["id"], species, err))
                continue
            fname = f"{species}_{saved:04d}.jpg"
            path = os.path.join(cls_dir, fname)
            img.save(path, quality=90)
            checksum = hashlib.sha256(open(path, "rb").read()).hexdigest()
            ts = rec.get("date_captured", "")
            month = ts[:7]
            saved_records.append({
                "class": species,
                "filename": f"{species}/{fname}",
                "image_id": rec["id"],
                "orig_filename": rec["file_name"],
                "location": rec.get("location", "?"),
                "seq_id": rec.get("seq_id", ""),
                "timestamp": ts,
                "month": month,
                "season": _season(int(ts[5:7])) if len(ts) >= 7 else "?",
                "bbox": ";".join(map(str, box)) if box else "",
                "has_bbox": box is not None,
                "checksum": checksum,
            })
            saved += 1
        n_loc = len({r["location"] for r in saved_records if r["class"] == species})
        n_month = len({r["month"] for r in saved_records if r["class"] == species})
        print(f"  {species:10s} {saved} images | {n_loc} locations | {n_month} months "
              f"| {sum(1 for r in saved_records if r['class']==species and r['has_bbox'])} with bbox")

    # Location-grouped split, then write the manifest.
    split_by_loc = location_grouped_split(saved_records, val_fraction, test_fraction, seed)
    for r in saved_records:
        r["split"] = split_by_loc[r["location"]]

    manifest_path = os.path.join(out_dir, "manifest.csv")
    cols = ["split", "class", "filename", "image_id", "orig_filename", "location",
            "seq_id", "timestamp", "month", "season", "bbox", "has_bbox", "checksum"]
    with open(manifest_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in sorted(saved_records, key=lambda r: r["filename"]):
            w.writerow(r)

    # Build report (issue #10).
    per_split = Counter(r["split"] for r in saved_records)
    reason_counts = Counter(r[2].split(":")[0] for r in rejects)
    report_path = os.path.join(out_dir, "build_report.txt")
    with open(report_path, "w") as fh:
        fh.write(f"images saved: {len(saved_records)}\n")
        fh.write(f"split (location-grouped): {dict(per_split)}\n")
        fh.write(f"rejected/failed: {len(rejects)}\n")
        fh.write(f"reject reasons: {dict(reason_counts)}\n\n")
        for iid, cls, reason in rejects:
            fh.write(f"REJECT {cls} {iid} {reason}\n")

    print(f"[night-wildlife] {len(species_list)} classes, {len(saved_records)} images -> {out_dir}")
    print(f"[split] {dict(per_split)}")
    print(f"[rejects] {len(rejects)} ({dict(reason_counts)}) -> {report_path}")
    print(f"[manifest] {manifest_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data/night_wildlife")
    ap.add_argument("--species", nargs="*", default=DEFAULT_SPECIES)
    ap.add_argument("--per-class", type=int, default=200)
    ap.add_argument("--per-location-cap", type=int, default=20)
    ap.add_argument("--store-size", type=int, default=384,
                    help="long side (px) of the stored uncropped frame")
    ap.add_argument("--cache-dir", default=".cct_cache")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--val-fraction", type=float, default=0.15)
    ap.add_argument("--test-fraction", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    build(args.out, args.species, args.per_class, args.per_location_cap,
          args.store_size, args.cache_dir, args.workers,
          args.val_fraction, args.test_fraction, args.seed)
