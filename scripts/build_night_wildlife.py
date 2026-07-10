#!/usr/bin/env python3
"""Build a real night-vision (infrared) wildlife dataset from Caltech Camera Traps.

The project targets night-vision / infrared camera-trap imagery. The LILA BC
repository hosts the **Caltech Camera Traps (CCT)** dataset on Google Cloud
Storage (storage.googleapis.com), so this script assembles a genuine infrared
night-vision wildlife dataset from it.

CCT contains ~243k camera-trap frames from the U.S. Southwest with species
labels and capture timestamps. At night the traps switch to an infrared flash,
producing grayscale, low-contrast frames — precisely the imagery this project is
about. This script:

  1. downloads the CCT label metadata (COCO format) from the LILA GCS mirror,
  2. selects a set of well-represented wild species,
  3. keeps only NIGHT captures (by timestamp) that are verified to be grayscale
     (mean HSV saturation below a threshold => genuine infrared frame),
  4. de-duplicates by capture sequence so near-identical burst frames do not leak
     between train / val / test,
  5. downloads, converts to grayscale, resizes, and saves them one folder per
     species in the ImageFolder layout the pipeline expects.

Data source / credit: Caltech Camera Traps, Beery et al., "Recognition in Terra
Incognita" (ECCV 2018); distributed via https://lila.science/datasets/caltech-camera-traps .
Licensed Community Data License Agreement (permissive variant).

Usage:
    python scripts/build_night_wildlife.py --out data/night_wildlife \
        --per-class 200 --image-size 224
"""
import argparse
import io
import json
import os
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed

GCS = "https://storage.googleapis.com/public-datasets-lila"
LABELS_URL = f"{GCS}/caltechcameratraps/labels/caltech_camera_traps.json.zip"
IMAGES_BASE = f"{GCS}/caltech-unzipped/cct_images/"

# Wild species with plenty of night captures in CCT.
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
    z = zipfile.ZipFile(zpath)
    data = json.loads(z.read(z.namelist()[0]))
    cats = {c["id"]: c["name"] for c in data["categories"]}
    images = {i["id"]: i for i in data["images"]}
    img_cat = {}
    for a in data["annotations"]:
        img_cat.setdefault(a["image_id"], a["category_id"])
    return cats, images, img_cat


def candidates_for(species, cats, images, img_cat):
    """Night, sequence-deduplicated image records for one species."""
    name_to_id = {v: k for k, v in cats.items()}
    cid = name_to_id[species]
    seen_seq = set()
    out = []
    for iid, c in img_cat.items():
        if c != cid:
            continue
        rec = images.get(iid)
        if not rec or not _is_night(rec.get("date_captured", "")):
            continue
        seq = rec.get("seq_id", iid)
        if seq in seen_seq:
            continue
        seen_seq.add(seq)
        out.append(rec)
    out.sort(key=lambda r: r["id"])   # deterministic ordering
    return out


def fetch_one(rec, image_size):
    """Download, verify grayscale, convert+resize. Returns a PIL image or None."""
    from PIL import Image
    import numpy as np

    try:
        raw = urllib.request.urlopen(IMAGES_BASE + rec["file_name"], timeout=40).read()
        im = Image.open(io.BytesIO(raw))
        hsv = np.asarray(im.convert("HSV"), dtype=np.float32)
        if hsv[:, :, 1].mean() > SAT_THRESHOLD:
            return None  # colour => daytime, skip
        g = im.convert("L")
        w, h = g.size
        scale = image_size / min(w, h)
        g = g.resize((max(image_size, int(w * scale)), max(image_size, int(h * scale))))
        # centre crop to square
        w, h = g.size
        left, top = (w - image_size) // 2, (h - image_size) // 2
        return g.crop((left, top, left + image_size, top + image_size))
    except Exception:
        return None


def build(out_dir, species_list, per_class, image_size, cache_dir, workers):
    cats, images, img_cat = load_metadata(cache_dir)
    total = 0
    for species in species_list:
        cls_dir = os.path.join(out_dir, species)
        os.makedirs(cls_dir, exist_ok=True)
        cands = candidates_for(species, cats, images, img_cat)
        # Over-request candidates since some will be rejected as non-grayscale.
        budget = cands[: per_class * 3]
        saved = 0
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(fetch_one, rec, image_size): rec for rec in budget}
            for fut in as_completed(futs):
                if saved >= per_class:
                    break
                img = fut.result()
                if img is None:
                    continue
                img.save(os.path.join(cls_dir, f"{species}_{saved:04d}.jpg"), quality=88)
                saved += 1
        total += saved
        print(f"  {species:10s} {saved} infrared night images "
              f"({len(cands)} night candidates available)")
    print(f"[night-wildlife] {len(species_list)} classes, {total} images -> {out_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data/night_wildlife")
    ap.add_argument("--species", nargs="*", default=DEFAULT_SPECIES)
    ap.add_argument("--per-class", type=int, default=200)
    ap.add_argument("--image-size", type=int, default=224)
    ap.add_argument("--cache-dir", default=".cct_cache")
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    build(args.out, args.species, args.per_class, args.image_size,
          args.cache_dir, args.workers)
