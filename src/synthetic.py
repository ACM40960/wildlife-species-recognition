"""Synthetic night-vision camera-trap dataset generator.

The real NTLNP dataset is large and must be downloaded separately (see the
README). To let anyone run the *entire* pipeline end-to-end — and to give CI a
fast, deterministic smoke test — this module fabricates a small dataset that
mimics the salient properties of night-vision camera-trap imagery:

  * grayscale / low-contrast "infrared" frames,
  * a coarse animal-like silhouette whose shape depends on the species,
  * sensor noise and a vignette, as real IR traps produce.

It is emphatically NOT a substitute for real data or a source of scientific
results; it exists so the code paths (loading, augmentation, transfer learning,
evaluation, plotting) can be exercised and demonstrated. Each species gets a
distinct silhouette so a CNN can actually learn to separate them.
"""
from __future__ import annotations

import os
from typing import List

import numpy as np
from PIL import Image, ImageDraw

# The real 17 NTLNP classes (Northeast Tiger & Leopard National Park).
# Using the real taxonomy so the demo dataset mirrors the actual folder layout.
DEFAULT_SPECIES = [
    "amur_tiger", "amur_leopard", "wild_boar", "roe_deer", "sika_deer",
    "asian_black_bear", "red_fox", "asian_badger", "raccoon_dog", "musk_deer",
    "siberian_weasel", "sable", "yellow_throated_marten", "leopard_cat",
    "manchurian_hare", "cow", "dog",
]


def _class_attributes(species_idx: int) -> dict:
    """Deterministic, per-class silhouette attributes.

    Each class gets a distinct combination of body shape, aspect ratio,
    brightness band, and appendages (ears / tail / legs) so that all 17 classes
    are visually separable — otherwise a classifier could not learn them apart.
    Variation *within* a class is added later from the RNG.
    """
    body_shapes = ["ellipse", "rounded_rect", "rect"]
    return {
        "body": body_shapes[species_idx % 3],
        "aspect": 1.1 + 0.22 * (species_idx % 5),      # length/height ratio
        "bright": 140 + 9 * (species_idx % 10),        # base IR brightness band
        "ears": (species_idx % 2 == 0),                # pointy ears
        "tail": (species_idx % 3 == 0),                # long tail
        "legs": [0, 2, 4][species_idx % 3],            # number of visible legs
        "size": 0.16 + 0.015 * (species_idx % 6),      # relative body size
    }


def _draw_silhouette(draw: ImageDraw.ImageDraw, species_idx: int, size: int,
                     rng: np.random.Generator) -> None:
    """Draw a rough, class-distinct bright shape on a dark background."""
    attr = _class_attributes(species_idx)
    cx = rng.integers(int(size * 0.38), int(size * 0.62))
    cy = rng.integers(int(size * 0.40), int(size * 0.60))
    scale = (attr["size"] + rng.uniform(-0.02, 0.02)) * size
    fill = int(np.clip(attr["bright"] + rng.integers(-15, 15), 90, 235))
    aspect = attr["aspect"]

    hw, hh = scale * aspect, scale  # half-width, half-height
    box = [cx - hw, cy - hh, cx + hw, cy + hh]
    if attr["body"] == "ellipse":
        draw.ellipse(box, fill=fill)
    elif attr["body"] == "rounded_rect":
        draw.rounded_rectangle(box, radius=int(scale * 0.5), fill=fill)
    else:
        draw.rectangle(box, fill=fill)

    if attr["tail"]:
        draw.line([cx + hw, cy, cx + hw + scale * 1.1, cy - scale * 0.9],
                  fill=fill, width=max(2, int(scale * 0.22)))
    if attr["ears"]:
        draw.polygon([(cx - hw * 0.7, cy - hh), (cx - hw * 0.4, cy - hh - scale * 0.9),
                      (cx - hw * 0.1, cy - hh)], fill=fill)
        draw.polygon([(cx + hw * 0.1, cy - hh), (cx + hw * 0.4, cy - hh - scale * 0.9),
                      (cx + hw * 0.7, cy - hh)], fill=fill)
    if attr["legs"]:
        step = (2 * hw) / (attr["legs"] + 1)
        for i in range(1, attr["legs"] + 1):
            lx = cx - hw + step * i
            draw.line([lx, cy + hh, lx, cy + hh + scale * 0.9],
                      fill=fill, width=max(2, int(scale * 0.13)))


def _make_image(species_idx: int, size: int, rng: np.random.Generator) -> Image.Image:
    base = int(rng.integers(20, 55))  # dark IR background
    img = Image.new("L", (size, size), color=base)
    draw = ImageDraw.Draw(img)
    _draw_silhouette(draw, species_idx, size, rng)

    arr = np.asarray(img, dtype=np.float32)
    # Sensor noise.
    arr += rng.normal(0, 12, arr.shape)
    # Radial vignette (traps darken toward the edges).
    yy, xx = np.mgrid[0:size, 0:size]
    r = np.sqrt((xx - size / 2) ** 2 + (yy - size / 2) ** 2) / (size / 2)
    arr *= np.clip(1.15 - 0.5 * r, 0.4, 1.0)
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, mode="L").convert("RGB")


def generate(out_dir: str, species: List[str] | None = None,
             per_class: int = 120, image_size: int = 128, seed: int = 0) -> str:
    """Generate a synthetic dataset under ``out_dir`` (ImageFolder layout)."""
    species = species or DEFAULT_SPECIES
    rng = np.random.default_rng(seed)
    for c, name in enumerate(species):
        cls_dir = os.path.join(out_dir, name)
        os.makedirs(cls_dir, exist_ok=True)
        for i in range(per_class):
            _make_image(c, image_size, rng).save(os.path.join(cls_dir, f"{name}_{i:04d}.jpg"))
    print(f"[synthetic] wrote {len(species)} classes x {per_class} images -> {out_dir}")
    return out_dir


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Generate a synthetic night-vision dataset")
    ap.add_argument("--out", default="data/synthetic")
    ap.add_argument("--per-class", type=int, default=120)
    ap.add_argument("--image-size", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    generate(args.out, per_class=args.per_class, image_size=args.image_size, seed=args.seed)
