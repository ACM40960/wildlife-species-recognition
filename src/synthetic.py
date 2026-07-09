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

# Stand-in species names; the real dataset defines its own class folders.
DEFAULT_SPECIES = ["amur_leopard", "amur_tiger", "red_fox", "wild_boar", "roe_deer"]


def _draw_silhouette(draw: ImageDraw.ImageDraw, species_idx: int, size: int,
                     rng: np.random.Generator) -> None:
    """Draw a rough, species-specific bright shape on a dark background."""
    cx = rng.integers(size // 3, 2 * size // 3)
    cy = rng.integers(size // 3, 2 * size // 3)
    scale = rng.uniform(0.18, 0.30) * size
    fill = int(rng.integers(150, 230))  # animals read brighter than background in IR

    shape = species_idx % 5
    if shape == 0:  # elongated body (big cat)
        draw.ellipse([cx - scale * 1.6, cy - scale * 0.6,
                      cx + scale * 1.6, cy + scale * 0.6], fill=fill)
    elif shape == 1:  # body + prominent tail (tiger-ish)
        draw.ellipse([cx - scale * 1.4, cy - scale * 0.7,
                      cx + scale * 1.4, cy + scale * 0.7], fill=fill)
        draw.line([cx + scale * 1.4, cy, cx + scale * 2.4, cy - scale],
                  fill=fill, width=max(2, int(scale * 0.25)))
    elif shape == 2:  # compact + pointy ears (fox)
        draw.ellipse([cx - scale, cy - scale, cx + scale, cy + scale], fill=fill)
        draw.polygon([(cx - scale, cy - scale), (cx - scale * 0.5, cy - scale * 1.8),
                      (cx, cy - scale)], fill=fill)
        draw.polygon([(cx, cy - scale), (cx + scale * 0.5, cy - scale * 1.8),
                      (cx + scale, cy - scale)], fill=fill)
    elif shape == 3:  # bulky low body (boar)
        draw.rectangle([cx - scale * 1.3, cy - scale * 0.5,
                        cx + scale * 1.3, cy + scale * 0.8], fill=fill)
    else:  # tall body + legs (deer)
        draw.ellipse([cx - scale, cy - scale * 0.6, cx + scale, cy + scale * 0.4],
                     fill=fill)
        for dx in (-0.7, -0.3, 0.3, 0.7):
            draw.line([cx + scale * dx, cy + scale * 0.4,
                       cx + scale * dx, cy + scale * 1.4],
                      fill=fill, width=max(2, int(scale * 0.15)))


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
