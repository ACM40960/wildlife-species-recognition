# Experiments — run log

A running table of training runs so results are traceable and easy to pull into
the poster/report later. Add a row per run.

**Fixed unless noted:** dataset = `data/night_wildlife` (6 species, 200
img/species, infrared night frames); ResNet-18; grayscale→RGB input; image size
224; batch size 32; AdamW lr 3e-4, weight decay 1e-4; cosine schedule; label
smoothing 0.05; seed 42. "Split" = how train/val/test is partitioned.

> **Dataset v2 (2026-07-21):** rebuilt with bounding-box cropping (letterbox
> fallback), images spread across camera sites, and a manifest with a
> location-grouped split. Rows below the divider use v2 and are the current
> results. The two rows above the divider used dataset **v1** (whole-frame
> centre-crop, random split) and are kept only for history — not comparable to v2.

| Date | Config | Split | Test acc | Macro F1 | Notes |
|------|--------|-------|----------|----------|-------|
| 2026-07-21 | v1 · pretrained, freeze `layer2`, 16 ep | stratified 70/15/15 | 0.772 | 0.771 | v1 dataset (centre-crop). Superseded. |
| 2026-07-21 | v1 · from scratch, train all, 16 ep | stratified 70/15/15 | 0.622 | 0.617 | v1 ablation. Transfer learning +0.15. Superseded. |
| — | — | — | — | — | — |
| 2026-07-21 | **v2 · pretrained, freeze `layer2`, 16 ep** | **location-held-out** | **0.368** | 0.351 | **Headline.** Honest generalisation to unseen sites. |
| 2026-07-21 | v2 · pretrained, freeze `layer2`, 16 ep | stratified (same-location) | 0.733 | 0.729 | Same data, random split. The 0.73→0.37 gap = background reliance. |

The key v2 result: holding out camera locations drops accuracy from **0.73 to
0.37** on identical data and settings — most of the same-location score was the
model recognising backgrounds, not animals.

## Planned runs

- Linear probe (`--freeze-until all`) — how much do the frozen ImageNet features
  alone get us on the location-held-out split?
- YOLOv8 detect-and-crop vs. full frame (Issue #2) — does detector cropping close
  part of the location-held-out gap?
- More images/species and more species, to test whether the 0.37 improves with
  scale.

## How to reproduce a row

```bash
# main result
python scripts/run_training.py --data-dir data/night_wildlife --epochs 16 \
    --image-size 224 --pretrained --grayscale --freeze-until layer2 \
    --learning-rate 3e-4 --output-dir results/demo --device cpu
python scripts/run_evaluation.py --output-dir results/demo --device cpu

# same-location comparison (stratified split on the same data)
python scripts/run_training.py --data-dir data/night_wildlife --epochs 16 \
    --image-size 224 --pretrained --grayscale --freeze-until layer2 \
    --learning-rate 3e-4 --split-by stratified \
    --output-dir results/samelocation --device cpu
python scripts/run_evaluation.py --output-dir results/samelocation --device cpu
```
