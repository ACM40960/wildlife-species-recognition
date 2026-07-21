# Experiments — run log

A running table of training runs so results are traceable and easy to pull into
the poster/report later. Add a row per run.

**Fixed unless noted:** dataset = `data/night_wildlife` (6 species, 200
img/species, infrared night frames); ResNet-18; grayscale→RGB input; image size
224; batch size 32; AdamW lr 3e-4, weight decay 1e-4; cosine schedule; label
smoothing 0.05; seed 42. "Split" = how train/val/test is partitioned.

> **Dataset versions.** v1 = whole-frame centre-crop, random split (superseded).
> v2 = bbox crop baked into files, location split. **v3 (current)** = deterministic
> + location/time-stratified sampling, single-species only, frames stored
> **uncropped** with the crop applied at load time, ~50% with bounding boxes, plus
> a validated manifest. Only compare rows within the same version.

| Date | Config | Split | Test acc | Macro F1 | Notes |
|------|--------|-------|----------|----------|-------|
| 2026-07-21 | v1 · pretrained, freeze `layer2` | stratified | 0.772 | 0.771 | v1 (centre-crop). Superseded. |
| 2026-07-21 | v1 · from scratch | stratified | 0.622 | 0.617 | v1 ablation (transfer learning +0.15). Superseded. |
| 2026-07-21 | v2 · pretrained, freeze `layer2` | location-held-out | 0.368 | 0.351 | v2. Superseded by v3. |
| 2026-07-21 | v2 · pretrained, freeze `layer2` | stratified | 0.733 | 0.729 | v2 same-location. Superseded. |
| — | — | — | — | — | — |
| 2026-07-21 | **v3 · pretrained, freeze `layer2`, 16 ep** | **location-held-out** | **0.506** | 0.505 | **Current headline.** Honest generalisation to unseen sites. |
| 2026-07-21 | v3 · pretrained, freeze `layer2`, 16 ep | stratified (same-location) | 0.611 | 0.608 | Same data, random split. |

Key v3 result: with the animal cropped at load time, the same-location advantage
is only **0.61 vs 0.51** (gap 0.10, down from v2's 0.36). Cropping to the animal
removed most of the background the model was exploiting, so it both generalises
better to new sites (0.37 → **0.51**) and leaks less from same-location backgrounds.

## Planned runs

- Linear probe (`--freeze-until all`) — how much do the frozen ImageNet features
  alone get us on the location-held-out split?
- YOLOv8 detect-and-crop vs. manifest-bbox vs. full frame (Issue #2).
- More images/species and more species, to test whether 0.51 improves with scale.

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
