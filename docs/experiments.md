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
| 2026-07-21 | v3 (pre model-fix) · freeze `layer2` | location-held-out | 0.506 | 0.505 | Before the frozen-BatchNorm fix. Superseded. |
| 2026-07-21 | v3 (pre model-fix) · freeze `layer2` | stratified | 0.611 | 0.608 | Superseded. |
| 2026-07-21 | v3 + model fixes · freeze `layer2` | location-held-out | 0.554 | 0.552 | Before the seen-location holdout was carved. |
| — | — | — | — | — | — |
| 2026-07-21 | **v3 · crop=detected · 16 ep** | **location · unseen** | **0.549** | 0.548 | **Current headline.** 95% CI 0.485–0.612; top-2 0.68; ECE 0.14. |
| 2026-07-21 | v3 · crop=detected · 16 ep | location · **seen** | 0.755 | — | Same model, seen-location holdout. Seen−unseen gap **+0.21**. |
| 2026-07-21 | v3 · crop=**full frame** · 16 ep | location · unseen | 0.459 | 0.455 | Full-frame input. Detected-animal beats it by ~0.09; ECE 0.22. |
| 2026-07-21 | v3 · crop=detected + **YOLO-filled boxes** (66%) | location · unseen | 0.545 | 0.542 | YOLO raised box coverage 50%→66%; accuracy unchanged within CI. |

Key results:

- **Seen vs. unseen** (one model): 0.76 on seen camera sites vs **0.55** on unseen
  sites — a +0.21 generalisation gap that persists even with the animal cropped.
- **Detected animal vs. full frame** (same split): cropping to the bounding box
  lifts unseen accuracy 0.46 → **0.55** and roughly halves calibration error.
- Every run writes `history.csv`/`history.json`, `environment.json`, and
  `error_analysis.png`; metrics carry 95% confidence intervals (test set is small).

## Planned runs

- Linear probe (`--freeze-until all`) — how much do the frozen ImageNet features
  alone get us on the location-held-out split?
- Wire a YOLO detector to supply boxes for the ~50% of frames without one (Issue #2).
- More images/species and more species, to test whether 0.55 improves with scale.

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
