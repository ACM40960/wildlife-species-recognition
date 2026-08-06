# Experiments — run log

A running table of training runs so results are traceable and easy to pull into
the poster/report later. Add a row per run.

**Fixed unless noted:** dataset = `data/night_wildlife` (6 species, 200
img/species, infrared night frames); ResNet-18; grayscale→RGB input; image size
224; batch size 32; AdamW lr 3e-4, weight decay 1e-4; cosine schedule; label
smoothing 0.05; seed 42. "Split" = how train/val/test is partitioned.

> **Dataset versions.** v1 = whole-frame centre-crop, random split. v2 = bbox crop
> baked into files. v3 = deterministic + location/time-stratified sampling,
> single-species, frames stored **uncropped** with the crop applied at load time.
> **v5 (current)** = v3 plus letterbox padding, infrared augmentation, and
> MegaDetector boxes (86% coverage). Only compare rows within the same version.

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
| 2026-07-21 | v3 · crop=detected · 16 ep | location · unseen | 0.549 | 0.548 | Previous headline (superseded by the v5 row below). 95% CI 0.485–0.612; top-2 0.69; ECE 0.15. |
| 2026-07-21 | v3 · crop=detected · 16 ep | location · **seen** | 0.718 | — | Same model, seen-location holdout. Seen−unseen gap **+0.17**. |
| 2026-07-21 | v3 · crop=**full frame** · 16 ep | location · unseen | 0.459 | 0.455 | Full-frame input. Detected-animal beats it by ~0.09; ECE 0.22. |
| 2026-07-21 | v3 · crop=detected + **YOLO-filled boxes** (66%) | location · unseen | 0.545 | 0.542 | YOLO raised box coverage 50%→66%; accuracy unchanged within CI. |
| 2026-08-05 | v4 · **letterbox + IR augmentation** (66% boxes) | location · unseen | 0.614 | 0.617 | Padding instead of centre-crop stops cutting animals off; gamma/erasing aug. |
| 2026-08-05 | **v5 · + MegaDetector boxes (86%)** | **location · unseen** | **0.687** | 0.682 | **Current headline.** 95% CI 0.625–0.743; top-2 0.83; ECE 0.048 (T=1.07). |
| 2026-08-05 | v5 · same model | location · **seen** | 0.800 | — | Seen−unseen gap narrowed to **+0.11** (was +0.17). |
| 2026-08-05 | v5 · **with TTA** (mirror averaging) | location · unseen | 0.682 | 0.677 | Negative result: TTA costs 2x inference and is slightly worse (ECE 0.072 vs 0.048). Disabled by default. |

Key results:

- **Seen vs. unseen** (one model): 0.80 on seen camera sites vs **0.69** on unseen
  sites — a +0.11 generalisation gap that persists even with the animal cropped.
- **What moved the number** from 0.55 to 0.69: letterbox padding + infrared
  augmentation (~+0.07) and MegaDetector boxes raising coverage 66%→86% (~+0.07).
- **What did not:** TTA measured slightly *worse* (0.687 → 0.682) and is off by
  default. Temperature scaling cannot change accuracy by construction (monotonic
  rescaling); it improved calibration only, ECE 0.054 → 0.048. The bulk of the ECE
  gain (0.150 → 0.054) came from the better model, not the calibration step.
- **Detected animal vs. full frame** (same split): cropping to the bounding box
  lifts unseen accuracy 0.46 → **0.69** and cuts calibration error sharply
  (0.22 → 0.05).
- Every run writes `history.csv`/`history.json`, `environment.json`, and
  `error_analysis.png`; metrics carry 95% confidence intervals (test set is small).

## Planned runs

- Linear probe (`--freeze-until all`) — how much do the frozen ImageNet features
  alone get us on the location-held-out split?
- Sequence-level aggregation using the stored `seq_id` (would change the task to
  classifying a burst, so it must be reported separately).
- More images/species and more species, to test whether 0.69 improves with scale.

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
