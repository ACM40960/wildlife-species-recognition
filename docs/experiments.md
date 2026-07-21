# Experiments — run log

A running table of training runs so results are traceable and easy to pull into
the poster/report later. Add a row per run.

**Fixed unless noted:** dataset = `data/night_wildlife` (6 species, 200
img/species, infrared night frames); ResNet-18; grayscale→RGB input; image size
224; batch size 32; AdamW lr 3e-4, weight decay 1e-4; cosine schedule; label
smoothing 0.05; seed 42. "Split" = how train/val/test is partitioned.

| Date | Config | Split | Test acc | Macro F1 | Notes |
|------|--------|-------|----------|----------|-------|
| 2026-07-21 | pretrained (ImageNet), freeze `layer2`, 16 ep | stratified 70/15/15 | **0.772** | 0.771 | Main result. Deer best (F1 0.90), bobcat worst (0.70). |
| 2026-07-21 | **from scratch** (no pretrained), train all layers, 16 ep | stratified 70/15/15 | 0.622 | 0.617 | Ablation: same setup without ImageNet init. Transfer learning adds **+0.15** accuracy. |

## Planned runs

- Linear probe (`--freeze-until all`) — how much do the frozen ImageNet features
  alone get us?
- Location-held-out split (Issue #1) — expected to drop accuracy; the gap vs. the
  stratified split is the headline generalisation number.
- YOLOv8 detect-and-crop vs. full frame (Issue #2).

## How to reproduce a row

```bash
# main result
python scripts/run_training.py --data-dir data/night_wildlife --epochs 16 \
    --image-size 224 --pretrained --grayscale --freeze-until layer2 \
    --learning-rate 3e-4 --output-dir results/demo --device cpu
python scripts/run_evaluation.py --output-dir results/demo --device cpu

# from-scratch ablation
python scripts/run_training.py --data-dir data/night_wildlife --epochs 16 \
    --image-size 224 --no-pretrained --freeze-until "" \
    --learning-rate 3e-4 --output-dir results/scratch --device cpu
python scripts/run_evaluation.py --output-dir results/scratch --device cpu
```
