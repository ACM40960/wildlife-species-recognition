# AI Animal Image Recognition on Night-Vision Camera-Trap Images

**ACM 40960 — Project 9** · University College Dublin, Summer 2026
Srivani Konda · Navya Sri Mungamuri

Camera traps produce millions of images a year, most containing no animal, and
the ones taken at night are grayscale, low-contrast infrared frames that models
trained on ordinary photos struggle with. This project builds a transfer-learning
image classifier for night-vision camera-trap imagery (the
[NTLNP dataset](https://github.com/myyyyw/NTLNP)) and measures its accuracy,
precision, and recall against published baselines.

See [`docs/literature_review.md`](docs/literature_review.md) for the background
and the baselines we compare against.

## What's here

```
src/
  config.py      Single Config dataclass — every hyperparameter lives here
  data.py        NTLNP loading, stratified train/val/test split, IR-aware transforms
  model.py       Transfer-learning ResNet builder (freeze early layers, retrain later)
  train.py       Training loop: class weighting, cosine LR, early stopping, checkpointing
  evaluate.py    Accuracy / precision / recall, confusion matrix, baseline comparison
  detect.py      Optional YOLOv8 detect-and-crop preprocessing stage
  synthetic.py   Synthetic night-vision data generator (so the pipeline runs anywhere)
  utils.py       Seeding + plotting helpers
scripts/
  make_synthetic_data.py   Fabricate a demo dataset
  run_training.py          Train a model
  run_evaluation.py        Score a checkpoint on the held-out test set
  predict.py               Classify a single image
docs/
  literature_review.md     Background, prior work, and baselines
  demo_results/            Committed plots + metrics from the synthetic-data demo
```

## Setup

```bash
pip install -r requirements.txt          # torch, torchvision, numpy, scikit-learn, matplotlib, tqdm
# optional detection stage:
pip install ultralytics opencv-python-headless
```

## Quick start (runs anywhere, no dataset download needed)

The full pipeline can be exercised on a synthetic night-vision dataset. This is a
**smoke test / demo**, not a scientific result — the images are fabricated
silhouettes designed only to make every code path runnable and every plot
reproducible on a plain CPU.

```bash
# 1. Make a small synthetic dataset (5 fake species, ImageFolder layout)
python scripts/make_synthetic_data.py --out data/synthetic --per-class 100 --image-size 128

# 2. Train (from scratch here; use --pretrained on a machine with internet)
python scripts/run_training.py --data-dir data/synthetic --epochs 8 --image-size 128 \
    --no-pretrained --freeze-until "" --output-dir results/demo --device cpu

# 3. Evaluate on the held-out test split
python scripts/run_evaluation.py --output-dir results/demo --device cpu

# 4. Classify one image
python scripts/predict.py data/synthetic/red_fox/red_fox_0000.jpg \
    --checkpoint results/demo/best_model.pt
```

Outputs (checkpoint, `metrics.json`, training curves, confusion matrix, baseline
bar chart) land in `results/demo/`. A committed copy of the plots from one such
run is in [`docs/demo_results/`](docs/demo_results/).

## Running on the real NTLNP dataset

1. Download NTLNP from https://github.com/myyyyw/NTLNP.
2. Arrange it as one folder per species (ImageFolder layout):

   ```
   data/ntlnp/
     amur_leopard/  img001.jpg ...
     amur_tiger/    ...
     red_fox/       ...
   ```
3. Train with ImageNet-pretrained transfer learning (retraining the last block):

   ```bash
   python scripts/run_training.py --data-dir data/ntlnp --backbone resnet18 \
       --pretrained --freeze-until layer4 --epochs 15 --output-dir results/ntlnp
   python scripts/run_evaluation.py --output-dir results/ntlnp
   ```

   > **Note on this sandbox:** downloading ImageNet weights requires network
   > access to `download.pytorch.org`, which is blocked by the egress policy of
   > the environment this repo was built in. That is why the demo above uses
   > `--no-pretrained`. On any normal machine, drop `--no-pretrained` (or pass
   > `--pretrained`) to get the transfer-learning setup the project is really
   > about, which is what drives the accuracies reported in the literature.

### Optional: detect-and-crop with YOLOv8

Camera-trap frames are mostly empty background. `src/detect.py` wraps YOLOv8 to
localise the animal and crop to it before classification, which removes
background confounders (see the literature review). It is optional and degrades
gracefully to the full frame when no animal is detected or `ultralytics` is not
installed.

## Method summary

- **Preprocessing** (`data.py`): infrared frames are single-channel; a
  grayscale→RGB step gives the 3-channel pretrained backbone a consistent input
  whether a frame was captured in colour or IR. Training adds random resized
  crops, flips, small rotations, and brightness/contrast jitter; evaluation uses
  a deterministic resize + centre crop. Normalisation uses ImageNet statistics.
- **Split** (`data.py`): stratified and deterministic given the seed, so no image
  leaks between train / val / test.
- **Model** (`model.py`): ImageNet-pretrained ResNet with a fresh head; early
  blocks frozen, later block(s) retrained — the transfer-learning recipe from
  Tabak et al. (2019).
- **Training** (`train.py`): inverse-frequency class weighting for imbalance,
  label smoothing, AdamW + cosine schedule, early stopping on validation
  accuracy, best-checkpoint saving.
- **Evaluation** (`evaluate.py`): accuracy, macro & weighted precision/recall/F1,
  a per-class report, a row-normalised confusion matrix, and a bar chart against
  the Norouzzadeh (2018) and Schneider (2020) baselines.

## Reproducibility

Every run is seeded (`--seed`, default 42) and the exact `Config` is written to
`config.json` next to the checkpoint, so evaluation reproduces the same split and
preprocessing. Results in `docs/demo_results/` were produced by the Quick-start
commands above.
