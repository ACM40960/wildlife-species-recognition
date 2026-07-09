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

## Test it in one command (no download, no setup beyond `pip install`)

The repo ships a small **ready-to-run dataset** at
[`data/demo_ntlnp/`](data/demo_ntlnp) — 17 folders named for the **real NTLNP
species** (amur_tiger, amur_leopard, red_fox, …), each with 60 synthetic
night-vision frames. This lets the *entire* pipeline run straight after
`git clone`, on a plain CPU, in a few minutes.

> These demo images are **fabricated IR silhouettes**, one distinct shape per
> species — a smoke test that exercises every code path and reproduces every
> plot. They are **not** the real NTLNP photos and the numbers below are **not**
> a scientific result. To run on the real data, see the next section.

```bash
pip install -r requirements.txt

# Train (from scratch; the sandbox blocks the ImageNet-weights download — see note)
python scripts/run_training.py --data-dir data/demo_ntlnp --epochs 12 --image-size 128 \
    --no-pretrained --freeze-until "" --output-dir results/demo --device cpu

# Evaluate on the held-out test split
python scripts/run_evaluation.py --output-dir results/demo --device cpu

# Classify one image
python scripts/predict.py data/demo_ntlnp/red_fox/red_fox_0000.jpg \
    --checkpoint results/demo/best_model.pt
```

On the committed demo data this reaches **~0.86 test accuracy across all 17
classes** (macro precision ~0.90 / recall ~0.86). Outputs — checkpoint,
`metrics.json`, training curves, confusion matrix, and the baseline bar chart —
land in `results/demo/`; a committed snapshot of the plots is in
[`docs/demo_results/`](docs/demo_results/).

You can regenerate the demo dataset with different size/seed:

```bash
python scripts/make_synthetic_data.py --out data/demo_ntlnp --per-class 60 --image-size 128
```

## Running on the real NTLNP dataset

The real dataset is **25,657 infrared camera-trap images across 17 species**,
hosted on Hugging Face. A downloader is included:

```bash
bash scripts/download_ntlnp.sh          # needs git-lfs; clones from Hugging Face
```

Then arrange the frames as one folder per species (ImageFolder layout):

```
data/ntlnp/
  amur_tiger/    img001.jpg ...
  amur_leopard/  ...
  red_fox/       ...
```

and train with ImageNet-pretrained transfer learning (retraining the last block):

```bash
python scripts/run_training.py --data-dir data/ntlnp --backbone resnet18 \
    --pretrained --freeze-until layer4 --epochs 15 --output-dir results/ntlnp
python scripts/run_evaluation.py --output-dir results/ntlnp
```

> **Why the real data isn't committed here.** This repo was built inside a
> sandbox whose egress policy **blocks `huggingface.co` and
> `download.pytorch.org`** (verified: both return HTTP 403). So neither the real
> dataset nor the ImageNet weights could be fetched from the build environment —
> that is why the committed demo uses a synthetic dataset and `--no-pretrained`.
> On any normal machine `scripts/download_ntlnp.sh` fetches the real data, and
> dropping `--no-pretrained` gives the transfer-learning setup the project is
> really about, which drives the accuracies reported in the literature.

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
