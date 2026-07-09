# AI Animal Image Recognition on Night-Vision Camera-Trap Images

**ACM 40960 — Project 9** · University College Dublin, Summer 2026
Srivani Konda · Navya Sri Mungamuri

Camera traps produce millions of images a year, most containing no animal, and
the ones taken at night are grayscale, low-contrast **infrared** frames that
models trained on ordinary photos struggle with. This project builds a
transfer-learning image classifier for night-vision camera-trap imagery and
measures its accuracy, precision, and recall against published baselines.

See [`docs/literature_review.md`](docs/literature_review.md) for the background
and the baselines we compare against.

## The dataset: real infrared night-vision frames

The project's target dataset is [NTLNP](https://github.com/myyyyw/NTLNP) (night
infrared frames from the Northeast Tiger & Leopard National Park), which is
hosted on Hugging Face. This repo therefore ships with a **ready-to-run dataset
of genuine infrared night-vision camera-trap images** at
[`data/night_wildlife/`](data/night_wildlife) — **6 wild species**
(bobcat, coyote, raccoon, opossum, rabbit, deer), 200 images each, all captured
at night under infrared flash (verified grayscale). They are drawn from the
**Caltech Camera Traps** dataset (Beery et al., ECCV 2018) via the
[LILA BC](https://lila.science/datasets/caltech-camera-traps) Google-Cloud
mirror, and rebuilt by [`scripts/build_night_wildlife.py`](scripts/build_night_wildlife.py).

These are real infrared frames of the exact kind the project studies — not
daytime colour photos and not synthetic images. (NTLNP itself is on Hugging
Face, which this build environment's egress policy blocks; Caltech Camera Traps
on Google Cloud Storage is reachable and is the same *class* of data — infrared
night-vision wildlife camera traps.)

## What's here

```
src/
  config.py      Single Config dataclass — every hyperparameter lives here
  data.py        Loading, stratified train/val/test split, IR-aware transforms
  model.py       Transfer-learning ResNet builder (freeze early layers, retrain later)
  train.py       Training loop: class weighting, cosine LR, early stopping, checkpointing
  evaluate.py    Accuracy / precision / recall, confusion matrix, baseline comparison
  detect.py      Optional YOLOv8 detect-and-crop preprocessing stage
  utils.py       Seeding + plotting helpers
scripts/
  build_night_wildlife.py     Build the real IR dataset from Caltech Camera Traps / LILA
  download_ntlnp.sh           Download the real NTLNP night-vision dataset (Hugging Face)
  fetch_pretrained_weights.py Fetch ImageNet weights offline (checksum-verified)
  run_training.py             Train a model
  run_evaluation.py           Score a checkpoint on the held-out test set
  predict.py                  Classify a single image
docs/
  literature_review.md     Background, prior work, and baselines
  demo_results/            Committed plots + metrics from the real IR training run
```

## Setup

```bash
pip install -r requirements.txt          # torch, torchvision, numpy, scikit-learn, matplotlib, tqdm
# optional detection stage:
pip install ultralytics opencv-python-headless
```

## Test it in one command (real infrared images, ship with the repo)

The committed `data/night_wildlife/` lets the entire transfer-learning pipeline
run on **real night-vision data** straight after `git clone`, on a plain CPU.

```bash
pip install -r requirements.txt

# (only if download.pytorch.org is blocked in your environment — otherwise skip;
#  torchvision fetches the ImageNet weights automatically on first --pretrained)
python scripts/fetch_pretrained_weights.py

# Transfer learning: ImageNet-pretrained ResNet-18, IR grayscale input,
# retraining layer3 + layer4 on the infrared frames
python scripts/run_training.py --data-dir data/night_wildlife --epochs 16 \
    --image-size 224 --pretrained --grayscale --freeze-until layer2 \
    --learning-rate 3e-4 --output-dir results/demo --device cpu

# Evaluate on the held-out test split
python scripts/run_evaluation.py --output-dir results/demo --device cpu

# Classify one infrared frame
python scripts/predict.py data/night_wildlife/raccoon/raccoon_0005.jpg \
    --checkpoint results/demo/best_model.pt
```

On this real infrared dataset the pipeline reaches **~0.77 test accuracy across
6 species** (chance = 0.17; macro precision/recall ~0.77) — in the same ballpark
as the out-of-location camera-trap baselines in the literature. Outputs —
checkpoint, `metrics.json`, training curves, confusion matrix, and the baseline
bar chart — land in `results/demo/`; a committed snapshot of the plots is in
[`docs/demo_results/`](docs/demo_results/).

### Rebuild / enlarge the infrared dataset

```bash
# pull more images per class, or different species, straight from LILA's GCS mirror
python scripts/build_night_wildlife.py --out data/night_wildlife \
    --per-class 300 --image-size 224 \
    --species bobcat coyote raccoon opossum rabbit deer skunk fox
```

The builder keeps only night captures that are verified grayscale (genuine
infrared), and de-duplicates by capture sequence so burst frames do not leak
between train / val / test.

## Running on the real NTLNP dataset

NTLNP is **25,657 infrared frames across 17 species**, hosted on Hugging Face:

```bash
bash scripts/download_ntlnp.sh          # needs git-lfs; clones from Hugging Face
```

Arrange it as one folder per species (`data/ntlnp/<species>/`) and train exactly
as above (grayscale→RGB is on by default for the infrared frames):

```bash
python scripts/run_training.py --data-dir data/ntlnp --backbone resnet18 \
    --pretrained --grayscale --freeze-until layer2 --epochs 15 --output-dir results/ntlnp
python scripts/run_evaluation.py --output-dir results/ntlnp
```

> **Egress note.** This repo was built in a sandbox whose egress policy blocks
> `huggingface.co` (NTLNP) and `download.pytorch.org` (ImageNet weights), both
> verified returning HTTP 403. The committed demo therefore sources its infrared
> images from Caltech Camera Traps on Google Cloud Storage (reachable) and its
> ImageNet weights from a checksum-verified GitHub mirror
> (`scripts/fetch_pretrained_weights.py`). On any unrestricted machine,
> `scripts/download_ntlnp.sh` fetches NTLNP and torchvision downloads the weights
> automatically.

### Optional: detect-and-crop with YOLOv8

Camera-trap frames are mostly empty background. `src/detect.py` wraps YOLOv8 to
localise the animal and crop to it before classification, which removes
background confounders. It is optional and degrades gracefully to the full frame
when no animal is detected or `ultralytics` is not installed.

## Method summary

- **Preprocessing** (`data.py`): infrared frames are single-channel; a
  grayscale→RGB step gives the 3-channel pretrained backbone a consistent input.
  Training adds random resized crops, flips, small rotations, and
  brightness/contrast jitter; evaluation uses a deterministic resize + centre
  crop. Normalisation uses ImageNet statistics.
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
preprocessing. Results in `docs/demo_results/` were produced by the commands
above.

## Data credit

Caltech Camera Traps — Beery, S., Van Horn, G. & Perona, P. "Recognition in Terra
Incognita", ECCV 2018. Distributed via LILA BC
(https://lila.science/datasets/caltech-camera-traps) under the Community Data
License Agreement. NTLNP — https://github.com/myyyyw/NTLNP.
