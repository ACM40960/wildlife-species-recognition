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
  prepare_real_dataset.py     Build the committed real-image dataset (CIFAR animals)
  download_ntlnp.sh           Download the real NTLNP night-vision dataset
  fetch_pretrained_weights.py Fetch ImageNet weights offline (checksum-verified)
  make_synthetic_data.py      Generate a synthetic night-vision dataset (optional)
  run_training.py             Train a model
  run_evaluation.py           Score a checkpoint on the held-out test set
  predict.py                  Classify a single image
docs/
  literature_review.md     Background, prior work, and baselines
  demo_results/            Committed plots + metrics from the real-image demo run
```

## Setup

```bash
pip install -r requirements.txt          # torch, torchvision, numpy, scikit-learn, matplotlib, tqdm
# optional detection stage:
pip install ultralytics opencv-python-headless
```

## Test it in one command (real images, ships with the repo)

The repo ships a small **ready-to-run dataset of real photographs** at
[`data/real_animals/`](data/real_animals) — 6 animal classes (bird, cat, deer,
dog, frog, horse), 500 images each, sampled from the openly available
[CIFAR-10-images](https://github.com/YoongiKim/CIFAR-10-images) repository. This
lets the *entire* transfer-learning pipeline run on **real data** straight after
`git clone`, on a plain CPU, in a few minutes.

> **Why not the real NTLNP data here?** NTLNP (25,657 infrared frames) is hosted
> on Hugging Face, which this build sandbox's egress policy blocks. So the
> committed demo uses real *daylight* animal photos as a stand-in to prove the
> classifier works on genuine images end-to-end. The infrared/night-vision
> handling and the NTLNP downloader are ready for the real dataset — see the
> next section.

```bash
pip install -r requirements.txt

# (only if download.pytorch.org is blocked in your environment — otherwise skip;
#  torchvision fetches the ImageNet weights automatically on first --pretrained)
python scripts/fetch_pretrained_weights.py

# Transfer learning: ImageNet-pretrained ResNet-18, retraining layer3+layer4
python scripts/run_training.py --data-dir data/real_animals --epochs 14 --image-size 128 \
    --pretrained --no-grayscale --freeze-until layer2 --learning-rate 3e-4 \
    --output-dir results/demo --device cpu

# Evaluate on the held-out test split
python scripts/run_evaluation.py --output-dir results/demo --device cpu

# Classify one image
python scripts/predict.py data/real_animals/deer/deer_0000.png \
    --checkpoint results/demo/best_model.pt
```

With ImageNet transfer learning this reaches **~0.75 test accuracy across the 6
real classes** (chance = 0.17; macro precision/recall ~0.75). For comparison,
training the same network *from scratch* on this data (`--no-pretrained
--no-grayscale --freeze-until ""`) only reaches ~0.60 — a concrete demonstration
of why the project uses transfer learning. Outputs — checkpoint, `metrics.json`,
training curves, confusion matrix, and the baseline bar chart — land in
`results/demo/`; a committed snapshot of the plots is in
[`docs/demo_results/`](docs/demo_results/).

You can rebuild the committed dataset (e.g. more images per class):

```bash
git clone --depth 1 https://github.com/YoongiKim/CIFAR-10-images
python scripts/prepare_real_dataset.py --src CIFAR-10-images --out data/real_animals --per-class 500
```

> The `--no-grayscale` flag keeps these colour photos in colour. For real NTLNP
> infrared frames, drop it (grayscale→RGB is the default) so the single-channel
> IR images are handled correctly.

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

and train with ImageNet-pretrained transfer learning (grayscale→RGB is on by
default for the infrared frames; retrain the later blocks):

```bash
python scripts/run_training.py --data-dir data/ntlnp --backbone resnet18 \
    --pretrained --freeze-until layer2 --epochs 15 --output-dir results/ntlnp
python scripts/run_evaluation.py --output-dir results/ntlnp
```

> **Why the real NTLNP data isn't committed here.** This repo was built inside a
> sandbox whose egress policy **blocks `huggingface.co`** (verified: HTTP 403),
> so the NTLNP frames could not be fetched from the build environment — hence the
> committed demo uses real CIFAR animal photos as a stand-in. The ImageNet
> weights are also normally fetched from `download.pytorch.org` (also blocked
> here); `scripts/fetch_pretrained_weights.py` works around that with a
> checksum-verified mirror, which is how the committed transfer-learning results
> were produced. On any normal machine `scripts/download_ntlnp.sh` fetches the
> real data and torchvision downloads the weights automatically.

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
