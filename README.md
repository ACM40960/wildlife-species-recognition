# AI Animal Image Recognition on Night-Vision Camera-Trap Images

**ACM 40960 — Project 9**
Srivani Konda and Navya Sri Mungamuri — University College Dublin, Summer 2026

Camera traps take millions of photos a year, and most of them are empty or taken
at night. The night ones are grayscale infrared images with low contrast, which
are hard for models trained on normal daytime photos. In this project we train a
model to recognise animal species in night-vision (infrared) camera-trap images
and check how well it does using accuracy, precision and recall.

The background reading and the baselines we compare against are in
[`docs/literature_review.md`](docs/literature_review.md).

## The dataset

We use real infrared night-vision camera-trap images from the **Caltech Camera
Traps** dataset (Beery et al., 2018), downloaded through the
[LILA BC](https://lila.science/datasets/caltech-camera-traps) Google Cloud mirror.
A ready-made subset is included in the repo at
[`data/night_wildlife/`](data/night_wildlife):

- 6 species: bobcat, coyote, raccoon, opossum, rabbit, deer
- 200 images per species (1,200 total)
- every image is a genuine night infrared frame (checked to be grayscale)

`scripts/build_night_wildlife.py` builds this subset. It:

- keeps only night captures that are actually grayscale (real infrared),
  single-species, and de-duplicated by capture sequence;
- samples **deterministically** and **stratified across camera locations and
  time** (round-robin over sites, spread across each site's date range) to reduce
  selection bias — the committed build spans 35–75 locations and 18–36 months per
  species;
- stores the frame **uncropped** (only downscaled) and records the animal's
  bounding box in the manifest; the crop to the animal is applied at *load* time
  (`crop_to_bbox`, see `src/data.py`), so nothing is baked into the files;
- reports every rejected/failed download with its id and reason
  (`build_report.txt`) and wipes the output directory first so a re-run can't
  leave stale files.

**Bounding boxes.** 66% of frames carry a ground-truth box from CCT;
**MegaDetector** — the standard camera-trap detector, trained on infrared frames
(`scripts/fill_boxes_yolo.py`) — fills in boxes for frames that lack one, raising
coverage to **86%** (798 ground-truth + 228 MegaDetector). Each row's `box_source`
column records `gt` / `megadetector` / `none`.

Every image is recorded in
[`data/night_wildlife/manifest.csv`](data/night_wildlife/manifest.csv) with its
source id, original filename, class, camera location, sequence id, timestamp,
month/season, bounding box, `box_source`, split, and SHA-256 checksum.
`python scripts/validate_dataset.py` checks class balance, file integrity,
split/location overlap, and manifest↔file consistency before training.

**Split.** Frames from the same camera share backgrounds, so the split is
**location-held-out**: whole camera sites go to a single split (train / val /
test), and no background is shared between them. This is what makes the reported
accuracy a measure of animal recognition rather than background recognition. See
[`docs/METHODOLOGY.md`](docs/METHODOLOGY.md) for details.

## How to run it

You need Python 3.9+ and the packages in `requirements.txt`.

```bash
pip install -r requirements.txt
# optional: `pip install -e .` installs the project so `import src...` works from
# anywhere (the scripts also run directly from the repo without it).
```

**1. Train.** ImageNet-pretrained ResNet-18, infrared (grayscale) input,
retraining the later layers on the night images:

```bash
python scripts/run_training.py --data-dir data/night_wildlife --epochs 16 \
    --image-size 224 --pretrained --grayscale --freeze-until layer2 \
    --learning-rate 3e-4 --output-dir results/demo --device cpu
```

(If your machine can't download the pretrained weights automatically, run
`python scripts/fetch_pretrained_weights.py` first.)

> **Running in Codespaces / Docker?** Data loading runs in-process by default
> (`num_workers=0`) because containers usually mount a tiny `/dev/shm` (~64MB),
> and DataLoader workers pass batches through it — which fails with
> `RuntimeError: unable to allocate shared memory ... No space left on device`.
> If your `/dev/shm` is large (`df -h /dev/shm`), speed loading up with
> `--num-workers 2`.

**2. Evaluate** on the held-out test set. This prints the scores and saves the
plots:

```bash
python scripts/run_evaluation.py --output-dir results/demo --device cpu
```

**3. Predict** on a single image:

```bash
python scripts/predict.py data/night_wildlife/bobcat/bobcat_0032.jpg \
    --checkpoint results/demo/best_model.pt
```

```
Predictions for data/night_wildlife/bobcat/bobcat_0032.jpg  [input: dataset manifest box, T=1.07]
  1. bobcat               0.843
  2. coyote               0.127
  3. raccoon              0.015
```

Prediction applies the **same animal crop the model was trained on**: the box comes
from the manifest when the image is part of a built dataset. For an image from
elsewhere, add `--detect` to locate the animal with YOLOv8, or `--no-crop` to
classify the whole frame. Confidences use the same temperature fitted during
evaluation, so they mean what the reported ECE says (`--no-calibration` shows the
raw softmax). The bracketed note says which input and calibration were used.

This is one example — at ~0.69 accuracy the model still gets individual images
wrong, so don't read a single prediction as representative.

Everything (checkpoint, `metrics.json`, and the plots) is written to
`results/demo/`. A saved copy of the plots is in
[`docs/demo_results/`](docs/demo_results/).

## Results

Trained on the 1,200 infrared images above, 16 epochs on CPU. The test set is
small (233 images), so every number is reported with a 95% confidence interval.

**Seen vs. unseen camera locations** (same model, evaluated on both):

| Locations | What it measures | Accuracy (95% CI) |
|-----------|------------------|-------------------|
| **Unseen** (held-out sites) | generalisation to **new cameras** — the honest number | **0.69** (0.63–0.74) |
| Seen (held-out images from training sites) | performance on familiar backgrounds | 0.80 |

The **+0.11** gap between seen and unseen locations is the key result: even with
the animal cropped out of the frame, a model still does noticeably better on
cameras it has seen. Random guessing with 6 classes is 0.17.

Other metrics on the unseen-location test set:

| Metric | Value |
|--------|-------|
| Balanced accuracy | 0.68 |
| Macro-F1 (95% CI) | 0.68 (0.62–0.74) |
| Top-2 / Top-3 accuracy | 0.83 / 0.87 |
| Expected calibration error | 0.05 (temperature-scaled, T=1.07) |

**Detected-animal vs. full-frame** (issue: does cropping to the animal help?),
same location split:

| Input | Box coverage | Unseen acc | ECE |
|-------|-------------|-----------|-----|
| **Detected animal** (crop to box) | 86% (GT + MegaDetector) | **0.69** | 0.05 |
| Full frame | — | 0.46 | 0.22 |

Cropping to the animal is what makes this work: showing the classifier the whole
frame costs ~20 points. Two changes took the honest number from 0.55 to **0.69**
(see [`docs/experiments.md`](docs/experiments.md) for the run log):

| Change | Unseen acc | ECE |
|--------|-----------|-----|
| Baseline (centre-crop, COCO-YOLO boxes at 66%) | 0.55 | 0.15 |
| + letterbox padding & infrared augmentation | 0.61 | — |
| + MegaDetector boxes (86% coverage) | **0.69** | 0.05 |

Two things we tried that did **not** help accuracy, reported for honesty:
**test-time augmentation** (mirror averaging) measured slightly *worse* on this
data (0.687 → 0.682) and is off by default; **temperature scaling** cannot change
accuracy at all — it is a monotonic rescaling — and improves only calibration
(ECE 0.054 → 0.048). Most of the calibration gain came from the better model
itself, not from the calibration step.

> The full-frame row is from a separate run under the same location split; its
> `results/samelocation` counterpart predates the current metrics schema, so treat
> the 0.64 same-location figure as indicative rather than exactly comparable.

Per-species, unseen-location test (with 95% CI on recall):

| Species  | Precision | Recall | F1   | Test images |
|----------|-----------|--------|------|-------------|
| rabbit   | 0.95      | 0.75   | 0.84 | 51 |
| deer     | 0.71      | 0.75   | 0.73 | 36 |
| coyote   | 0.59      | 0.71   | 0.64 | 38 |
| raccoon  | 0.70      | 0.58   | 0.63 | 33 |
| bobcat   | 0.59      | 0.66   | 0.62 | 41 |
| opossum  | 0.58      | 0.62   | 0.60 | 34 |

![training curves](docs/demo_results/training_curves.png)
![confusion matrix](docs/demo_results/confusion_matrix.png)
![error analysis](docs/demo_results/error_analysis.png)

`scripts/run_evaluation.py` writes all of the above to `metrics.json` plus the
confusion matrix and an `error_analysis.png` montage of the most-confident correct
predictions and errors.

## Known limitations

- The 0.69 is from a **small** dataset (200 images/species) with only six species;
  behaviour on rare species and at larger scale is untested.
- Bounding boxes cover 86% of the frames (ground-truth + MegaDetector); the
  remaining 14% are classified from the whole (letterbox-padded) frame, so those
  images still contain background.
- MegaDetector still misses 174 frames; a detector fine-tuned on this park's
  imagery would push coverage higher.

## Making the dataset bigger

You can pull more images per class, or add more species, straight from the mirror:

```bash
python scripts/build_night_wildlife.py --out data/night_wildlife \
    --per-class 300 --species bobcat coyote raccoon opossum rabbit deer skunk fox
```

## Reproducing the reported numbers

**Split protocol (read this before quoting a number).** Results are reported on a
**location-held-out** split: whole camera sites are assigned to train / val / test,
so test images come from cameras never seen in training. The **0.55** headline is
this unseen-location accuracy. A same-location (random) split scores higher (0.64)
but shares backgrounds with training and is reported only for contrast. **We do
not claim the model generalises broadly** — it is trained on 6 species and 1,200
low-resolution frames, and 0.55 is the measured accuracy on *these* unseen
Caltech Camera Traps sites, not a claim about other datasets, regions, or species.

> **Caveat on the 0.55 vs 0.64 gap.** The location-held-out run also holds out a
> *seen-location* slice from training (`seen_test_fraction`, for the seen-vs-unseen
> comparison), so it trains on fewer images (~625) than the same-location run
> (~840). Part of the 0.55→0.64 difference is therefore this training-set-size
> difference, not only the split. The **seen-vs-unseen** comparison below is the
> clean one, because it uses a single model evaluated on both.

Exact commands (CPU; seed 42; each writes `metrics.json` + plots):

```bash
pip install -r requirements.txt
python scripts/fetch_pretrained_weights.py    # only if download.pytorch.org is blocked

# Headline: detected-animal crop, location-held-out split
python scripts/run_training.py --data-dir data/night_wildlife --epochs 16 \
    --image-size 224 --pretrained --grayscale --freeze-until layer2 \
    --learning-rate 3e-4 --split-by location --crop-to-bbox \
    --output-dir results/demo --device cpu
python scripts/run_evaluation.py --output-dir results/demo --device cpu

# Full-frame comparison (same split): add --no-crop-to-bbox, --output-dir results/fullframe
# Same-location contrast:            add --split-by stratified, --output-dir results/samelocation
```

Tested dependency versions (ranges in `requirements.txt`; Python 3.11):

| Package | Tested |
|---------|--------|
| torch / torchvision | 2.13 / 0.28 |
| numpy | 2.4 |
| scikit-learn | 1.9 |
| matplotlib | 3.11 |
| Pillow | 12.3 |

Every run also records the exact environment it ran in to `environment.json`.

## What's in the repo

```
src/        config, data loading, location-grouped split, model, training, evaluation, detection
scripts/    build/validate the dataset, train, evaluate, predict, detection
tests/      unit tests + a tiny end-to-end smoke test (run: pytest)
docs/       literature review, methodology, experiment log, licensing, result plots
data/       the ready-made infrared dataset + manifest.csv
```

In short: ImageNet-pretrained ResNet-18, infrared grayscale input, later layers
retrained, with class weighting, augmentation, and early stopping. The full
pipeline — data selection, preprocessing, model, training, and evaluation
settings — is documented in [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md).

## Contributors

| Name | Student number | Main responsibility |
|------|----------------|---------------------|
| Srivani Konda (@srivanik8) | 25211398 | Data pipeline — dataset builder, preprocessing, and splits |
| Navya Sri Mungamuri | 25200230 | Model, training, and evaluation |

Both authors contributed to the literature review and the write-up.

## Licence and credit

- **Code:** MIT — see [`LICENSE`](LICENSE).
- **Data & weights:** the committed dataset is a derivative subset of Caltech
  Camera Traps (CDLA-Permissive), and the pretrained weights carry their own
  licences. Full details, attribution requirements, and exactly what is
  redistributed are in [`docs/DATA_LICENSE.md`](docs/DATA_LICENSE.md).

Caltech Camera Traps — Beery, Van Horn & Perona, *Recognition in Terra Incognita*,
ECCV 2018, via LILA BC (https://lila.science/datasets/caltech-camera-traps).
