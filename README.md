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

Every image is recorded in
[`data/night_wildlife/manifest.csv`](data/night_wildlife/manifest.csv) with its
source id, original filename, class, camera location, sequence id, timestamp,
month/season, bounding box, split, and SHA-256 checksum.
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

**2. Evaluate** on the held-out test set. This prints the scores and saves the
plots:

```bash
python scripts/run_evaluation.py --output-dir results/demo --device cpu
```

**3. Predict** on a single image:

```bash
python scripts/predict.py data/night_wildlife/coyote/coyote_0003.jpg \
    --checkpoint results/demo/best_model.pt
```

```
Predictions for data/night_wildlife/coyote/coyote_0003.jpg:
  1. coyote               0.933
  2. raccoon              0.046
  3. rabbit               0.009
```

Everything (checkpoint, `metrics.json`, and the plots) is written to
`results/demo/`. A saved copy of the plots is in
[`docs/demo_results/`](docs/demo_results/).

## Results

Trained on the 1,200 infrared images above, 16 epochs on CPU. The headline
number is the **location-held-out** split, where whole camera sites are kept out
of training so the model cannot lean on backgrounds it has already seen.

| Split | What it measures | Test acc | Macro F1 |
|-------|------------------|----------|----------|
| **Location-held-out** | generalisation to **new camera sites** (the honest number) | **0.55** | 0.55 |
| Same-location (stratified) | shares backgrounds with training | 0.64 | 0.64 |

Random guessing with 6 classes is 0.17. Because the model is trained on the
animal crop rather than the whole frame, the same-location advantage is small
(0.64 vs **0.55**) — i.e. the score reflects the animal, not the background. The
location-held-out **0.55** is the number to trust.

Per-species, location-held-out split (test = 233 images):

| Species  | Precision | Recall | F1   | Test images |
|----------|-----------|--------|------|-------------|
| deer     | 0.81      | 0.72   | 0.76 | 36 |
| rabbit   | 0.57      | 0.69   | 0.62 | 51 |
| opossum  | 0.72      | 0.53   | 0.61 | 34 |
| bobcat   | 0.42      | 0.56   | 0.48 | 41 |
| raccoon  | 0.61      | 0.33   | 0.43 | 33 |
| coyote   | 0.38      | 0.42   | 0.40 | 38 |

![training curves](docs/demo_results/training_curves.png)
![confusion matrix](docs/demo_results/confusion_matrix.png)

## Known limitations

- The 0.55 is from a **small** dataset (200 images/species) with only six species;
  behaviour on rare species and at larger scale is untested.
- Bounding boxes cover ~50% of the frames; the rest are classified from the whole
  (letterboxed) frame, so some test images still include background.
- The YOLOv8 detection stage in `src/detect.py` is optional and **not yet part of
  the reported results** (see
  [Issue #2](https://github.com/srivanik8/final-project/issues/2)).

## Making the dataset bigger

You can pull more images per class, or add more species, straight from the mirror:

```bash
python scripts/build_night_wildlife.py --out data/night_wildlife \
    --per-class 300 --species bobcat coyote raccoon opossum rabbit deer skunk fox
```

## What's in the repo

```
src/        config, data loading, location-grouped split, model, training, evaluation, detection
scripts/    build the dataset, train, evaluate, predict
docs/       literature review, methodology, experiment log, saved result plots
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

## Credit

Caltech Camera Traps — Beery, Van Horn & Perona, *Recognition in Terra Incognita*,
ECCV 2018, via LILA BC (https://lila.science/datasets/caltech-camera-traps).
