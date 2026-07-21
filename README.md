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

`scripts/build_night_wildlife.py` builds this subset. It keeps only night
captures that are actually grayscale (real infrared), de-duplicates by capture
sequence, spreads images across many camera sites, and **crops to the animal's
bounding box** where CCT provides one (otherwise an aspect-preserving letterbox —
never a blind centre-crop that could slice the animal out). Every image is
recorded in [`data/night_wildlife/manifest.csv`](data/night_wildlife/manifest.csv)
with its source id, original filename, class, camera location, sequence id,
timestamp, split, and SHA-256 checksum, so the dataset is verifiable and the
split reproducible.

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
| **Location-held-out** | generalisation to **new camera sites** (the honest number) | **0.37** | 0.35 |
| Same-location (stratified) | preliminary — shares backgrounds with training | 0.73 | 0.73 |

Random guessing with 6 classes is 0.17, so the model does learn something about
the animals — but the drop from 0.73 to **0.37** when locations are held out shows
that most of the same-location score came from **recognising the background, not
the animal**. The location-held-out 0.37 is the number to trust; the 0.73 should
be read only as a same-location upper bound.

Per-species, location-held-out split (test = 356 images):

| Species  | Precision | Recall | F1   | Test images |
|----------|-----------|--------|------|-------------|
| coyote   | 0.41      | 0.54   | 0.47 | 69 |
| deer     | 0.42      | 0.40   | 0.41 | 77 |
| rabbit   | 0.32      | 0.47   | 0.38 | 47 |
| opossum  | 0.37      | 0.25   | 0.30 | 75 |
| raccoon  | 0.32      | 0.25   | 0.28 | 40 |
| bobcat   | 0.29      | 0.25   | 0.27 | 48 |

![training curves](docs/demo_results/training_curves.png)
![confusion matrix](docs/demo_results/confusion_matrix.png)

## Known limitations

- The 0.37 is from a **small** dataset (200 images/species) with only six species;
  behaviour on rare species and at larger scale is untested.
- Bounding boxes exist for only ~28% of the frames; the rest use a whole-frame
  letterbox, so some test images still contain background around the animal.
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
