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

`scripts/build_night_wildlife.py` builds this subset: it keeps only night
captures that are actually grayscale (real infrared), and de-duplicates by
capture sequence so near-identical burst frames don't end up split across
training and test.

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

Trained on the 1,200 infrared images above (70% train / 15% val / 15% test),
16 epochs on CPU. Test set = 180 images, 30 per class.

**Overall test accuracy: 0.77** (random guessing would be 0.17 with 6 classes).
Macro precision 0.77, recall 0.77, F1 0.77.

| Species  | Precision | Recall | F1   | Test images |
|----------|-----------|--------|------|-------------|
| deer     | 0.88      | 0.93   | 0.90 | 30 |
| rabbit   | 0.83      | 0.80   | 0.81 | 30 |
| coyote   | 0.78      | 0.70   | 0.74 | 30 |
| raccoon  | 0.78      | 0.70   | 0.74 | 30 |
| opossum  | 0.69      | 0.80   | 0.74 | 30 |
| bobcat   | 0.70      | 0.70   | 0.70 | 30 |

Deer are the easiest (distinct shape and legs); bobcat is the hardest, and most
of its mistakes are with the other similarly-sized carnivores. This lines up with
the literature: accuracy on night infrared camera-trap images sits well below the
90%+ that models reach on clean daytime photos.

![training curves](docs/demo_results/training_curves.png)
![confusion matrix](docs/demo_results/confusion_matrix.png)

## Making the dataset bigger

You can pull more images per class, or add more species, straight from the mirror:

```bash
python scripts/build_night_wildlife.py --out data/night_wildlife \
    --per-class 300 --species bobcat coyote raccoon opossum rabbit deer skunk fox
```

## What's in the repo

```
src/        config, data loading + preprocessing, model, training, evaluation, detection
scripts/    build the dataset, train, evaluate, predict
docs/       literature review + saved result plots
data/       the ready-made infrared dataset
```

Main design choices:

- **Preprocessing** — infrared frames are single-channel, so we convert them to
  3 channels and normalise with ImageNet stats. Training adds random crops,
  flips, rotations and brightness/contrast jitter.
- **Model** — ResNet-18 pretrained on ImageNet, with a new final layer for our
  species. Early layers are frozen and the later layers are retrained (transfer
  learning).
- **Training** — class weighting for imbalance, label smoothing, AdamW with a
  cosine schedule, early stopping, and saving the best checkpoint.
- **Evaluation** — accuracy, precision, recall, F1 (per class and averaged), a
  confusion matrix, and a comparison against published baselines.
- **Detection (optional)** — `src/detect.py` can use YOLOv8 to crop to the animal
  before classifying, which removes background. Install `ultralytics` to use it.

## Credit

Caltech Camera Traps — Beery, Van Horn & Perona, *Recognition in Terra Incognita*,
ECCV 2018, via LILA BC (https://lila.science/datasets/caltech-camera-traps).
