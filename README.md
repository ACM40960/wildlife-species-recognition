# Night-Vision Wildlife Species Recognition

**Identifying animal species in infrared camera-trap images**
ACM 40960 — Project 9 · University College Dublin, Summer 2026

Camera traps photograph millions of animals a year, and after dark they switch to
an infrared flash: grayscale, low-contrast frames that models trained on ordinary
daytime photos handle poorly. This project trains a convolutional neural network
to identify six species in those night-time frames, and — crucially — measures how
well it works on **camera sites it has never seen**.

**Headline result: 0.687 accuracy (95% CI 0.625–0.743), macro AUC 0.891**, on
held-out camera locations. Random guessing would be 0.167.

![confusion matrix](docs/demo_results/confusion_matrix.png)
![ROC curves](docs/demo_results/roc_curves.png)

---

## Quick start

```bash
conda env create -f environment.yml && conda activate night-wildlife
# or: pip install -r requirements.txt

python scripts/run_training.py   --data-dir data/night_wildlife --output-dir results/demo --device cpu
python scripts/run_evaluation.py --output-dir results/demo --device cpu
python scripts/predict.py data/night_wildlife/bobcat/bobcat_0032.jpg \
    --checkpoint results/demo/best_model.pt
```

```
Predictions for data/night_wildlife/bobcat/bobcat_0032.jpg  [input: dataset manifest box, T=1.07]
  1. bobcat               0.843
  2. coyote               0.127
  3. raccoon              0.015
```

Training takes ~15 min on a laptop CPU. Everything needed is in the repo — the
dataset ships with it.

> **In a container (Codespaces/Docker)?** Data loading runs in-process by default
> because containers mount a tiny `/dev/shm`. If yours is large, add
> `--num-workers 2`.

---

## Results

Evaluated on **233 images from camera locations excluded from training**. All
metrics are macro-averaged (each species counts equally) and reported with 95%
confidence intervals, because the test set is small.

| Metric | Value |
|--------|-------|
| **Accuracy** | **0.687** (0.625 – 0.743) |
| Balanced accuracy | 0.680 |
| Precision (macro) | 0.691 |
| Recall (macro) | 0.680 |
| **F1 (macro)** | **0.682** (0.623 – 0.740) |
| **AUC (macro, one-vs-rest)** | **0.891** |
| Top-2 accuracy | 0.828 |
| Expected calibration error | 0.048 |

The AUC of 0.891 says the model *ranks* species well even where its top-1 choice
is wrong — the confusions are between genuinely similar animals, not random.

**Per species:**

| Species | Precision | Recall | F1 | Test images |
|---------|-----------|--------|-----|-------------|
| rabbit | 0.93 | 0.78 | 0.85 | 51 |
| deer | 0.72 | 0.78 | 0.75 | 36 |
| opossum | 0.70 | 0.62 | 0.66 | 34 |
| raccoon | 0.69 | 0.61 | 0.65 | 33 |
| coyote | 0.58 | 0.68 | 0.63 | 38 |
| bobcat | 0.53 | 0.61 | 0.57 | 41 |

Rabbit and deer are easiest — distinctive silhouettes. Bobcat and coyote are
hardest and are mostly confused with *each other*: similar-sized four-legged
carnivores, which in a grayscale infrared frame look much alike.

### The key finding: seen vs. unseen cameras

The same model, evaluated on both:

| Camera locations | Accuracy |
|------------------|----------|
| **Unseen** (never trained on) | **0.687** |
| Seen (held-out images, familiar backgrounds) | 0.800 |

The **+0.11 gap** is the scientifically interesting result. Camera-trap frames from
one site share a background, so a model can learn the *place* instead of the
*animal*. Splitting by location — and cropping to the detected animal — is what
makes the 0.687 an honest measure of species recognition. Reported on a random
split instead, the same pipeline would look better than it is.

![training curves](docs/demo_results/training_curves.png)

---

## Method in brief

Infrared frames → crop to the animal → ImageNet-pretrained **ResNet-18** with the
later layers retrained → six-way classification.

| Stage | Choice |
|-------|--------|
| Data | 1,200 night infrared frames, 6 species × 200, from Caltech Camera Traps |
| Boxes | 86% of frames have an animal box (50% dataset ground truth, rest from MegaDetector) |
| Split | **Location-held-out** — whole camera sites go to train *or* val *or* test |
| Preprocessing | Crop to animal box; letterbox pad (nothing cut off); infrared augmentation |
| Model | ResNet-18, ImageNet weights, `layer3`/`layer4`/head retrained |
| Training | AdamW 3e-4, cosine schedule, class weighting, early stopping, 16 epochs |
| Evaluation | Confusion matrix, ROC/AUC, macro P/R/F1, calibration, seen-vs-unseen |

**Full detail is in [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md)** — every setting,
why it was chosen, and what each change was worth.

What moved the number, measured one change at a time
([run log](docs/experiments.md)):

| Change | Accuracy |
|--------|----------|
| Baseline | 0.55 |
| + letterbox padding & infrared augmentation | 0.61 |
| + MegaDetector boxes | **0.69** |

Two things we tried that did **not** help are recorded there too — test-time
augmentation was slightly worse, and temperature scaling improves calibration
without touching accuracy.

---

## Repository

```
src/        config · data loading · location split · model · training · evaluation · detection
scripts/    build the dataset · validate it · train · evaluate · predict
docs/       methodology · literature review · experiment log · results
data/       the infrared dataset + manifest.csv (provenance & checksums)
tests/      39 tests — run with `pytest`
```

- [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md) — the full pipeline
- [`docs/literature_review.md`](docs/literature_review.md) — background and prior work
- [`docs/experiments.md`](docs/experiments.md) — every run and what it changed
- [`docs/DATA_LICENSE.md`](docs/DATA_LICENSE.md) — data licensing and attribution

**Reproducing:** every run is seeded and writes `config.json`, `environment.json`
and a full per-epoch `history.csv` beside its checkpoint. `pytest` covers the
split logic, preprocessing, metrics and an end-to-end training smoke test.
`python scripts/validate_dataset.py` checks the dataset before you train.

## Limitations

- 6 species and 1,200 images. 0.687 is the accuracy on *these* unseen Caltech
  Camera Traps sites — not a claim about other regions, datasets or species.
- 14% of frames have no animal box and are classified from the whole frame.
- Single model, single seed; no cross-validation over multiple location splits.

## Contributors

| Name | Student number | Main responsibility |
|------|----------------|---------------------|
| Srivani Konda ([@srivanik8](https://github.com/srivanik8)) | 25211398 | Data pipeline — dataset builder, preprocessing, splits |
| Navya Sri Mungamuri | 25200230 | Model, training, evaluation, results |

## Licence and credit

Code: **MIT** ([`LICENSE`](LICENSE)). Data: Caltech Camera Traps — Beery, Van Horn
& Perona, *Recognition in Terra Incognita*, ECCV 2018, via
[LILA BC](https://lila.science/datasets/caltech-camera-traps) under
CDLA-Permissive. Attribution and what is redistributed:
[`docs/DATA_LICENSE.md`](docs/DATA_LICENSE.md).
