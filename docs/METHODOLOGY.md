# Methodology

Full detail on the pipeline. For a quick overview see the
[README](../README.md); this document is the reference for how each stage works
and the exact settings used.

## 1. Data

**Source.** Real infrared night-vision camera-trap frames from the Caltech
Camera Traps dataset (Beery et al., 2018), pulled from the LILA BC Google Cloud
mirror by `scripts/build_night_wildlife.py`.

**Selection.** From the CCT metadata (COCO format) we keep frames that are:

- labelled with one of six wild species — bobcat, coyote, raccoon, opossum,
  rabbit, deer;
- captured at **night** (local capture hour 19:00–06:59, from the `date_captured`
  field);
- verified to be **grayscale infrared** — the mean HSV saturation of the
  downloaded image is below 6 (daytime colour frames sit around 80);
- **de-duplicated by capture sequence** (`seq_id`): at most one frame per burst,
  so near-identical frames don't inflate the dataset;
- **spread across camera locations** — a per-location cap (default 20 images per
  species per site) stops any single camera dominating a class, so the
  location-held-out split below has enough distinct sites to work with (16–31
  locations per species).

**Cropping.** We do **not** blindly centre-crop, because that can slice the
animal out of frame. Instead:

- if the CCT metadata provides a **bounding box** for the frame, we crop to it
  (with 15% padding), then letterbox to 224×224;
- otherwise we use an **aspect-preserving letterbox** — resize the whole frame so
  the long side is 224 and pad the short side — so the animal is never cut and the
  aspect ratio is never distorted.

The committed subset is 200 images per species (1,200 total).

**Split — location-held-out.** Camera-trap frames from the same site share
backgrounds, so a random split lets the model recognise the *location* instead of
the *animal*. `src/split.py` therefore assigns whole camera **locations** to a
single split (70/15/15 by image count, targeted per species so every species
appears in every split). No location — and therefore no background — is shared
between train, validation and test. The assignment is deterministic given the
seed and is recorded in the dataset manifest (below); `src/data.py` reads the
split straight from the manifest. A stratified random split is still available
(`--split-by`/`split_by="stratified"`) for comparison.

**Manifest.** `scripts/build_night_wildlife.py` writes
`data/night_wildlife/manifest.csv`, one row per image, recording: assigned split,
class, saved filename, source CCT image id, original filename, camera location,
sequence id, capture timestamp, whether a bounding box was used, and a SHA-256
checksum. This makes the dataset verifiable and the experiment reproducible.

## 2. Preprocessing and augmentation

Infrared frames are single-channel. Because the backbone was pretrained on
3-channel ImageNet images, we replicate the grayscale channel to 3 channels
(`--grayscale`, on by default) and normalise with ImageNet statistics
(mean `[0.485, 0.456, 0.406]`, std `[0.229, 0.224, 0.225]`).

- **Training transforms:** `RandomResizedCrop(224, scale=0.7–1.0)`, horizontal
  flip, `RandomRotation(±10°)`, and colour jitter (brightness ±0.2, contrast
  ±0.2). These mimic the pose, framing, and brightness variation of real camera
  traps.
- **Validation/test transforms:** deterministic resize (to 1.15×) + centre crop.
  No augmentation, so the reported metrics are on clean inputs.

## 3. Model

Transfer learning from an ImageNet-pretrained **ResNet-18** (`src/model.py`):

- The final fully-connected layer is replaced with a fresh linear head sized to
  the number of species.
- Early residual blocks are **frozen** and the later blocks are **retrained**.
  The freeze point is controlled by `--freeze-until` (block order: `conv1`,
  `bn1`, `layer1`, `layer2`, `layer3`, `layer4`). The reported run uses
  `--freeze-until layer2`, i.e. `conv1`/`bn1`/`layer1`/`layer2` are frozen and
  `layer3`/`layer4`/head are trained. `""` trains the whole network; `all` freezes
  the backbone entirely (a linear probe on frozen features).

The rationale (Tabak et al., 2019): early convolutional filters — edges and
textures — transfer well across domains, while the deeper, task-specific layers
benefit from adapting to camera-trap imagery.

## 4. Training

`src/train.py`:

- **Loss:** cross-entropy with **inverse-frequency class weighting** (handles
  species imbalance) and **label smoothing** (0.05).
- **Optimiser:** AdamW, learning rate `3e-4`, weight decay `1e-4`.
- **Schedule:** cosine annealing over the epoch budget.
- **Epochs / batch size:** 16 / 32 for the reported run.
- **Early stopping:** training stops if validation accuracy does not improve for
  5 epochs (`--early-stop-patience`).
- **Checkpointing:** the best model by validation accuracy is saved to
  `results/<name>/best_model.pt`, together with the class names and the exact
  `Config`.

Everything is seeded (`--seed`, default 42) for reproducibility, and the config
is written to `config.json` next to the checkpoint so evaluation reproduces the
same split and preprocessing.

## 5. Evaluation

`src/evaluate.py` scores the best checkpoint on the location-held-out test split
and writes:

- **Metrics** (`metrics.json`): accuracy; precision, recall, and F1 both
  **macro-averaged** (each species weighted equally) and **weighted** (by
  support); plus a full per-class report and the split strategy used.
- **Confusion matrix** (`confusion_matrix.png`), row-normalised.

Macro and weighted averages are both reported because the macro number is the
honest one when class support is uneven — it does not let common species mask
poor performance on rare ones.

We deliberately do **not** plot our accuracy against the Norouzzadeh (2018) or
Schneider (2020) numbers. Those studies used different datasets, species, and
evaluation protocols, so a side-by-side bar chart would imply a comparison that
isn't valid. They appear in the literature review as context only.

## 6. Optional detection stage

`src/detect.py` wraps YOLOv8 to localise the animal and crop to it before
classification, removing background. It is optional (needs `ultralytics`),
degrades gracefully to the full frame when nothing is detected, and is **not yet
part of the reported results** (Issue #2).

## References

- Beery, S., Van Horn, G. & Perona, P. (2018). Recognition in Terra Incognita. ECCV.
- Tabak, M.A. et al. (2019). Machine learning to classify animal species in
  camera-trap images. Methods in Ecology and Evolution.
- Norouzzadeh, M.S. et al. (2018). Automatically identifying, counting, and
  describing wild animals in camera-trap images with deep learning. PNAS.
- Schneider, S. et al. (2020). Three critical factors affecting automated image
  species recognition in wildlife.
