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
- **single-species** — frames annotated with more than one species are excluded
  (this is single-label classification);
- captured at **night** (local capture hour 19:00–06:59, from the `date_captured`
  field);
- verified to be **grayscale infrared** — the mean HSV saturation of the
  downloaded image is below 6 (daytime colour frames sit around 80);
- **de-duplicated by capture sequence** (`seq_id`): at most one frame per burst.

**Deterministic, stratified sampling.** Candidates are chosen in a fixed order
*before* any download, so filenames and which images are kept do not depend on
which concurrent download finishes first (re-running with the same seed is
byte-identical). Selection is stratified across **camera locations and time**:
within each location the frames are sorted by date and sampled at even intervals
across the whole range, then locations are visited round-robin. A per-location cap
(default 20) stops any one site dominating a class. The committed build spans
35–75 locations and 18–36 months per species.

**Non-destructive storage.** The downloaded frame is stored **uncropped** — only
downscaled to `--store-size` (default 384 px long side) — as grayscale JPEG. The
animal's bounding box is recorded in the manifest and the **crop is applied at
load time** (`crop_to_bbox`, `src/data.py`): if a box exists the loader crops to
it (15% padding); otherwise the whole frame is used. Nothing is baked into the
files, so the crop strategy (box vs. whole frame) is a runtime choice and the
originals are preserved. ~50% of the committed frames carry a bounding box.

**Split — location-held-out.** Camera-trap frames from the same site share
backgrounds, so a random split lets the model recognise the *location* instead of
the *animal*. `src/split.py` therefore assigns whole camera **locations** to a
single split (70/15/15 by image count, targeted per species so every species
appears in every split). No location — and therefore no background — is shared
between train, validation and test. The assignment is deterministic given the
seed and recorded in the manifest; `src/data.py` reads it. A stratified random
split is still available (`--split-by stratified`) for comparison.

**Manifest & validation.** `scripts/build_night_wildlife.py` writes
`data/night_wildlife/manifest.csv`, one row per image: split, class, saved
filename, source CCT image id, original filename, camera location, sequence id,
timestamp, month, season, bounding box, whether a box exists, and a SHA-256
checksum. It also writes `build_report.txt` logging every rejected/failed
download with its id and reason. `scripts/validate_dataset.py` then checks class
balance, file integrity (checksums + openability), split/location overlap, and
manifest↔file consistency — run it before training.

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
