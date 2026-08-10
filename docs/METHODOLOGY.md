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
originals are preserved. 86% of the committed frames carry a bounding box:
50% ground-truth (598), the rest added by the detector fill step — see §6.

**Split — location-held-out.** Camera-trap frames from the same site share
backgrounds, so a random split lets the model recognise the *location* instead of
the *animal*. `src/split.py` therefore assigns whole camera **locations** to a
single split. Targets are 70/15/15 by image count and are applied per species so
every species appears in every split; because whole locations move together the
realised split on the committed data is 61/19/19 (735/232/233). No location — and
therefore no background — is shared between train, validation and test. The assignment is deterministic given the
seed and recorded in the manifest; `src/data.py` reads it. A stratified random
split is still available (`--split-by stratified`) for comparison.

Note that the location-held-out run additionally holds out a *seen-location* slice
from training (`seen_test_fraction`, §5), so it trains on fewer images (~625) than
the stratified run (~840). When comparing the location-held-out (0.69) and
same-location numbers, part of any difference is this training-set-size
difference, not only the split — so the seen-vs-unseen comparison (§5), which uses
one model on both, is the cleaner measure of the generalisation gap.

**Manifest & validation.** `scripts/build_night_wildlife.py` writes
`data/night_wildlife/manifest.csv`, one row per image: split, class, saved
filename, source CCT image id, original filename, camera location, sequence id,
timestamp, month, season, bounding box, whether a box exists, and a SHA-256
checksum. It also writes `build_report.txt` logging every rejected/failed
download with its id and reason. `scripts/validate_dataset.py` then checks class
balance, file integrity (checksums + openability), split/location overlap,
manifest↔file consistency, and manifest self-consistency (`has_bbox`, `bbox` and
`box_source` must agree on every row) — run it before training.

## 2. Preprocessing and augmentation

Infrared frames are single-channel. Because the backbone was pretrained on
3-channel ImageNet images, we replicate the grayscale channel to 3 channels
(`--grayscale`, on by default) and normalise with ImageNet statistics
(mean `[0.485, 0.456, 0.406]`, std `[0.229, 0.224, 0.225]`).

- **Training transforms:** letterbox, then `RandomResizedCrop(224,
  scale=0.8–1.0)`, horizontal flip, `RandomRotation(±10°)`, and colour jitter
  (brightness ±0.2, contrast ±0.2). These mimic the pose, framing, and brightness
  variation of real camera traps.
- **Letterbox padding:** frames are resized preserving aspect ratio and padded to
  a square rather than centre-cropped. A centre crop of a 4:3 frame discarded ~35%
  of its width and could remove the animal entirely.
- **Infrared-specific augmentation:** gamma/brightness jitter (IR flash exposure
  varies widely), mild sharpness jitter, and random erasing, which simulates
  occlusion and discourages relying on the background.
- **Validation/test transforms:** letterbox only — deterministic, no augmentation,
  and nothing cropped away.

## 3. Model

Transfer learning from an ImageNet-pretrained **ResNet-18** (`src/model.py`):

- The final fully-connected layer is replaced with a fresh linear head sized to
  the number of species.
- Early residual blocks are **frozen** and the later blocks are **retrained**.
  The freeze point is controlled by `--freeze-until` (block order: `conv1`,
  `bn1`, `layer1`, `layer2`, `layer3`, `layer4`); the **default is `layer2`**, so
  `conv1`/`bn1`/`layer1`/`layer2` are frozen and `layer3`/`layer4`/head are
  trained (~10.5M of 11.2M params). `""` trains the whole network; `all` (or
  `layer4`) freezes the backbone entirely (head-only / linear probe).
- **Frozen BatchNorm is kept in eval mode** during training. A frozen BN layer
  otherwise keeps updating its running mean/variance under `net.train()`, quietly
  changing a "frozen" layer; pinning it to eval keeps the ImageNet statistics.
  This alone lifted the held-out accuracy from 0.51 to 0.55.

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
  `Config`. The best-so-far accuracy starts below zero, so a checkpoint is written
  even if the first epoch's validation accuracy is 0.

**Reproducibility.** Every run is seeded (`--seed`, default 42) and, unless
`deterministic` is turned off, PyTorch is asked for deterministic algorithms
(`torch.use_deterministic_algorithms(warn_only=True)`, cuDNN autotuning off).
Each run writes: `config.json` (exact settings), `environment.json` (Python,
PyTorch, Torchvision, CUDA and OS versions), and the full per-epoch history as
both `history.json` and `history.csv` (loss, accuracy, learning rate, timing) —
not only the training-curve plot.

## 5. Evaluation

`src/evaluate.py` first **validates the checkpoint** against the current dataset —
the class names must match exactly (same order) and every setting that decides
which pixels the model sees or which images it was trained on — backbone, image
size, grayscale, `crop_to_bbox`, `pad_to_square`, `split_by`, `seed`,
`seen_test_fraction`, and (under the stratified split) the val/test fractions —
must match what the checkpoint was trained with; a mismatch raises rather than
silently reporting nonsense. It then scores the checkpoint and writes:

> **Two things to know before quoting any number below.**
>
> 1. **Which interval to report.** Accuracy and macro-F1 carry a **cluster
>    bootstrap over the 25 test camera locations** (`accuracy_cluster_95ci`,
>    `f1_macro_cluster_95ci`). That is the one to quote. Frames from one camera
>    share a background, a season and often the same individual, so they are not
>    independent; the split is location-grouped for exactly that reason. The
>    image-level Wilson and bootstrap intervals are still written out but are
>    suffixed **`_naive`** and kept only for comparison with the older runs in
>    [`experiments.md`](experiments.md). They are roughly 35% narrower and
>    understate the uncertainty of any claim about new cameras: 0.625 to 0.743
>    per image against 0.597 to 0.780 clustered.
> 2. **The test result is a development estimate.** The unseen-location test set
>    was consulted while choosing the crop strategy, the detector, the
>    augmentation and whether to use TTA, so it guided pipeline selection and is
>    not an untouched held-out estimate. `metrics.json` records this as
>    `test_set_status`. Treat 0.687 as slightly optimistic. Section
>    [Limitations in the README](../README.md#limitations) explains why it cannot
>    be repaired with this dataset.

- **Metrics** (`metrics.json`), all with 95% confidence intervals because the
  test set is small (~30/species):
  - accuracy with a **location-clustered bootstrap** interval (and a Wilson
    interval retained as `accuracy_wilson_95ci_naive`);
  - **temperature scaling** (on by default): one temperature is fitted on the
    *validation* split (never on test) and applied to the logits. It leaves
    predictions and accuracy untouched and only makes the confidences honest —
    on the reported run ECE fell 0.054 → 0.048 (T = 1.07). The bulk of the
    calibration gain (0.150 → 0.054) came from the better model itself, not from
    this step. `--no-temperature-scaling` reports raw confidences.
  - **test-time augmentation** (`--tta`, **off by default**): averages the logits
    over the frame and its horizontal mirror. In principle a free ensemble, but
    measured on this data it is slightly *worse* (0.687 → 0.682 accuracy, ECE
    0.048 → 0.072) for 2x the inference cost, so it is disabled — see
    [`experiments.md`](experiments.md).
  - **balanced accuracy** and macro precision/recall/F1, with a **location-
    clustered bootstrap** interval on macro-F1 (the image-level one is kept as
    `f1_macro_boot_95ci_naive`);
  - **top-2 / top-3** accuracy;
  - **expected calibration error** (are the confidences trustworthy?);
  - a per-class report with a Wilson interval on each species' recall. These are
    per-image and therefore optimistic in the same way; read them as indicative.
  - a breakdown by **`box_source`**, plus `oracle_gt_box` and `no_oracle_box`.
    Roughly half the test frames carry a ground-truth box that would not exist at
    inference, and they score 0.778 against 0.608 for the rest. See the
    detector-only evaluation below.
- **Seen vs. unseen locations.** A fixed, deterministic fraction of the
  training-location images is held out of training (`seen_test_fraction`) and
  scored separately, so the report shows accuracy on **seen** camera sites next to
  the **unseen** number — the gap is the generalisation cost. Both are
  development estimates, per the note above.
- **Per-image predictions** (`predictions.csv`): filename, location, box source,
  true and predicted class, and confidence, so any metric here can be recomputed
  from source rather than taken on trust.
- **Confusion matrix** (`confusion_matrix.png`) and an **error-analysis montage**
  (`error_analysis.png`) of the most-confident correct predictions and the
  most-confident mistakes, to inspect whether errors come from darkness,
  occlusion, cropping or similar species.

Macro averages are emphasised because they don't let common species mask poor
performance on rare ones. We deliberately do **not** plot our accuracy against the
Norouzzadeh (2018) or Schneider (2020) numbers — different datasets, species and
protocols make that comparison invalid; they appear in the literature review as
context only.

**Detected-animal vs. full-frame.** Because the crop is applied at load time
(`crop_to_bbox`), the same split can be trained and evaluated with the animal
cropped or on the full frame just by toggling `--crop-to-bbox` / `--no-crop-to-bbox`.
Cropping to the animal (using the dataset/MegaDetector boxes) improves
unseen-location accuracy from 0.46 to 0.69 and cuts the calibration error
sharply (0.22 → 0.05), so detection is a genuine part of the pipeline, not an afterthought.

**Detector-only (end-to-end) evaluation.** The 0.69 above uses whatever box the
manifest holds, and for about half the test frames that is a *ground-truth* box,
which is an oracle no deployed system has. To measure the pipeline as it would
actually run, `scripts/make_detector_manifest.py` rebuilds the manifest with every
ground-truth box discarded and MegaDetector run over every test frame, and
evaluation is pointed at it:

```bash
python scripts/make_detector_manifest.py --splits test
python scripts/run_evaluation.py --output-dir results/v5_final \
    --manifest-name manifest_detector.csv \
    --artifacts-dir results/v5_detector_only --device cpu
```

`--artifacts-dir` is required in practice: without it this run overwrites the
baseline `metrics.json` and plots in `results/v5_final`, destroying the very
numbers it is meant to be compared against. The two committed bundles are
[`demo_results`](demo_results) (baseline) and
[`demo_results_detector_only`](demo_results_detector_only), each recording the
`manifest_sha256` it used.

The end-to-end result is **0.682 accuracy, AUC 0.900** at 74% box coverage,
against 0.687 / 0.891 at 84% coverage with the oracle boxes. The difference sits
well inside the interval, so the headline does not depend on the ground-truth
annotations. What does matter is whether a frame gets a box at all: within the
detector-only run, frames with a box score 0.808 and frames without score 0.328.

## 6. Detection stage (MegaDetector, with a YOLOv8 fallback)

`src/megadetector.py` wraps MegaDetector v5 (the default) and `src/detect.py`
wraps a COCO-pretrained YOLOv8n detector (`--detector yolov8`). About half of the CCT
frames (598 of 1,200) have a ground-truth bounding box; `scripts/fill_boxes_yolo.py` runs the
detector over the frames that don't and writes the detected box into the manifest
(`box_source = gt | megadetector | yolov8 | none`), raising box coverage from 50% to **86%**. The
detector runs on the already-stored frames, so no re-download is needed, and the
box is used by the same load-time `crop_to_bbox` path as the ground-truth boxes.
`scripts/fetch_yolo_weights.py` fetches the weights from a checksum-verified
mirror for offline environments.

The detector's *class* prediction is ignored — only the box is used, to crop to
the animal. The COCO-pretrained YOLOv8 reached 66% coverage but added no
measurable accuracy, because it has never seen a grayscale infrared frame.
Switching to **MegaDetector** — trained on camera-trap imagery, with classes
animal/person/vehicle — raised coverage to 86% and contributed roughly +0.07
accuracy (0.614 -> 0.687, see docs/experiments.md). `--detector yolov8` keeps the
old behaviour for comparison.

## References

- Beery, S., Van Horn, G. & Perona, P. (2018). Recognition in Terra Incognita. ECCV.
- Tabak, M.A. et al. (2019). Machine learning to classify animal species in
  camera-trap images. Methods in Ecology and Evolution.
- Norouzzadeh, M.S. et al. (2018). Automatically identifying, counting, and
  describing wild animals in camera-trap images with deep learning. PNAS.
- Schneider, S. et al. (2020). Three critical factors affecting automated image
  species recognition in wildlife.
