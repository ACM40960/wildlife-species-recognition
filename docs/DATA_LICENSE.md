# Data licensing and attribution

This project's **source code** is licensed under the MIT License (see
[`LICENSE`](../LICENSE)). The **data** and **model weights** come from third
parties and carry their own licences, summarised here.

## Dataset: Caltech Camera Traps (CCT)

- **Source:** Caltech Camera Traps, distributed by the LILA BC repository —
  https://lila.science/datasets/caltech-camera-traps
- **Paper / attribution (required):** Beery, S., Van Horn, G. & Perona, P.
  "Recognition in Terra Incognita." *ECCV 2018.* Please cite this paper in any
  work that uses this data.
- **Licence:** the CCT dataset is released by LILA under the
  **Community Data License Agreement – Permissive** (CDLA-Permissive). Under
  that agreement you may use and redistribute the data and derivatives, provided
  the attribution and licensing text are preserved. See
  https://lila.science/terms and https://cdla.dev/permissive-1-0/ for the exact
  terms.

### What is redistributed in this repository

`data/night_wildlife/` is a small **derivative subset** of CCT, not the original
dataset. Specifically it contains:

- **1,200 frames** (6 species × 200), selected as night-time **infrared** frames
  only, single-species, and de-duplicated by capture sequence;
- each frame **converted to grayscale and downscaled** (long side 384 px) — it is
  therefore a reduced-resolution derivative, not the original image;
- `manifest.csv`, which records for every frame the **original CCT image id** and
  filename, so each image can be traced back to the source dataset.

No CCT metadata beyond what is in the manifest is redistributed. To obtain the
full-resolution originals or the complete dataset, download it from LILA using
the ids in the manifest. The subset is provided for **research and educational**
use, consistent with the CDLA-Permissive terms, with attribution above.

If you are the data owner and have any concern about this redistribution, please
open an issue and we will address it.

## Model weights

- **ResNet-18 (ImageNet):** downloaded from torchvision; the ImageNet-pretrained
  weights are provided by the PyTorch/torchvision project under its BSD-3-Clause
  licence. `scripts/fetch_pretrained_weights.py` fetches an identical,
  checksum-verified copy for offline use.
- **YOLOv8n (COCO):** from Ultralytics. Ultralytics code and the YOLOv8 weights
  are distributed under **AGPL-3.0**; if you build on the detection stage, review
  the AGPL terms at https://github.com/ultralytics/ultralytics. YOLO is an
  optional stage here (`src/detect.py`, `scripts/fill_boxes_yolo.py`).

## Summary

| Component | Licence | Attribution |
|-----------|---------|-------------|
| This repository's code | MIT (`LICENSE`) | — |
| `data/night_wildlife/` (CCT subset) | CDLA-Permissive | Beery et al., ECCV 2018 |
| ResNet-18 ImageNet weights | BSD-3-Clause (torchvision) | PyTorch project |
| YOLOv8n weights (optional) | AGPL-3.0 (Ultralytics) | Ultralytics |
