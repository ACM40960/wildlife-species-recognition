"""Optional YOLOv8 animal-detection / cropping stage.

Camera-trap frames are mostly empty background with a small animal somewhere in
the frame. A common, accuracy-boosting preprocessing step is to first *detect*
the animal with an object detector and crop to it, so the classifier sees the
animal rather than a field of grass. This module wraps Ultralytics YOLOv8 for
that purpose. It is optional: install with ``pip install ultralytics``.

The generic COCO-pretrained YOLOv8 model already knows broad animal categories
(bird, cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe). For a
production pipeline you would fine-tune YOLO on camera-trap boxes, but even the
off-the-shelf model is useful for background removal.
"""
from __future__ import annotations

from typing import Optional, Tuple

from PIL import Image

# COCO class ids that correspond to animals.
_COCO_ANIMAL_IDS = {14, 15, 16, 17, 18, 19, 20, 21, 22, 23}


def yolo_available() -> bool:
    try:
        import ultralytics  # noqa: F401
        return True
    except Exception:
        return False


# Default location the offline fetcher writes to (scripts/fetch_yolo_weights.py).
DEFAULT_WEIGHTS = ".cct_cache/yolov8n.pt"


def load_detector(weights: str = None):
    """Load a YOLOv8 model.

    Prefers the checksum-verified local weights fetched by
    ``scripts/fetch_yolo_weights.py``; otherwise falls back to ultralytics'
    automatic download (needs internet).
    """
    import os
    from ultralytics import YOLO

    if weights is None:
        weights = DEFAULT_WEIGHTS if os.path.exists(DEFAULT_WEIGHTS) else "yolov8n.pt"
    return YOLO(weights)


def best_animal_box(model, image, conf: float = 0.2):
    """Return the highest-confidence animal box as (x, y, w, h) in pixel coords,
    or None if no animal is detected. ``image`` is a path or PIL image.

    Only COCO animal categories are considered; the *class* YOLO assigns is
    irrelevant here (a deer may be called "cow") — we only want the box to crop to.
    """
    results = model(image, conf=conf, verbose=False)
    best, best_conf = None, -1.0
    for r in results:
        if r.boxes is None:
            continue
        for box in r.boxes:
            if int(box.cls.item()) in _COCO_ANIMAL_IDS and float(box.conf.item()) > best_conf:
                best_conf = float(box.conf.item())
                x1, y1, x2, y2 = box.xyxy.squeeze().tolist()
                best = (int(x1), int(y1), int(x2 - x1), int(y2 - y1))
    return best


def detect_and_crop(model, image_path: str, conf: float = 0.25,
                    pad: float = 0.10) -> Tuple[Image.Image, Optional[Tuple[int, int, int, int]]]:
    """Return the crop around the highest-confidence animal box, or the full
    image if nothing is detected. Also returns the box (x1,y1,x2,y2) or None.
    """
    img = Image.open(image_path).convert("RGB")
    results = model(image_path, conf=conf, verbose=False)

    best_box, best_conf = None, -1.0
    for r in results:
        if r.boxes is None:
            continue
        for box in r.boxes:
            cls_id = int(box.cls.item())
            c = float(box.conf.item())
            if cls_id in _COCO_ANIMAL_IDS and c > best_conf:
                best_conf = c
                best_box = box.xyxy.squeeze().tolist()

    if best_box is None:
        return img, None

    w, h = img.size
    x1, y1, x2, y2 = best_box
    px, py = (x2 - x1) * pad, (y2 - y1) * pad
    x1 = max(0, int(x1 - px)); y1 = max(0, int(y1 - py))
    x2 = min(w, int(x2 + px)); y2 = min(h, int(y2 + py))
    return img.crop((x1, y1, x2, y2)), (x1, y1, x2, y2)
