"""MegaDetector v5 wrapper — camera-trap animal detection.

MegaDetector is the standard detector for camera-trap imagery: it is trained on
millions of camera-trap frames (including infrared night captures) and predicts
just three classes — ``animal``, ``person``, ``vehicle``. That makes it a much
better fit here than a COCO-pretrained detector, which has never seen a grayscale
IR frame and mislabels a deer as "cow" (the label is irrelevant to us — we only
want the box — but its *recall* on IR frames is what matters, and MegaDetector's
is substantially higher).

The released weights are a YOLOv5 checkpoint, so this module needs the ``yolov5``
package and aliases the top-level ``models``/``utils`` modules the checkpoint was
pickled against.

Weights: scripts/fetch_megadetector.py (checksum-verified).
Credit: Beery, S., Morris, D. & Yang, S. "Efficient pipeline for camera trap image
review" / MegaDetector, https://github.com/agentmorris/MegaDetector (MIT).
"""
from __future__ import annotations

import os
import sys

DEFAULT_WEIGHTS = ".cct_cache/md_v5a.0.0.pt"
ANIMAL_CLASS = 0                     # MegaDetector: 0=animal, 1=person, 2=vehicle
INPUT_SIZE = 640


def available() -> bool:
    try:
        import yolov5  # noqa: F401
        return True
    except Exception:
        return False


def _alias_yolov5_modules():
    """MegaDetector was pickled against the yolov5 repo's top-level module names."""
    import yolov5.models
    import yolov5.models.yolo
    import yolov5.utils

    sys.modules.setdefault("models", yolov5.models)
    sys.modules.setdefault("models.yolo", yolov5.models.yolo)
    sys.modules.setdefault("utils", yolov5.utils)


def load_detector(weights: str = None):
    """Load MegaDetector v5 on CPU in eval mode.

    Security note: this unpickles a ``.pt`` file, so only use weights you trust —
    ``scripts/fetch_megadetector.py`` verifies the official SHA-256.
    """
    import torch

    weights = weights or DEFAULT_WEIGHTS
    if not os.path.exists(weights):
        raise SystemExit(
            f"MegaDetector weights not found at {weights}. "
            "Run: python scripts/fetch_megadetector.py")
    _alias_yolov5_modules()
    ckpt = torch.load(weights, map_location="cpu", weights_only=False)
    return ckpt["model"].float().eval()


def best_animal_box(model, image, conf: float = 0.2, iou: float = 0.45):
    """Highest-confidence animal box as ``(x, y, w, h)`` in image pixels, or None.

    ``image`` is a path or a PIL image. Boxes are mapped back from the detector's
    letterboxed 640x640 input to the original image's coordinates.
    """
    import numpy as np
    import torch
    from PIL import Image
    from yolov5.utils.augmentations import letterbox
    from yolov5.utils.general import non_max_suppression

    if not hasattr(image, "size") or isinstance(image, str):
        image = Image.open(image)
    im0 = np.array(image.convert("RGB"))
    h0, w0 = im0.shape[:2]

    padded, ratio, (dw, dh) = letterbox(im0, INPUT_SIZE, stride=32, auto=False)
    x = torch.from_numpy(
        np.ascontiguousarray(padded.transpose(2, 0, 1)[::-1])).float().unsqueeze(0) / 255
    with torch.no_grad():
        pred = model(x)[0]
    det = non_max_suppression(pred, conf, iou)[0]
    if det is None or not len(det):
        return None

    best, best_conf = None, -1.0
    for *xyxy, score, cls in det.tolist():
        if int(cls) != ANIMAL_CLASS or score <= best_conf:
            continue
        best_conf = score
        x1, y1, x2, y2 = xyxy
        # undo letterbox padding and scaling
        x1 = (x1 - dw) / ratio[0]
        x2 = (x2 - dw) / ratio[0]
        y1 = (y1 - dh) / ratio[1]
        y2 = (y2 - dh) / ratio[1]
        x1, y1 = max(0.0, x1), max(0.0, y1)
        x2, y2 = min(float(w0), x2), min(float(h0), y2)
        if x2 - x1 > 1 and y2 - y1 > 1:
            best = (int(x1), int(y1), int(x2 - x1), int(y2 - y1))
    return best
