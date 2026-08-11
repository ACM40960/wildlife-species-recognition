"""optional YOLOv8 animal-detection / cropping stage."""
from __future__ import annotations

# COCO class ids that correspond to animals
_COCO_ANIMAL_IDS = {14, 15, 16, 17, 18, 19, 20, 21, 22, 23}


def yolo_available() -> bool:
    try:
        import ultralytics  # noqa: F401
        return True
    except Exception:
        return False


# default location the offline fetcher writes to (scripts/fetch_yolo_weights.py)
DEFAULT_WEIGHTS = ".cct_cache/yolov8n.pt"


def load_detector(weights: str = None):
    """load a YOLOv8 model."""
    import os
    from ultralytics import YOLO

    if weights is None:
        weights = DEFAULT_WEIGHTS if os.path.exists(DEFAULT_WEIGHTS) else "yolov8n.pt"
    return YOLO(weights)


def best_animal_box(model, image, conf: float = 0.2):
    """return the highest-confidence animal box as (x, y, w, h) in pixel coords, or None if."""
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
