#!/usr/bin/env python3
"""Classify a single image with a trained checkpoint.

The model is trained and evaluated on the **animal crop**, so this script applies
the same crop before classifying — otherwise the model sees a full frame it was
never trained on and accuracy drops sharply. The box is taken from the dataset
manifest when the image belongs to a built dataset; for an arbitrary image you can
pass `--detect` to locate the animal (MegaDetector when available), or
`--no-crop` to force the full frame.

Usage:
    python scripts/predict.py --checkpoint results/demo/best_model.pt path/to/image.jpg
    python scripts/predict.py --checkpoint results/demo/best_model.pt --detect photo.jpg
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import Config  # noqa: E402
from src.data import build_transforms, crop_to_box, lookup_bbox  # noqa: E402
from src.model import build_model  # noqa: E402


def _saved_temperature(checkpoint_path):
    """Read the temperature fitted during evaluation, from metrics.json alongside
    the checkpoint. Falls back to 1.0 (no calibration) when unavailable."""
    import json

    path = os.path.join(os.path.dirname(os.path.abspath(checkpoint_path)),
                        "metrics.json")
    if not os.path.exists(path):
        return 1.0
    try:
        with open(path) as fh:
            return float(json.load(fh).get("temperature", 1.0)) or 1.0
    except Exception:
        return 1.0


def resolve_box(image_path, use_detect, no_crop):
    """Return (box, how) describing which crop to apply."""
    if no_crop:
        return None, "full frame (--no-crop)"
    box = lookup_bbox(image_path)
    if box is not None:
        return box, "dataset manifest box"
    if use_detect:
        # Prefer MegaDetector: it is what the dataset's boxes were built with, and
        # a COCO detector has poor recall on infrared frames. Fall back to YOLOv8.
        try:
            from src import megadetector as md
            if md.available():
                box = md.best_animal_box(md.load_detector(), image_path)
                if box is not None:
                    return box, "MegaDetector detection"
                return None, "full frame (MegaDetector found no animal)"
        except SystemExit:
            pass                       # weights missing; try the COCO detector
        from src.detect import load_detector, best_animal_box, yolo_available
        if not yolo_available():
            print('[warn] --detect needs a detector: pip install yolov5 "setuptools<81"'
                  " (MegaDetector) or ultralytics (YOLOv8)")
            return None, "full frame (no detector available)"
        box = best_animal_box(load_detector(), image_path)
        if box is not None:
            return box, "YOLOv8 detection (COCO; weak on infrared)"
        return None, "full frame (no animal detected)"
    return None, "full frame (no box in manifest; try --detect)"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("image")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--topk", type=int, default=3)
    ap.add_argument("--detect", action="store_true",
                    help="use YOLOv8 to find the animal when the image has no "
                         "manifest box")
    ap.add_argument("--no-calibration", action="store_true",
                    help="skip the fitted temperature and show raw softmax")
    ap.add_argument("--no-crop", action="store_true",
                    help="classify the whole frame (matches a model trained with "
                         "--no-crop-to-bbox)")
    args = ap.parse_args()

    import torch
    from PIL import Image
    from src.utils import load_checkpoint

    state = load_checkpoint(args.checkpoint, map_location="cpu")  # weights_only=True
    class_names = state["class_names"]
    saved = state.get("config", {})
    cfg = Config(**{k: v for k, v in saved.items() if k in Config.__dataclass_fields__})

    net = build_model(cfg.backbone, len(class_names), pretrained=False, freeze_until="")
    net.load_state_dict(state["model_state"])
    net.eval()

    # Respect how the checkpoint was trained: a model trained on full frames
    # should be served full frames.
    no_crop = args.no_crop or not saved.get("crop_to_bbox", True)
    box, how = resolve_box(args.image, args.detect, no_crop)

    tf = build_transforms(cfg.image_size, cfg.grayscale_to_rgb, train=False,
                          pad_to_square=getattr(cfg, "pad_to_square", True))
    img = crop_to_box(Image.open(args.image).convert("RGB"), box)
    x = tf(img).unsqueeze(0)

    # Match evaluation exactly: same TTA setting, and the temperature that was
    # fitted on the validation split during evaluation (recorded in metrics.json),
    # so a printed confidence means the same thing as the reported ECE.
    tta = getattr(cfg, "tta", False)
    temperature = _saved_temperature(args.checkpoint) if not args.no_calibration else 1.0
    with torch.no_grad():
        logits = net(x)
        if tta:
            logits = (logits + net(torch.flip(x, dims=[3]))) / 2
        probs = torch.softmax(logits / temperature, dim=1).squeeze(0)

    topk = min(args.topk, len(class_names))
    scores, idx = probs.topk(topk)
    calib = f", T={temperature:.2f}" if temperature != 1.0 else ""
    print(f"Predictions for {args.image}  [input: {how}{'; TTA' if tta else ''}{calib}]")
    for rank, (s, i) in enumerate(zip(scores.tolist(), idx.tolist()), 1):
        print(f"  {rank}. {class_names[i]:20s} {s:.3f}")


if __name__ == "__main__":
    main()
