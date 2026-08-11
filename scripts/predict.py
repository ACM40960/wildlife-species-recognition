#!/usr/bin/env python3
"""classify a single image with a trained checkpoint."""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import Config  # noqa: E402
from src.data import build_transforms, crop_to_box, lookup_bbox  # noqa: E402
from src.model import build_model  # noqa: E402


def _saved_calibration(checkpoint_path):
    """read (tta, temperature) from the metrics.json written next to the checkpoint."""
    import json

    path = os.path.join(os.path.dirname(os.path.abspath(checkpoint_path)),
                        "metrics.json")
    if not os.path.exists(path):
        return False, 1.0
    try:
        with open(path) as fh:
            m = json.load(fh)
        return bool(m.get("tta", False)), float(m.get("temperature", 1.0)) or 1.0
    except Exception:
        return False, 1.0


def resolve_box(image_path, use_detect, no_crop):
    """return (box, how) describing which crop to apply."""
    if no_crop:
        return None, "full frame (--no-crop)"
    box = lookup_bbox(image_path)
    if box is not None:
        return box, "dataset manifest box"
    if use_detect:
        # prefer MegaDetector: it is what the dataset's boxes were built with, and a COCO
        try:
            from src import megadetector as md
            if md.available():
                box = md.best_animal_box(md.load_detector(), image_path)
                if box is not None:
                    return box, "MegaDetector detection"
                return None, "full frame (MegaDetector found no animal)"
        except SystemExit:
            pass  # weights missing; try the COCO detector
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
                    help="detect the animal when the image has no manifest box "
                         "(MegaDetector if installed, else YOLOv8)")
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

    # respect how the checkpoint was trained: a model trained on full frames should be served
    no_crop = args.no_crop or not saved.get("crop_to_bbox", True)
    box, how = resolve_box(args.image, args.detect, no_crop)

    tf = build_transforms(cfg.image_size, cfg.grayscale_to_rgb, train=False,
                          pad_to_square=getattr(cfg, "pad_to_square", True))
    img = crop_to_box(Image.open(args.image).convert("RGB"), box)
    x = tf(img).unsqueeze(0)

    # match evaluation exactly: same TTA setting and the temperature fitted on val
    if args.no_calibration:
        tta, temperature = getattr(cfg, "tta", False), 1.0
    else:
        tta, temperature = _saved_calibration(args.checkpoint)
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
