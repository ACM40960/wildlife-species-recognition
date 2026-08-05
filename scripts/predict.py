#!/usr/bin/env python3
"""Classify a single image with a trained checkpoint.

The model is trained and evaluated on the **animal crop**, so this script applies
the same crop before classifying — otherwise the model sees a full frame it was
never trained on and accuracy drops sharply. The box is taken from the dataset
manifest when the image belongs to a built dataset; for an arbitrary image you can
pass `--detect` to locate the animal with YOLOv8, or `--no-crop` to force the full
frame.

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


def resolve_box(image_path, use_detect, no_crop):
    """Return (box, how) describing which crop to apply."""
    if no_crop:
        return None, "full frame (--no-crop)"
    box = lookup_bbox(image_path)
    if box is not None:
        return box, "dataset manifest box"
    if use_detect:
        from src.detect import load_detector, best_animal_box, yolo_available
        if not yolo_available():
            print("[warn] --detect needs ultralytics: pip install ultralytics")
            return None, "full frame (YOLO unavailable)"
        box = best_animal_box(load_detector(), image_path)
        if box is not None:
            return box, "YOLO detection"
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
    with torch.no_grad():
        probs = torch.softmax(net(x), dim=1).squeeze(0)

    topk = min(args.topk, len(class_names))
    scores, idx = probs.topk(topk)
    print(f"Predictions for {args.image}  [input: {how}]")
    for rank, (s, i) in enumerate(zip(scores.tolist(), idx.tolist()), 1):
        print(f"  {rank}. {class_names[i]:20s} {s:.3f}")


if __name__ == "__main__":
    main()
