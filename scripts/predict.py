#!/usr/bin/env python3
"""Classify a single image with a trained checkpoint.

Usage:
    python scripts/predict.py --checkpoint results/demo/best_model.pt path/to/image.jpg
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import Config  # noqa: E402
from src.data import build_transforms  # noqa: E402
from src.model import build_model  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--topk", type=int, default=3)
    args = ap.parse_args()

    import torch
    from PIL import Image

    state = torch.load(args.checkpoint, map_location="cpu")
    class_names = state["class_names"]
    saved = state.get("config", {})
    cfg = Config(**{k: v for k, v in saved.items() if k in Config.__dataclass_fields__})

    net = build_model(cfg.backbone, len(class_names), pretrained=False, freeze_until="")
    net.load_state_dict(state["model_state"])
    net.eval()

    tf = build_transforms(cfg.image_size, cfg.grayscale_to_rgb, train=False)
    img = Image.open(args.image).convert("RGB")
    x = tf(img).unsqueeze(0)
    with torch.no_grad():
        probs = torch.softmax(net(x), dim=1).squeeze(0)

    topk = min(args.topk, len(class_names))
    scores, idx = probs.topk(topk)
    print(f"Predictions for {args.image}:")
    for rank, (s, i) in enumerate(zip(scores.tolist(), idx.tolist()), 1):
        print(f"  {rank}. {class_names[i]:20s} {s:.3f}")


if __name__ == "__main__":
    main()
