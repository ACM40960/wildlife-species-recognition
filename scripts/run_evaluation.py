#!/usr/bin/env python3
"""Evaluate a trained checkpoint on the held-out test split.

Reads the config saved next to the checkpoint (results/config.json) so the same
data split and preprocessing are reproduced, then reports accuracy / precision /
recall and writes plots.

Examples:
    python scripts/run_evaluation.py --output-dir results/demo
    python scripts/run_evaluation.py --output-dir results --checkpoint results/best_model.pt
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import Config  # noqa: E402
from src.evaluate import evaluate  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output-dir", default="results",
                    help="dir containing config.json and the checkpoint")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--device", default=None, choices=["auto", "cpu", "cuda"])
    args = ap.parse_args()

    cfg_path = os.path.join(args.output_dir, "config.json")
    if os.path.exists(cfg_path):
        cfg = Config.from_json(cfg_path)
    else:
        raise SystemExit(f"No config.json in {args.output_dir}; run training first.")
    cfg.output_dir = args.output_dir
    if args.device is not None:
        cfg.device = args.device

    evaluate(cfg, checkpoint=args.checkpoint)


if __name__ == "__main__":
    main()
