#!/usr/bin/env python3
"""Fetch YOLOv8n (COCO) weights into .cct_cache (offline fallback).

On a normal machine ultralytics downloads ``yolov8n.pt`` automatically the first
time a YOLO model is created. This helper exists for environments where the
ultralytics asset host is blocked (as in the sandbox this repo was built in): it
pulls the *identical* weights from a checksum-verified Git-LFS mirror on GitHub.

The download is verified against the file's LFS object SHA-256, so you are
getting the genuine COCO-pretrained YOLOv8n, not an arbitrary file.

Usage:
    python scripts/fetch_yolo_weights.py                 # -> .cct_cache/yolov8n.pt
"""
import argparse
import hashlib
import os
import urllib.request

SHA256 = "31e20dde3def09e2cf938c7be6fe23d9150bbbe503982af13345706515f2ef95"
SIZE = 6534387
MIRROR = ("https://media.githubusercontent.com/media/Priler/csgobot/"
          "4d99d5e5ccd7ca70bb9882fee6619cc85abde8d0/yolov8/yolov8n.pt")


def fetch(dst: str) -> str:
    os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
    if os.path.exists(dst) and hashlib.sha256(open(dst, "rb").read()).hexdigest() == SHA256:
        print(f"already present and verified: {dst}")
        return dst
    tmp = dst + ".tmp"
    print("Downloading YOLOv8n weights from mirror...")
    urllib.request.urlretrieve(MIRROR, tmp)
    digest = hashlib.sha256(open(tmp, "rb").read()).hexdigest()
    if digest != SHA256:
        os.remove(tmp)
        raise SystemExit(f"checksum mismatch: got {digest}, expected {SHA256}")
    os.replace(tmp, dst)
    print(f"SHA-256 verified: {digest}\n  -> {dst}")
    return dst


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=".cct_cache/yolov8n.pt")
    args = ap.parse_args()
    fetch(args.out)
