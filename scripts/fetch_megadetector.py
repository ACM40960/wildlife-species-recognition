#!/usr/bin/env python3
"""Download MegaDetector v5a weights (checksum-verified).

MegaDetector is the standard camera-trap detector — trained on camera-trap frames
including infrared night captures, with classes animal/person/vehicle. We use it
to locate the animal in frames that have no ground-truth bounding box.

The weights are ~280MB and are released by the MegaDetector project on GitHub.

Usage:
    pip install yolov5 "setuptools<81"
    python scripts/fetch_megadetector.py
"""
import argparse
import hashlib
import os
import urllib.request

URL = ("https://github.com/agentmorris/MegaDetector/releases/download/v5.0/"
       "md_v5a.0.0.pt")
SHA256 = "94e88fe97c8050f2e3d0cc4cb4f64729d639d74312dcbe2f74f8eecd3b01b276"


def fetch(dst: str) -> str:
    os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
    if os.path.exists(dst):
        if hashlib.sha256(open(dst, "rb").read()).hexdigest() == SHA256:
            print(f"already present and verified: {dst}")
            return dst
        print("existing file failed checksum; re-downloading")
    tmp = dst + ".tmp"
    print(f"Downloading MegaDetector v5a (~280MB) from {URL} ...")
    urllib.request.urlretrieve(URL, tmp)
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
    ap.add_argument("--out", default="src/../.cct_cache/md_v5a.0.0.pt")
    args = ap.parse_args()
    fetch(os.path.normpath(args.out))
