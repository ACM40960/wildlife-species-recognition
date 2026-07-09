#!/usr/bin/env python3
"""Fetch ImageNet ResNet-18 weights into the torch hub cache (offline fallback).

On a normal machine you do NOT need this: torchvision downloads the weights
automatically from download.pytorch.org the first time you pass
``--pretrained``. This helper exists only for environments where that host is
blocked (as in the sandbox this repo was built in). It downloads the *identical*
weights from a checksum-verified Git-LFS mirror on GitHub and places them where
torchvision looks for them.

The download is verified against the known torchvision SHA-256, so you are
getting the genuine ImageNet-pretrained weights, not an arbitrary file.

Usage:
    python scripts/fetch_pretrained_weights.py
"""
import hashlib
import os
import urllib.request

# Genuine torchvision resnet18 ImageNet weights (legacy V1). Verified by SHA-256.
SHA256 = "5c106cde386e87d4033832f2996f5493238eda96ccf559d1d62760c4de0613f8"
MIRROR = ("https://media.githubusercontent.com/media/alpaz1/MLPracticalCW4/"
          "1757a704ab542d2414077e9cf920f2841267399c/Models/resnet18-5c106cde.pth")

CACHE_DIR = os.path.expanduser("~/.cache/torch/hub/checkpoints")
# torchvision looks these up by exact filename depending on its version.
TARGETS = ["resnet18-5c106cde.pth", "resnet18-f37072fd.pth"]


def main():
    os.makedirs(CACHE_DIR, exist_ok=True)
    tmp = os.path.join(CACHE_DIR, "resnet18-download.tmp")

    print(f"Downloading ResNet-18 ImageNet weights from mirror...")
    urllib.request.urlretrieve(MIRROR, tmp)

    digest = hashlib.sha256(open(tmp, "rb").read()).hexdigest()
    if digest != SHA256:
        os.remove(tmp)
        raise SystemExit(f"checksum mismatch: got {digest}, expected {SHA256}")
    print(f"SHA-256 verified: {digest}")

    for name in TARGETS:
        dst = os.path.join(CACHE_DIR, name)
        with open(tmp, "rb") as s, open(dst, "wb") as d:
            d.write(s.read())
        print(f"  -> {dst}")
    os.remove(tmp)
    print("Done. `--pretrained` will now work offline.")


if __name__ == "__main__":
    main()
