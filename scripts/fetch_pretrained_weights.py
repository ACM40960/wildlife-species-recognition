#!/usr/bin/env python3
"""Fetch ImageNet ResNet weights into the torch hub cache (offline fallback).

On a normal machine you do NOT need this: torchvision downloads the weights
automatically from download.pytorch.org the first time you pass ``--pretrained``.
This helper exists only for environments where that host is blocked (as in the
sandbox this repo was built in): it fetches the *identical* weights from a
checksum-verified Git-LFS mirror and places them where torchvision looks.

Scope: only **resnet18** has a verified mirror configured here. resnet34 and
resnet50 are offered by the training CLI, but there is no vetted offline mirror
for them in this repo, so this script fails clearly for those rather than
pulling an unverified file. On a machine with normal internet access, just use
``--pretrained`` and torchvision fetches whichever backbone you chose.

Usage:
    python scripts/fetch_pretrained_weights.py                 # resnet18
    python scripts/fetch_pretrained_weights.py --backbone resnet34   # errors clearly
"""
import argparse
import hashlib
import os
import urllib.request

# backbone -> (sha256, mirror_url, [torchvision cache filenames]).
# mirror_url is None where no verified offline mirror is available.
REGISTRY = {
    "resnet18": (
        "5c106cde386e87d4033832f2996f5493238eda96ccf559d1d62760c4de0613f8",
        "https://media.githubusercontent.com/media/alpaz1/MLPracticalCW4/"
        "1757a704ab542d2414077e9cf920f2841267399c/Models/resnet18-5c106cde.pth",
        ["resnet18-5c106cde.pth", "resnet18-f37072fd.pth"],
    ),
    "resnet34": (None, None, ["resnet34-b627a593.pth"]),
    "resnet50": (None, None, ["resnet50-0676ba61.pth", "resnet50-11ad3fa6.pth"]),
}

CACHE_DIR = os.path.expanduser("~/.cache/torch/hub/checkpoints")


def fetch(backbone: str) -> None:
    if backbone not in REGISTRY:
        raise SystemExit(f"unknown backbone {backbone!r}; choose from {list(REGISTRY)}")
    sha256, mirror, targets = REGISTRY[backbone]
    if mirror is None:
        raise SystemExit(
            f"No verified offline mirror is configured for {backbone}.\n"
            f"Only resnet18 can be fetched offline by this script. Options:\n"
            f"  - run on a machine with access to download.pytorch.org and just "
            f"use --pretrained (torchvision downloads {backbone} automatically), or\n"
            f"  - use --backbone resnet18, or\n"
            f"  - add a checksum-verified mirror for {backbone} to REGISTRY.")

    os.makedirs(CACHE_DIR, exist_ok=True)
    tmp = os.path.join(CACHE_DIR, f"{backbone}-download.tmp")
    print(f"Downloading {backbone} ImageNet weights from mirror...")
    urllib.request.urlretrieve(mirror, tmp)

    digest = hashlib.sha256(open(tmp, "rb").read()).hexdigest()
    if digest != sha256:
        os.remove(tmp)
        raise SystemExit(f"checksum mismatch: got {digest}, expected {sha256}")
    print(f"SHA-256 verified: {digest}")

    for name in targets:
        dst = os.path.join(CACHE_DIR, name)
        with open(tmp, "rb") as s, open(dst, "wb") as d:
            d.write(s.read())
        print(f"  -> {dst}")
    os.remove(tmp)
    print(f"Done. `--pretrained --backbone {backbone}` will now work offline.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backbone", default="resnet18",
                    choices=["resnet18", "resnet34", "resnet50"])
    args = ap.parse_args()
    fetch(args.backbone)
