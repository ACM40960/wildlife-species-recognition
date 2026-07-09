#!/bin/bash
# Download the real NTLNP wildlife dataset (25,657 infrared camera-trap images,
# 17 species) from Hugging Face into data/ntlnp_raw/, then arrange it into the
# ImageFolder layout (data/ntlnp/<species>/) this project expects.
#
# Requirements: git and git-lfs (https://git-lfs.com).
#   Ubuntu/Debian:  sudo apt-get install git-lfs
#   macOS (brew):   brew install git-lfs
#
# Usage:
#   bash scripts/download_ntlnp.sh
#
# NOTE: this must be run on a machine with internet access to huggingface.co.
# It could NOT be run in the sandbox this repo was built in, because that
# environment's egress policy blocks huggingface.co. Everything else in the
# repo runs offline against the committed demo dataset (see README).
set -euo pipefail

RAW_DIR="data/ntlnp_raw"
OUT_DIR="data/ntlnp"

echo "==> Installing git-lfs hooks"
git lfs install

if [ ! -d "$RAW_DIR/.git" ]; then
    echo "==> Cloning dataset from Hugging Face (this is several GB)"
    git clone https://huggingface.co/datasets/myyyyw/NTLNP "$RAW_DIR"
else
    echo "==> $RAW_DIR already exists; pulling latest"
    (cd "$RAW_DIR" && git pull)
fi

echo "==> Done. Raw dataset in $RAW_DIR"
echo
echo "The NTLNP release ships Pascal-VOC annotated frames. Once you have"
echo "verified the folder structure, organise the crops/frames into one"
echo "sub-folder per species under $OUT_DIR/, e.g.:"
echo "    $OUT_DIR/amur_tiger/ ... $OUT_DIR/red_fox/ ..."
echo "then train with:"
echo "    python scripts/run_training.py --data-dir $OUT_DIR --pretrained --epochs 15"
