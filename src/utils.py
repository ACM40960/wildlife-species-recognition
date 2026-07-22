"""Utility helpers: reproducibility, plotting, and small conveniences."""
from __future__ import annotations

import os
import random
from typing import Sequence

import numpy as np


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and (if available) PyTorch for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def enable_determinism() -> None:
    """Ask PyTorch for deterministic algorithms where it can provide them.

    ``warn_only=True`` means ops without a deterministic implementation warn
    rather than crash, so training still runs while being as reproducible as the
    backend allows. Also disables cuDNN autotuning (a source of run-to-run
    variation) and sets the cuBLAS workspace required for determinism on CUDA.
    """
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    try:
        import torch
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass


def environment_info() -> dict:
    """Record versions needed to reproduce a run (issue: reproducibility)."""
    import platform

    info = {
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    try:
        import torch
        info["torch"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        info["cuda"] = getattr(torch.version, "cuda", None)
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
    except Exception:
        info["torch"] = None
    try:
        import torchvision
        info["torchvision"] = torchvision.__version__
    except Exception:
        info["torchvision"] = None
    try:
        info["numpy"] = np.__version__
        import sklearn
        info["scikit_learn"] = sklearn.__version__
    except Exception:
        pass
    return info


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def plot_training_curves(history: dict, out_path: str) -> None:
    """Save train/val loss and accuracy curves to a PNG."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

    ax1.plot(epochs, history["train_loss"], label="train")
    ax1.plot(epochs, history["val_loss"], label="val")
    ax1.set_title("Loss")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("cross-entropy")
    ax1.legend()

    ax2.plot(epochs, history["train_acc"], label="train")
    ax2.plot(epochs, history["val_acc"], label="val")
    ax2.set_title("Accuracy")
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("accuracy")
    ax2.set_ylim(0, 1)
    ax2.legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_confusion_matrix(cm: np.ndarray, class_names: Sequence[str], out_path: str) -> None:
    """Save a normalised confusion matrix heatmap to a PNG."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cm = np.asarray(cm, dtype=float)
    row_sums = cm.sum(axis=1, keepdims=True)
    norm = np.divide(cm, row_sums, out=np.zeros_like(cm), where=row_sums != 0)

    n = len(class_names)
    fig, ax = plt.subplots(figsize=(max(5, n * 0.7), max(4, n * 0.7)))
    im = ax.imshow(norm, cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title("Confusion matrix (row-normalised)")

    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{norm[i, j]:.2f}", ha="center", va="center",
                    color="white" if norm[i, j] < 0.5 else "black", fontsize=8)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
