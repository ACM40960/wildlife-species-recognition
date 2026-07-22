"""Evaluation: metrics with confidence intervals, calibration, seen-vs-unseen,
and visual error analysis.

Highlights
----------
* Validates that the checkpoint's class order and configuration match the current
  dataset before scoring (a silent mismatch would report nonsense).
* Reports accuracy with a Wilson interval, macro precision/recall/F1 and balanced
  accuracy with bootstrap intervals, top-k accuracy, and expected calibration
  error — the test set is small (~30/species), so the intervals matter.
* Scores the unseen-location test set and, when available, a held-out
  seen-location set, so the generalisation gap is explicit.
* Saves representative correct / false-positive / false-negative examples.
"""
from __future__ import annotations

import json
import math
import os
from typing import Dict

import numpy as np

from .data import load_datasets
from .model import build_model
from .utils import ensure_dir, set_seed, plot_confusion_matrix


# --------------------------------------------------------------------------- #
# Small statistics helpers
# --------------------------------------------------------------------------- #
def wilson_ci(k: int, n: int, z: float = 1.96):
    """Wilson score interval for a binomial proportion k/n."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def bootstrap_ci(y_true, y_pred, metric_fn, n_boot: int = 2000, seed: int = 0):
    """Percentile bootstrap 95% CI for a metric over (y_true, y_pred)."""
    rng = np.random.default_rng(seed)
    yt, yp = np.asarray(y_true), np.asarray(y_pred)
    n = len(yt)
    if n == 0:
        return (0.0, 0.0)
    stats = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        stats[b] = metric_fn(yt[idx], yp[idx])
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return (float(lo), float(hi))


def expected_calibration_error(confidences, correct, n_bins: int = 10) -> float:
    """ECE: average gap between confidence and accuracy across confidence bins."""
    confidences, correct = np.asarray(confidences), np.asarray(correct, dtype=float)
    if len(confidences) == 0:
        return 0.0
    bins = np.linspace(0, 1, n_bins + 1)
    ece, N = 0.0, len(confidences)
    for i in range(n_bins):
        m = (confidences > bins[i]) & (confidences <= bins[i + 1])
        if m.sum() > 0:
            ece += m.sum() / N * abs(correct[m].mean() - confidences[m].mean())
    return float(ece)


def topk_accuracy(probs, y_true, k: int) -> float:
    probs, y_true = np.asarray(probs), np.asarray(y_true)
    if len(y_true) == 0 or k > probs.shape[1]:
        return float("nan")
    topk = np.argsort(probs, axis=1)[:, -k:]
    return float(np.mean([yt in row for yt, row in zip(y_true, topk)]))


# --------------------------------------------------------------------------- #
# Prediction collection & checkpoint validation
# --------------------------------------------------------------------------- #
def _collect(net, loader, device):
    import torch

    net.eval()
    y_true, y_pred, probs = [], [], []
    with torch.no_grad():
        for images, labels in loader:
            p = torch.softmax(net(images.to(device)), dim=1).cpu().numpy()
            probs.append(p)
            y_pred.extend(p.argmax(1).tolist())
            y_true.extend(labels.tolist())
    return y_true, y_pred, (np.concatenate(probs) if probs else np.empty((0, 0)))


def _validate_checkpoint(state, cfg, dataset_class_names):
    """Fail loudly if the checkpoint doesn't match the current dataset/config."""
    ckpt_classes = state.get("class_names")
    if ckpt_classes is None:
        raise ValueError("checkpoint has no class_names; cannot validate.")
    if list(ckpt_classes) != list(dataset_class_names):
        raise ValueError(
            "class order mismatch between checkpoint and dataset:\n"
            f"  checkpoint: {list(ckpt_classes)}\n"
            f"  dataset:    {list(dataset_class_names)}\n"
            "Re-evaluate against the dataset the checkpoint was trained on.")
    ck = state.get("config", {}) or {}
    for field in ("backbone", "image_size", "grayscale_to_rgb"):
        if field in ck and getattr(cfg, field) != ck[field]:
            raise ValueError(
                f"config mismatch on '{field}': checkpoint={ck[field]!r}, "
                f"current={getattr(cfg, field)!r}. Use the checkpoint's settings.")
    if len(ckpt_classes) != len(dataset_class_names):
        raise ValueError("number of classes differs between checkpoint and dataset.")
    return len(ckpt_classes)


# --------------------------------------------------------------------------- #
# Metric bundle for one split
# --------------------------------------------------------------------------- #
def _metrics_for(y_true, y_pred, probs, class_names, seed):
    from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                                 precision_recall_fscore_support, f1_score)

    n = len(y_true)
    acc = accuracy_score(y_true, y_pred)
    n_correct = int(round(acc * n))
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    p_macro, r_macro, f_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0)

    correct = (np.asarray(y_true) == np.asarray(y_pred)).astype(float)
    conf = probs.max(1) if probs.size else np.array([])

    return {
        "n": n,
        "accuracy": acc,
        "accuracy_wilson_95ci": wilson_ci(n_correct, n),
        "balanced_accuracy": bal_acc,
        "f1_macro": f_macro,
        "f1_macro_boot_95ci": bootstrap_ci(
            y_true, y_pred,
            lambda a, b: f1_score(a, b, average="macro", zero_division=0), seed=seed),
        "precision_macro": p_macro, "recall_macro": r_macro,
        "top2_accuracy": topk_accuracy(probs, y_true, 2),
        "top3_accuracy": topk_accuracy(probs, y_true, 3),
        "expected_calibration_error": expected_calibration_error(conf, correct),
        "mean_confidence": float(conf.mean()) if conf.size else float("nan"),
    }


def _per_class(y_true, y_pred, class_names):
    from sklearn.metrics import precision_recall_fscore_support

    p, r, f, s = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(len(class_names))), zero_division=0)
    out = {}
    for i, name in enumerate(class_names):
        # Wilson CI on recall = correctly-classified fraction among true class i.
        out[name] = {
            "precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]),
            "support": int(s[i]),
            "recall_wilson_95ci": wilson_ci(int(round(r[i] * s[i])), int(s[i])),
        }
    return out


# --------------------------------------------------------------------------- #
# Visual error analysis (issue 24)
# --------------------------------------------------------------------------- #
def _save_error_examples(dataset, y_true, y_pred, probs, class_names, out_path,
                         max_each=6):
    """Save a montage of correct / false-positive / false-negative examples."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image
    from .data import _parse_bbox

    rows = getattr(dataset, "rows", None)
    if rows is None:
        return None
    conf = probs.max(1)

    correct_idx = [i for i in range(len(y_true)) if y_true[i] == y_pred[i]]
    wrong_idx = [i for i in range(len(y_true)) if y_true[i] != y_pred[i]]
    # most-confident correct, and most-confident *wrong* (the instructive errors)
    correct_idx.sort(key=lambda i: -conf[i])
    wrong_idx.sort(key=lambda i: -conf[i])
    picks = ([("correct", i) for i in correct_idx[:max_each]] +
             [("error", i) for i in wrong_idx[:max_each]])
    if not picks:
        return None

    def load(i):
        r = rows[i]
        img = Image.open(os.path.join(dataset.data_dir, r["filename"])).convert("L")
        box = _parse_bbox(r.get("bbox", "")) if dataset.crop_to_bbox else None
        if box:
            x, y, w, h = box
            W, H = img.size
            img = img.crop((max(0, x), max(0, y), min(W, x + w), min(H, y + h)))
        return img

    cols = max_each
    rows_n = 2
    fig, axes = plt.subplots(rows_n, cols, figsize=(cols * 2, rows_n * 2.4))
    axes = np.atleast_2d(axes)
    for ax in axes.ravel():
        ax.axis("off")
    for pos, (kind, i) in enumerate(picks):
        r, c = divmod(pos, cols)
        if r >= rows_n:
            break
        ax = axes[r][c]
        ax.imshow(load(i), cmap="gray")
        ax.axis("off")
        t, p = class_names[y_true[i]], class_names[y_pred[i]]
        if kind == "correct":
            ax.set_title(f"{t}\n({conf[i]:.2f})", fontsize=8, color="green")
        else:
            ax.set_title(f"true {t}\npred {p} ({conf[i]:.2f})", fontsize=8, color="red")
    axes[0][0].set_ylabel("most-confident correct")
    fig.suptitle("Top: confident correct predictions   Bottom: confident errors",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def evaluate(cfg, checkpoint: str | None = None) -> Dict:
    """Score the checkpoint on the unseen-location test set (and seen-location set
    if present). Writes metrics.json, confusion_matrix.png and error_analysis.png.
    """
    import torch
    from sklearn.metrics import confusion_matrix

    set_seed(cfg.seed)
    device = cfg.resolved_device()
    out = ensure_dir(cfg.output_dir)
    checkpoint = checkpoint or os.path.join(out, cfg.checkpoint_name)

    datasets = load_datasets(cfg)
    class_names = datasets.class_names
    n_classes = len(class_names)

    state = torch.load(checkpoint, map_location=device)
    _validate_checkpoint(state, cfg, class_names)   # issues 20 & 21

    net = build_model(cfg.backbone, n_classes, pretrained=False, freeze_until="").to(device)
    net.load_state_dict(state["model_state"])

    from torch.utils.data import DataLoader
    common = dict(batch_size=cfg.batch_size, num_workers=cfg.num_workers)
    test_loader = DataLoader(datasets.test, shuffle=False, **common)

    y_true, y_pred, probs = _collect(net, test_loader, device)
    unseen = _metrics_for(y_true, y_pred, probs, class_names, cfg.seed)
    per_class = _per_class(y_true, y_pred, class_names)

    cm = confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))
    plot_confusion_matrix(cm, class_names, os.path.join(out, "confusion_matrix.png"))
    _save_error_examples(datasets.test, y_true, y_pred, probs, class_names,
                         os.path.join(out, "error_analysis.png"))

    metrics = {
        "split_by": getattr(cfg, "split_by", "location"),
        "crop_to_bbox": getattr(cfg, "crop_to_bbox", True),
        "class_names": class_names,
        "unseen_locations": unseen,       # the honest held-out result
        "per_class": per_class,
    }

    # Seen-location test (issue 23): same model, backgrounds it has seen.
    if datasets.seen_test is not None and len(datasets.seen_test) > 0:
        seen_loader = DataLoader(datasets.seen_test, shuffle=False, **common)
        st, sp, spr = _collect(net, seen_loader, device)
        seen = _metrics_for(st, sp, spr, class_names, cfg.seed)
        metrics["seen_locations"] = seen
        metrics["seen_minus_unseen_accuracy"] = seen["accuracy"] - unseen["accuracy"]

    with open(os.path.join(out, "metrics.json"), "w") as fh:
        json.dump(metrics, fh, indent=2, default=list)

    lo, hi = unseen["accuracy_wilson_95ci"]
    print(f"[test/unseen] n={unseen['n']}  accuracy={unseen['accuracy']:.3f} "
          f"(95% CI {lo:.3f}-{hi:.3f})  balanced={unseen['balanced_accuracy']:.3f}")
    f_lo, f_hi = unseen["f1_macro_boot_95ci"]
    print(f"              macro-F1={unseen['f1_macro']:.3f} (95% CI {f_lo:.3f}-{f_hi:.3f})  "
          f"top-2={unseen['top2_accuracy']:.3f}  ECE={unseen['expected_calibration_error']:.3f}")
    if "seen_locations" in metrics:
        s = metrics["seen_locations"]
        print(f"[test/seen]   n={s['n']}  accuracy={s['accuracy']:.3f}  "
              f"=> seen-unseen gap {metrics['seen_minus_unseen_accuracy']:+.3f}")
    return metrics
