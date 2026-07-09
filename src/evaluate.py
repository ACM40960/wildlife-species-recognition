"""Evaluation: accuracy, precision, recall, confusion matrix, baseline plot."""
from __future__ import annotations

import json
import os
from typing import Dict

from .data import load_datasets, make_loaders
from .model import build_model
from .utils import (ensure_dir, set_seed, plot_confusion_matrix,
                    plot_baseline_comparison)


def _collect_predictions(net, loader, device):
    import torch

    net.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            preds = net(images).argmax(1).cpu()
            y_true.extend(labels.tolist())
            y_pred.extend(preds.tolist())
    return y_true, y_pred


def evaluate(cfg, checkpoint: str | None = None) -> Dict:
    """Load the best checkpoint and score it on the held-out test split.

    Writes ``metrics.json``, ``confusion_matrix.png``, and
    ``baseline_comparison.png`` into ``cfg.output_dir`` and returns the metrics.
    """
    import torch
    from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                                 confusion_matrix, classification_report)

    set_seed(cfg.seed)
    device = cfg.resolved_device()
    out = ensure_dir(cfg.output_dir)
    checkpoint = checkpoint or os.path.join(out, cfg.checkpoint_name)

    datasets = load_datasets(cfg)
    class_names = datasets.class_names
    n_classes = len(class_names)
    _, _, test_loader = make_loaders(cfg, datasets)

    net = build_model(cfg.backbone, n_classes, pretrained=False,
                      freeze_until="").to(device)
    state = torch.load(checkpoint, map_location=device)
    net.load_state_dict(state["model_state"])

    y_true, y_pred = _collect_predictions(net, test_loader, device)

    acc = accuracy_score(y_true, y_pred)
    # Macro = unweighted mean over classes (treats rare species equally);
    # weighted accounts for support. We report both.
    p_macro, r_macro, f_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0)
    p_w, r_w, f_w, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0)

    cm = confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))
    report = classification_report(y_true, y_pred, labels=list(range(n_classes)),
                                   target_names=class_names, zero_division=0,
                                   output_dict=True)

    metrics = {
        "test_accuracy": acc,
        "precision_macro": p_macro, "recall_macro": r_macro, "f1_macro": f_macro,
        "precision_weighted": p_w, "recall_weighted": r_w, "f1_weighted": f_w,
        "n_test": len(y_true), "n_classes": n_classes,
        "class_names": class_names,
        "per_class": report,
        "baselines": cfg.baselines,
    }

    with open(os.path.join(out, "metrics.json"), "w") as fh:
        json.dump(metrics, fh, indent=2)
    plot_confusion_matrix(cm, class_names, os.path.join(out, "confusion_matrix.png"))
    plot_baseline_comparison(acc, cfg.baselines,
                             os.path.join(out, "baseline_comparison.png"))

    print(f"[test] n={len(y_true)}  accuracy={acc:.3f}")
    print(f"       precision(macro)={p_macro:.3f} recall(macro)={r_macro:.3f} "
          f"f1(macro)={f_macro:.3f}")
    print(f"       precision(w)={p_w:.3f} recall(w)={r_w:.3f} f1(w)={f_w:.3f}")
    for name, base in cfg.baselines.items():
        delta = acc - base
        print(f"       vs {name}: {delta:+.3f}")
    return metrics
