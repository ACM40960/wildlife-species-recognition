"""Training / validation loop with early stopping and checkpointing."""
from __future__ import annotations

import os
import time
from typing import Dict

from .data import load_datasets, make_loaders, class_weights
from .model import build_model, count_parameters
from .utils import ensure_dir, set_seed, plot_training_curves


def _run_epoch(net, loader, criterion, optimizer, device, train: bool):
    import torch

    net.train(train)
    total_loss, correct, seen = 0.0, 0, 0
    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            if train:
                optimizer.zero_grad()
            logits = net(images)
            loss = criterion(logits, labels)
            if train:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * images.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
            seen += images.size(0)
    return total_loss / max(seen, 1), correct / max(seen, 1)


def train(cfg) -> Dict:
    """Train a model per ``cfg`` and return a results dict.

    Side effects: writes the best checkpoint, config, training-curve plot, and a
    class-names file into ``cfg.output_dir``.
    """
    import torch
    import torch.nn as nn

    set_seed(cfg.seed)
    device = cfg.resolved_device()
    out = ensure_dir(cfg.output_dir)

    datasets = load_datasets(cfg)
    n_classes = len(datasets.class_names)
    train_loader, val_loader, _ = make_loaders(cfg, datasets)

    net = build_model(cfg.backbone, n_classes, cfg.pretrained, cfg.freeze_until).to(device)
    trainable, total = count_parameters(net)
    print(f"[model] {cfg.backbone}: {trainable:,} trainable / {total:,} total params")
    print(f"[data]  classes={n_classes} "
          f"train={len(datasets.train)} val={len(datasets.val)} test={len(datasets.test)}")

    weights = class_weights(datasets, n_classes).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=cfg.label_smoothing)
    params = [p for p in net.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)

    history = {k: [] for k in ("train_loss", "train_acc", "val_loss", "val_acc")}
    best_val_acc, best_epoch, epochs_no_improve = 0.0, 0, 0
    ckpt_path = os.path.join(out, cfg.checkpoint_name)

    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = _run_epoch(net, train_loader, criterion, optimizer, device, True)
        va_loss, va_acc = _run_epoch(net, val_loader, criterion, optimizer, device, False)
        scheduler.step()

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(va_loss)
        history["val_acc"].append(va_acc)

        improved = va_acc > best_val_acc
        print(f"epoch {epoch:2d}/{cfg.epochs} | "
              f"train loss {tr_loss:.3f} acc {tr_acc:.3f} | "
              f"val loss {va_loss:.3f} acc {va_acc:.3f} | "
              f"{time.time() - t0:.1f}s{'  *best' if improved else ''}")

        if improved:
            best_val_acc, best_epoch, epochs_no_improve = va_acc, epoch, 0
            torch.save({"model_state": net.state_dict(),
                        "class_names": datasets.class_names,
                        "config": cfg.__dict__}, ckpt_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= cfg.early_stop_patience:
                print(f"[early-stop] no val improvement for "
                      f"{cfg.early_stop_patience} epochs; stopping at epoch {epoch}")
                break

    cfg.to_json(os.path.join(out, "config.json"))
    with open(os.path.join(out, "class_names.txt"), "w") as fh:
        fh.write("\n".join(datasets.class_names))
    plot_training_curves(history, os.path.join(out, "training_curves.png"))

    print(f"[done] best val acc {best_val_acc:.3f} at epoch {best_epoch}; "
          f"checkpoint -> {ckpt_path}")
    return {"history": history, "best_val_acc": best_val_acc,
            "best_epoch": best_epoch, "checkpoint": ckpt_path,
            "class_names": datasets.class_names}
