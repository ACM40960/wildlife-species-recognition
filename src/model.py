"""Transfer-learning model builder.

We start from a CNN pretrained on ImageNet, replace the final classification
head with one sized to our number of species, and (by default) freeze all
layers except the last residual block and the head. This is the "retrain the
later layers" strategy described in the project approach: early convolutional
filters (edges, textures) transfer well, while the deeper, task-specific layers
are adapted to camera-trap imagery.
"""
from __future__ import annotations

from typing import List


_BACKBONES = {"resnet18", "resnet34", "resnet50"}


def build_model(backbone: str, num_classes: int, pretrained: bool = True,
                freeze_until: str = "layer4"):
    """Create a ResNet with a fresh classification head.

    Args:
        backbone: one of resnet18 / resnet34 / resnet50.
        num_classes: number of output species.
        pretrained: load ImageNet weights.
        freeze_until: name of the residual block up to which parameters are
            frozen. "" trains the whole network; "all" freezes the entire
            backbone (a linear probe on frozen features).
    """
    import torch.nn as nn
    from torchvision import models

    if backbone not in _BACKBONES:
        raise ValueError(f"backbone must be one of {_BACKBONES}, got {backbone!r}")

    weights = "IMAGENET1K_V1" if pretrained else None
    net = getattr(models, backbone)(weights=weights)

    _apply_freezing(net, freeze_until)

    # Replace the head. ResNet's final layer is ``fc``.
    in_features = net.fc.in_features
    net.fc = nn.Linear(in_features, num_classes)  # new head is always trainable
    return net


def _apply_freezing(net, freeze_until: str) -> None:
    """Freeze parameters from the stem up to (and including) ``freeze_until``.

    ResNet block order: conv1, bn1, layer1, layer2, layer3, layer4, fc.
    Everything before the *next* block after ``freeze_until`` is frozen.
    """
    if freeze_until == "":
        return  # full fine-tuning

    order = ["conv1", "bn1", "layer1", "layer2", "layer3", "layer4"]

    if freeze_until == "all":
        freeze_set = set(order)
    elif freeze_until in order:
        freeze_set = set(order[: order.index(freeze_until) + 1])
    else:
        raise ValueError(
            f"freeze_until must be '', 'all', or one of {order}, got {freeze_until!r}")

    for name, module in net.named_children():
        if name in freeze_set:
            for p in module.parameters():
                p.requires_grad = False


def trainable_parameter_names(net) -> List[str]:
    return [n for n, p in net.named_parameters() if p.requires_grad]


def count_parameters(net):
    """Return (trainable, total) parameter counts."""
    trainable = sum(p.numel() for p in net.parameters() if p.requires_grad)
    total = sum(p.numel() for p in net.parameters())
    return trainable, total
