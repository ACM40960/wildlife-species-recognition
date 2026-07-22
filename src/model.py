"""Transfer-learning model builder.

We start from a CNN pretrained on ImageNet and replace the final classification
head with one sized to our number of species. By default we freeze the stem and
the first two residual blocks (``freeze_until="layer2"``) and **retrain layer3,
layer4 and the head** — the "retrain the later layers" strategy: early
convolutional filters (edges, textures) transfer well, while the deeper,
task-specific layers are adapted to camera-trap imagery.
"""
from __future__ import annotations

from typing import List


_BACKBONES = {"resnet18", "resnet34", "resnet50"}
# ResNet block order from stem to head.
_BLOCK_ORDER = ["conv1", "bn1", "layer1", "layer2", "layer3", "layer4"]


def build_model(backbone: str, num_classes: int, pretrained: bool = True,
                freeze_until: str = "layer2"):
    """Create a ResNet with a fresh classification head.

    Args:
        backbone: one of resnet18 / resnet34 / resnet50.
        num_classes: number of output species.
        pretrained: load ImageNet weights.
        freeze_until: name of the residual block up to and including which
            parameters are frozen. The default ``"layer2"`` freezes
            conv1/bn1/layer1/layer2 and trains layer3/layer4/head. ``""`` trains
            the whole network; ``"all"`` freezes the entire backbone (a linear
            probe on frozen features); ``"layer4"`` freezes the whole backbone
            too (only the head trains).
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
    """
    if freeze_until == "":
        return  # full fine-tuning

    if freeze_until == "all":
        freeze_set = set(_BLOCK_ORDER)
    elif freeze_until in _BLOCK_ORDER:
        freeze_set = set(_BLOCK_ORDER[: _BLOCK_ORDER.index(freeze_until) + 1])
    else:
        raise ValueError(
            f"freeze_until must be '', 'all', or one of {_BLOCK_ORDER}, got {freeze_until!r}")

    for name, module in net.named_children():
        if name in freeze_set:
            for p in module.parameters():
                p.requires_grad = False


def freeze_frozen_batchnorm(net) -> int:
    """Put fully-frozen BatchNorm modules into eval mode.

    A frozen BatchNorm still updates its running mean/variance under
    ``net.train()`` even though its affine parameters don't learn, which quietly
    changes a "frozen" layer. Calling this after ``net.train()`` each epoch keeps
    such layers genuinely fixed. Returns how many BN modules were pinned.
    """
    import torch.nn as nn

    pinned = 0
    for module in net.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            params = list(module.parameters(recurse=False))
            if params and all(not p.requires_grad for p in params):
                module.eval()
                pinned += 1
    return pinned


def trainable_parameter_names(net) -> List[str]:
    return [n for n, p in net.named_parameters() if p.requires_grad]


def count_parameters(net):
    """Return (trainable, total) parameter counts."""
    trainable = sum(p.numel() for p in net.parameters() if p.requires_grad)
    total = sum(p.numel() for p in net.parameters())
    return trainable, total
