"""Tests for preprocessing transforms and model freezing."""
import pytest

from src.data import build_transforms
from src.model import (build_model, _apply_freezing, count_parameters,
                       freeze_frozen_batchnorm)

PIL = pytest.importorskip("PIL")
torch = pytest.importorskip("torch")
from PIL import Image


def _dummy_image(size=300):
    return Image.new("L", (size, size), color=120).convert("RGB")


def test_transforms_output_shape_and_grayscale():
    tf = build_transforms(image_size=224, grayscale_to_rgb=True, train=False)
    out = tf(_dummy_image())
    assert out.shape == (3, 224, 224)
    # grayscale->RGB means the 3 channels are identical after Grayscale(3),
    # before normalisation makes them differ per-channel; check pre-norm path via
    # a train transform still yields 3 channels.
    tf_train = build_transforms(224, True, train=True)
    assert tf_train(_dummy_image()).shape == (3, 224, 224)


def test_transforms_respect_image_size():
    tf = build_transforms(image_size=96, grayscale_to_rgb=True, train=False)
    assert tf(_dummy_image()).shape == (3, 96, 96)


def test_freeze_default_trains_later_layers():
    # Default freeze_until="layer2" must train layer3/layer4/fc (not just fc).
    net = build_model("resnet18", num_classes=6, pretrained=False)
    trainable = {n.split(".")[0] for n, p in net.named_parameters() if p.requires_grad}
    assert "layer3" in trainable and "layer4" in trainable and "fc" in trainable
    assert "layer1" not in trainable and "conv1" not in trainable


def test_freeze_all_is_linear_probe():
    net = build_model("resnet18", num_classes=6, pretrained=False, freeze_until="all")
    trainable = {n.split(".")[0] for n, p in net.named_parameters() if p.requires_grad}
    assert trainable == {"fc"}      # only the head trains


def test_freeze_empty_trains_everything():
    net = build_model("resnet18", num_classes=6, pretrained=False, freeze_until="")
    assert all(p.requires_grad for p in net.parameters())


def test_freeze_until_invalid_raises():
    net = build_model("resnet18", num_classes=6, pretrained=False, freeze_until="")
    with pytest.raises(ValueError):
        _apply_freezing(net, "not_a_block")


def test_frozen_batchnorm_goes_eval():
    net = build_model("resnet18", num_classes=6, pretrained=False, freeze_until="layer2")
    net.train()                      # everything to train mode first
    pinned = freeze_frozen_batchnorm(net)
    assert pinned > 0                # some frozen BN layers exist
    # A frozen BN (in layer1) must be in eval; a trainable one (layer4) in train.
    assert net.layer1[0].bn1.training is False
    assert net.layer4[0].bn1.training is True


def test_count_parameters_head_replaced():
    net = build_model("resnet18", num_classes=6, pretrained=False)
    trainable, total = count_parameters(net)
    assert 0 < trainable < total
    assert net.fc.out_features == 6
