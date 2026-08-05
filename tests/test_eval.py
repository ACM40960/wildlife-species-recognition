"""Tests for checkpoint validation and evaluation statistics."""
import numpy as np
import pytest

from src.config import Config
from src.evaluate import (wilson_ci, bootstrap_ci, expected_calibration_error,
                          topk_accuracy, _validate_checkpoint)


# ----- checkpoint validation (issues 20 & 21) -----
def _state(class_names, **cfg_over):
    cfg = {"backbone": "resnet18", "image_size": 224, "grayscale_to_rgb": True}
    cfg.update(cfg_over)
    return {"class_names": class_names, "config": cfg}


def test_validate_checkpoint_accepts_match():
    cfg = Config()
    _validate_checkpoint(_state(["a", "b", "c"]), cfg, ["a", "b", "c"])  # no raise


def test_validate_checkpoint_rejects_class_order():
    cfg = Config()
    with pytest.raises(ValueError):
        _validate_checkpoint(_state(["a", "b", "c"]), cfg, ["c", "b", "a"])


def test_validate_checkpoint_rejects_config_mismatch():
    cfg = Config()
    cfg.image_size = 128
    with pytest.raises(ValueError):
        _validate_checkpoint(_state(["a", "b"], image_size=224), cfg, ["a", "b"])


def test_validate_checkpoint_requires_class_names():
    with pytest.raises(ValueError):
        _validate_checkpoint({"config": {}}, Config(), ["a"])


# ----- statistics (issue 22 & 23) -----
def test_wilson_ci_brackets_point_estimate():
    lo, hi = wilson_ci(8, 10)
    assert 0 <= lo <= 0.8 <= hi <= 1
    # zero trials is well defined
    assert wilson_ci(0, 0) == (0.0, 0.0)


def test_wilson_ci_shrinks_with_more_data():
    narrow = wilson_ci(80, 100)
    wide = wilson_ci(8, 10)
    assert (narrow[1] - narrow[0]) < (wide[1] - wide[0])


def test_bootstrap_ci_contains_metric():
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 3, 200)
    y_pred = y_true.copy()
    flip = rng.random(200) < 0.2
    y_pred[flip] = (y_pred[flip] + 1) % 3
    from sklearn.metrics import f1_score
    point = f1_score(y_true, y_pred, average="macro")
    lo, hi = bootstrap_ci(y_true, y_pred,
                          lambda a, b: f1_score(a, b, average="macro"), n_boot=300)
    assert lo <= point <= hi


def test_topk_accuracy_monotonic():
    probs = np.array([[0.5, 0.3, 0.2], [0.1, 0.2, 0.7]])
    y = [1, 2]
    top1 = topk_accuracy(probs, y, 1)
    top2 = topk_accuracy(probs, y, 2)
    assert 0 <= top1 <= top2 <= 1
    # argmax predictions are [0, 2]; against y=[1, 2] only the 2nd is correct.
    assert topk_accuracy(probs, [1, 2], 1) == 0.5
    # top-2 for row 0 is {0,1} which now includes true label 1 => both correct.
    assert topk_accuracy(probs, [1, 2], 2) == 1.0


def test_ece_perfectly_calibrated_is_low():
    # confidence == accuracy in each region => ECE ~ 0
    conf = np.array([1.0] * 50 + [0.0] * 50)
    correct = np.array([1] * 50 + [0] * 50)
    assert expected_calibration_error(conf, correct) < 1e-6
    # over-confident wrong predictions => ECE high
    conf2 = np.array([0.9] * 10)
    correct2 = np.array([0] * 10)
    assert expected_calibration_error(conf2, correct2) > 0.8
