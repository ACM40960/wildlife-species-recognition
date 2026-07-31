"""End-to-end smoke test: build a tiny dataset, train 1 epoch, evaluate.

Runs entirely offline (no ImageNet weights, no dataset download) so CI can verify
the whole pipeline wires together and produces its outputs.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

pytest.importorskip("torch")
pytest.importorskip("torchvision")

from src.config import Config
from src.train import train
from src.evaluate import evaluate
from scripts.make_tiny_dataset import make


def test_train_and_eval_smoke(tmp_path):
    data_dir = tmp_path / "tiny"
    make(str(data_dir), image_size=64)

    cfg = Config()
    cfg.data_dir = str(data_dir)
    cfg.image_size = 64
    cfg.epochs = 1
    cfg.batch_size = 8
    cfg.pretrained = False          # no weight download
    cfg.freeze_until = ""           # train the whole tiny net
    cfg.device = "cpu"
    cfg.num_workers = 0
    cfg.output_dir = str(tmp_path / "out")
    cfg.seed = 0

    result = train(cfg)
    assert os.path.exists(result["checkpoint"])
    for f in ("history.json", "history.csv", "environment.json", "config.json"):
        assert os.path.exists(os.path.join(cfg.output_dir, f))

    metrics = evaluate(cfg, checkpoint=result["checkpoint"])
    assert "unseen_locations" in metrics
    acc = metrics["unseen_locations"]["accuracy"]
    assert 0.0 <= acc <= 1.0
    assert os.path.exists(os.path.join(cfg.output_dir, "confusion_matrix.png"))
