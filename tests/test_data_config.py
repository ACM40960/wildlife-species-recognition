"""Tests for load-time bbox cropping and Config JSON round-tripping."""
import pytest

from src.config import Config

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")
from PIL import Image
from torchvision import transforms

from src.data import ManifestDataset


def _tiny_transform():
    # deterministic, no augmentation: grayscale -> 16x16 -> tensor in [0, 1]
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((16, 16)),
        transforms.ToTensor(),
    ])


def test_crop_to_bbox_changes_the_loaded_image(tmp_path):
    # Black 100x100 frame with a white 20x20 block at (10, 10); bbox around it.
    img = Image.new("L", (100, 100), color=0)
    for y in range(10, 30):
        for x in range(10, 30):
            img.putpixel((x, y), 255)
    cdir = tmp_path / "x"
    cdir.mkdir()
    img.save(cdir / "img.jpg")

    rows = [{"class": "x", "filename": "x/img.jpg", "bbox": "10;10;20;20",
             "image_id": "id0"}]
    tf = _tiny_transform()
    cropped = ManifestDataset(rows, str(tmp_path), {"x": 0}, tf, crop_to_bbox=True)[0][0]
    full = ManifestDataset(rows, str(tmp_path), {"x": 0}, tf, crop_to_bbox=False)[0][0]

    # Cropped to the (mostly white) animal box => much brighter than the full,
    # mostly-black frame. This proves the crop is applied at load time.
    assert cropped.mean().item() > full.mean().item() + 0.3


def test_crop_disabled_uses_full_frame(tmp_path):
    img = Image.new("L", (64, 64), color=100)
    (tmp_path / "x").mkdir()
    img.save(tmp_path / "x" / "img.jpg")
    rows = [{"class": "x", "filename": "x/img.jpg", "bbox": "", "image_id": "id0"}]
    tf = _tiny_transform()
    # No bbox present -> crop flag has no effect; both return the full frame.
    a = ManifestDataset(rows, str(tmp_path), {"x": 0}, tf, crop_to_bbox=True)[0][0]
    b = ManifestDataset(rows, str(tmp_path), {"x": 0}, tf, crop_to_bbox=False)[0][0]
    assert torch.allclose(a, b)


def test_config_json_round_trip(tmp_path):
    cfg = Config()
    cfg.epochs = 7
    cfg.learning_rate = 1.5e-4
    cfg.split_by = "stratified"
    path = str(tmp_path / "config.json")
    cfg.to_json(path)
    loaded = Config.from_json(path)
    assert loaded == cfg


def test_config_from_json_ignores_unknown_keys(tmp_path):
    import json
    path = tmp_path / "config.json"
    data = Config().__dict__.copy()
    data["a_removed_field"] = 123          # simulate an older/newer schema
    path.write_text(json.dumps(data))
    loaded = Config.from_json(str(path))   # must not raise
    assert loaded == Config()
    assert not hasattr(loaded, "a_removed_field")
