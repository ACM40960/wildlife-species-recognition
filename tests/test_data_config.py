"""Tests for load-time bbox cropping and Config JSON round-tripping."""
import numpy as np
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


def test_predict_uses_same_crop_as_training(tmp_path):
    """Serving must apply the same animal crop the model was trained on.

    Regression test: predict.py used to classify the full frame while training and
    evaluation used the bbox crop, which nearly halved accuracy on boxed frames.
    """
    import csv

    from src.data import crop_to_box, lookup_bbox

    (tmp_path / "x").mkdir()
    img = Image.new("L", (100, 100), color=0)
    for y in range(10, 30):
        for x in range(10, 30):
            img.putpixel((x, y), 255)
    img.save(tmp_path / "x" / "img.jpg")
    with open(tmp_path / "manifest.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["split", "class", "filename", "bbox",
                                           "has_bbox", "image_id"])
        w.writeheader()
        w.writerow({"split": "test", "class": "x", "filename": "x/img.jpg",
                    "bbox": "10;10;20;20", "has_bbox": "True", "image_id": "id0"})

    path = str(tmp_path / "x" / "img.jpg")
    box = lookup_bbox(path)
    assert box == (10, 10, 20, 20)          # found via the manifest

    tf = _tiny_transform()
    served = tf(crop_to_box(Image.open(path).convert("RGB"), box))
    rows = [{"class": "x", "filename": "x/img.jpg", "bbox": "10;10;20;20",
             "image_id": "id0"}]
    trained = ManifestDataset(rows, str(tmp_path), {"x": 0}, tf, crop_to_bbox=True)[0][0]
    assert torch.allclose(served, trained)  # identical preprocessing


def test_lookup_bbox_returns_none_without_manifest(tmp_path):
    (tmp_path / "y").mkdir()
    Image.new("L", (32, 32)).save(tmp_path / "y" / "img.jpg")
    from src.data import lookup_bbox
    assert lookup_bbox(str(tmp_path / "y" / "img.jpg")) is None


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


def test_letterbox_pads_and_preserves_aspect():
    """Letterbox must keep the whole frame, unlike a centre crop."""
    from src.data import Letterbox

    # A 4:3 frame with a bright band down the far-left edge (wide enough to
    # survive the downsample to 224px).
    img = Image.new("L", (400, 300), color=0)
    for y in range(300):
        for x in range(10):
            img.putpixel((x, y), 255)

    out = Letterbox(224)(img)
    assert out.size == (224, 224)                     # square output
    arr = np.array(out)
    # The left band survives (a centre crop of a 4:3 frame would remove it).
    assert arr[:, :8].max() > 200
    # Aspect preserved: content occupies 224*3/4 = 168 rows, rest is padding.
    rows_with_content = [r for r in range(224) if arr[r].max() > 200]
    assert 150 <= (max(rows_with_content) - min(rows_with_content) + 1) <= 175


def test_letterbox_beats_centre_crop_at_keeping_edges():
    from src.data import build_transforms

    img = Image.new("L", (400, 300), color=0)
    for y in range(300):
        for x in range(10):
            img.putpixel((x, y), 255)
    padded = build_transforms(224, True, train=False, pad_to_square=True)(img)
    cropped = build_transforms(224, True, train=False, pad_to_square=False)(img)
    # The edge marker is visible after padding, gone after centre-cropping.
    assert padded.max() > cropped.max()


def test_megadetector_box_maps_back_to_image_coordinates():
    """The letterbox-undo maths must return boxes in the original image frame."""
    # src.megadetector imports yolov5 lazily inside best_animal_box, so importing
    # the module alone does NOT prove the dependency is present - skip on yolov5
    # itself. yolov5 is an optional extra (CI installs neither detector).
    pytest.importorskip("yolov5")
    md = pytest.importorskip("src.megadetector")
    torch_mod = pytest.importorskip("torch")

    class FakeDet:
        """Returns one 'animal' box covering a known region of the 640x640 input."""
        def __call__(self, x):
            # xyxy in letterboxed space, conf, cls=0 (animal)
            return [torch_mod.tensor([[[320.0, 320.0, 40.0, 40.0, 5.0, 9.0, 0.1, 0.1]]])]

    img = Image.new("RGB", (400, 300), color=0)
    box = md.best_animal_box(FakeDet(), img, conf=0.01)
    if box is None:
        pytest.skip("detector stub incompatible with this yolov5 version")
    x, y, w, h = box
    assert 0 <= x < 400 and 0 <= y < 300          # inside the ORIGINAL image
    assert x + w <= 400 and y + h <= 300


def test_box_fill_preserves_provenance_on_rerun(tmp_path, monkeypatch):
    """Re-running the box fill must not relabel detector boxes as ground truth.

    Regression test: the fill treated any row that already had a box as 'gt', so a
    second run rewrote 200 YOLO-detected rows to 'gt' and the dataset appeared to
    have far more ground-truth annotation than it did.
    """
    import csv
    import importlib.util
    import sys

    (tmp_path / "x").mkdir()
    for name in ("a", "b", "c"):
        Image.new("L", (64, 64)).save(tmp_path / "x" / f"{name}.jpg")
    fields = ["class", "filename", "bbox", "has_bbox", "box_source", "image_id"]
    start = [
        {"class": "x", "filename": "x/a.jpg", "bbox": "1;1;9;9", "has_bbox": "True",
         "box_source": "gt", "image_id": "a"},
        {"class": "x", "filename": "x/b.jpg", "bbox": "2;2;9;9", "has_bbox": "True",
         "box_source": "yolov8", "image_id": "b"},
        {"class": "x", "filename": "x/c.jpg", "bbox": "", "has_bbox": "False",
         "box_source": "none", "image_id": "c"},
    ]
    with open(tmp_path / "manifest.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(start)

    spec = importlib.util.spec_from_file_location("fbx", "scripts/fill_boxes_yolo.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["fbx"] = mod
    spec.loader.exec_module(mod)

    # Stub the detector so the test needs no weights: it always finds a box.
    monkeypatch.setattr(mod, "load_detector", lambda w=None: object())
    monkeypatch.setattr(mod, "best_animal_box", lambda m, p, conf=0.2: (3, 3, 9, 9))
    monkeypatch.setattr(mod, "yolo_available", lambda: True)

    counts = mod.fill_manifest_boxes(str(tmp_path), detector="yolov8")

    after = {r["image_id"]: r["box_source"]
             for r in csv.DictReader(open(tmp_path / "manifest.csv"))}
    assert after["a"] == "gt"        # ground truth stays ground truth
    assert after["b"] == "yolov8"    # detector box is NOT promoted to gt
    assert after["c"] == "yolov8"    # the boxless row gets detected
    assert counts["gt"] == 1


def test_failed_refresh_clears_the_stale_box(tmp_path, monkeypatch):
    """A re-detection that finds nothing must clear the box, not just relabel it.

    Regression test: the row kept has_bbox=True and its old bbox while box_source
    became 'none', so the loader (which reads bbox) still cropped to a box the
    manifest said did not exist.
    """
    import csv
    import importlib.util
    import sys

    (tmp_path / "x").mkdir()
    Image.new("L", (64, 64)).save(tmp_path / "x" / "a.jpg")
    fields = ["class", "filename", "bbox", "has_bbox", "box_source", "image_id"]
    with open(tmp_path / "manifest.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerow({"class": "x", "filename": "x/a.jpg", "bbox": "5;5;20;20",
                    "has_bbox": "True", "box_source": "megadetector",
                    "image_id": "a"})

    spec = importlib.util.spec_from_file_location("fbx2", "scripts/fill_boxes_yolo.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["fbx2"] = mod
    spec.loader.exec_module(mod)
    monkeypatch.setattr(mod, "load_detector", lambda w=None: object())
    monkeypatch.setattr(mod, "yolo_available", lambda: True)
    monkeypatch.setattr(mod, "best_animal_box", lambda m, p, conf=0.2: None)

    mod.fill_manifest_boxes(str(tmp_path), detector="yolov8", refresh=True)

    row = list(csv.DictReader(open(tmp_path / "manifest.csv")))[0]
    assert row["bbox"] == ""
    assert str(row["has_bbox"]).lower() == "false"
    assert row["box_source"] == "none"
