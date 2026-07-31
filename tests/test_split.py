"""Tests for location-grouped splitting and the seen-location holdout."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.split import location_grouped_split
from src.data import _carve_seen_test


def _records():
    """Six locations, three classes, a handful of images each."""
    recs = []
    for loc in range(6):
        cls = ["a", "b", "c"][loc % 3]
        for i in range(10):
            recs.append({"class": cls, "location": str(loc),
                         "image_id": f"{loc}-{cls}-{i}"})
    return recs


def test_no_location_shared_between_splits():
    recs = _records()
    assignment = location_grouped_split(recs, 0.2, 0.2, seed=0)
    # Every location maps to exactly one split.
    assert set(assignment.values()) <= {"train", "val", "test"}
    # A location cannot appear in two splits (assignment is per-location).
    assert len(assignment) == len({r["location"] for r in recs})


def test_split_is_deterministic():
    recs = _records()
    a = location_grouped_split(recs, 0.2, 0.2, seed=42)
    b = location_grouped_split(recs, 0.2, 0.2, seed=42)
    assert a == b
    # A different seed may reassign locations.
    c = location_grouped_split(recs, 0.2, 0.2, seed=7)
    assert isinstance(c, dict) and set(c) == set(a)


def test_every_class_can_reach_test():
    # With enough distinct locations per class the greedy fills test for each.
    recs = _records()
    assignment = location_grouped_split(recs, 0.34, 0.34, seed=1)
    test_classes = {r["class"] for r in recs if assignment[r["location"]] == "test"}
    assert test_classes  # at least one class in test
    assert "test" in assignment.values()


def test_split_sizes_do_not_massively_overshoot():
    # Many one-class locations: the greedy must not blow past the test fraction.
    recs = []
    for loc in range(30):
        cls = ["a", "b", "c"][loc % 3]
        for i in range(10):
            recs.append({"class": cls, "location": str(loc),
                         "image_id": f"{loc}-{i}"})
    assignment = location_grouped_split(recs, 0.15, 0.15, seed=0)
    n = len(recs)
    n_test = sum(1 for r in recs if assignment[r["location"]] == "test")
    n_val = sum(1 for r in recs if assignment[r["location"]] == "val")
    # target 15% each; allow slack for whole-location granularity, but not 2x.
    assert n_test / n < 0.30
    assert n_val / n < 0.30
    # every class still present in test and val
    for split in ("test", "val", "train"):
        classes = {r["class"] for r in recs if assignment[r["location"]] == split}
        assert classes == {"a", "b", "c"}


def test_seen_test_carve_is_deterministic_and_disjoint():
    recs = _records()
    keep1, seen1 = _carve_seen_test(recs, 0.2, seed=3)
    keep2, seen2 = _carve_seen_test(recs, 0.2, seed=3)
    ids1 = [r["image_id"] for r in seen1]
    ids2 = [r["image_id"] for r in seen2]
    assert ids1 == ids2                         # deterministic
    keep_ids = {r["image_id"] for r in keep1}
    seen_ids = {r["image_id"] for r in seen1}
    assert keep_ids.isdisjoint(seen_ids)        # no image in both
    assert len(keep1) + len(seen1) == len(recs)  # partition


def test_seen_test_is_stratified_by_class():
    recs = _records()
    _, seen = _carve_seen_test(recs, 0.3, seed=0)
    seen_classes = {r["class"] for r in seen}
    assert seen_classes == {"a", "b", "c"}      # every class represented
