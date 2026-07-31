"""Location-grouped train/val/test splitting.

Camera-trap frames from the same site share backgrounds, so a naive random split
lets the model recognise the *location* instead of the *animal*. To measure real
generalisation we assign whole camera **locations** to a single split, so no
background is shared between train, validation and test.

The assignment is greedy and class-aware: it fills the test and validation splits
towards their target fractions **per species**, so every species is represented
in every split even though the split is by location. It is deterministic given
the seed.
"""
from __future__ import annotations

import random
from collections import Counter, defaultdict
from typing import Dict, List


def location_grouped_split(records: List[dict], val_fraction: float,
                           test_fraction: float, seed: int) -> Dict[str, str]:
    """Assign each camera location to 'train', 'val' or 'test'.

    Args:
        records: dicts each with at least ``class`` and ``location`` keys.
        val_fraction, test_fraction: target share of images (per class).
        seed: determinism.

    Returns:
        {location: split} for every location present in ``records``.
    """
    class_total = Counter(r["class"] for r in records)
    test_target = {c: test_fraction * n for c, n in class_total.items()}
    val_target = {c: val_fraction * n for c, n in class_total.items()}
    total = sum(class_total.values())
    test_total_target = test_fraction * total
    val_total_target = val_fraction * total

    loc_classes: Dict[str, Counter] = defaultdict(Counter)
    for r in records:
        loc_classes[r["location"]][r["class"]] += 1

    locations = sorted(loc_classes)           # deterministic base order
    random.Random(seed).shuffle(locations)

    test_count: Counter = Counter()
    val_count: Counter = Counter()
    assignment: Dict[str, str] = {}

    for loc in locations:
        classes_here = loc_classes[loc]
        # A location is sent to a split if some class there is still under its
        # per-class target AND either the split has overall room left OR the
        # location introduces a class that split doesn't have yet. The
        # total-room cap stops a single big location from massively overshooting
        # the target fraction, while the "missing class" clause guarantees every
        # class still reaches every split (so coverage is preserved).
        test_under = any(test_count[c] < test_target[c] for c in classes_here)
        test_room = sum(test_count.values()) < test_total_target
        test_missing = any(test_count[c] == 0 for c in classes_here)
        val_under = any(val_count[c] < val_target[c] for c in classes_here)
        val_room = sum(val_count.values()) < val_total_target
        val_missing = any(val_count[c] == 0 for c in classes_here)

        if test_under and (test_room or test_missing):
            assignment[loc] = "test"
            test_count.update(classes_here)
        elif val_under and (val_room or val_missing):
            assignment[loc] = "val"
            val_count.update(classes_here)
        else:
            assignment[loc] = "train"
    return assignment
