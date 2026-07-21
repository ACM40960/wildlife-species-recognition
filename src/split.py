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
        needs_test = any(test_count[c] < test_target[c] for c in classes_here)
        needs_val = any(val_count[c] < val_target[c] for c in classes_here)
        if needs_test:
            assignment[loc] = "test"
            test_count.update(classes_here)
        elif needs_val:
            assignment[loc] = "val"
            val_count.update(classes_here)
        else:
            assignment[loc] = "train"
    return assignment
