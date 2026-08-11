"""location-grouped train/val/test splitting."""
from __future__ import annotations

import random
from collections import Counter, defaultdict
from typing import Dict, List


def location_grouped_split(records: List[dict], val_fraction: float,
                           test_fraction: float, seed: int) -> Dict[str, str]:
    """assign each camera location to 'train', 'val' or 'test'."""
    class_total = Counter(r["class"] for r in records)
    test_target = {c: test_fraction * n for c, n in class_total.items()}
    val_target = {c: val_fraction * n for c, n in class_total.items()}
    total = sum(class_total.values())
    test_total_target = test_fraction * total
    val_total_target = val_fraction * total

    loc_classes: Dict[str, Counter] = defaultdict(Counter)
    for r in records:
        loc_classes[r["location"]][r["class"]] += 1

    locations = sorted(loc_classes)  # deterministic base order
    random.Random(seed).shuffle(locations)

    # how many locations hold each class, and how many we have already given away to test/val
    class_locations = Counter()
    for classes_here in loc_classes.values():
        for c in classes_here:
            class_locations[c] += 1
    given_away: Counter = Counter()

    def keeps_a_training_location(classes_here) -> bool:
        return all(class_locations[c] - given_away[c] - 1 >= 1 for c in classes_here)

    test_count: Counter = Counter()
    val_count: Counter = Counter()
    assignment: Dict[str, str] = {}

    for loc in locations:
        classes_here = loc_classes[loc]
        # a location is sent to a split if some class there is still under its per-class target
        can_give_away = keeps_a_training_location(classes_here)
        test_under = any(test_count[c] < test_target[c] for c in classes_here)
        test_room = sum(test_count.values()) < test_total_target
        test_missing = any(test_count[c] == 0 for c in classes_here)
        val_under = any(val_count[c] < val_target[c] for c in classes_here)
        val_room = sum(val_count.values()) < val_total_target
        val_missing = any(val_count[c] == 0 for c in classes_here)

        if can_give_away and test_under and (test_room or test_missing):
            assignment[loc] = "test"
            test_count.update(classes_here)
            given_away.update(classes_here.keys())
        elif can_give_away and val_under and (val_room or val_missing):
            assignment[loc] = "val"
            val_count.update(classes_here)
            given_away.update(classes_here.keys())
        else:
            assignment[loc] = "train"
    return assignment
