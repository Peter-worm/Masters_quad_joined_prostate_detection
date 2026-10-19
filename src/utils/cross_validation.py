import random
from collections.abc import Sequence


def create_folds(
    items: Sequence[str],
    n_splits: int = 5,
    random_seed: int = 42,
) -> list[list[str]]:
    """Create reproducible, non-overlapping folds from patient-level items."""
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2.")
    if len(items) < n_splits:
        raise ValueError("At least n_splits items are required.")

    shuffled_items = list(items)
    random.Random(random_seed).shuffle(shuffled_items)
    base_size, remainder = divmod(len(shuffled_items), n_splits)
    folds: list[list[str]] = []
    start = 0
    for fold_number in range(n_splits):
        fold_size = base_size + (fold_number < remainder)
        folds.append(sorted(shuffled_items[start : start + fold_size]))
        start += fold_size
    return folds
