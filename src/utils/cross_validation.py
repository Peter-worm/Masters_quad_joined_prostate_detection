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


def split_train_validation(
    items: Sequence[str],
    validation_fraction: float = 0.1,
    random_seed: int = 42,
) -> tuple[list[str], list[str]]:
    """Create a reproducible train/validation split from the development pool."""
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1.")
    if len(items) < 2:
        raise ValueError("At least two items are required.")

    shuffled_items = list(items)
    random.Random(random_seed).shuffle(shuffled_items)
    validation_count = max(1, round(len(shuffled_items) * validation_fraction))
    if validation_count >= len(shuffled_items):
        raise ValueError("validation_fraction must leave at least one training item.")

    validation_items = sorted(shuffled_items[:validation_count])
    train_items = sorted(shuffled_items[validation_count:])
    return train_items, validation_items
