import math
import random


def split_patient_ids(
    patient_ids: list[str],
    test_fraction: float = 0.2,
    random_seed: int = 42,
) -> tuple[list[str], list[str]]:
    """Return a reproducible patient-level train/test split."""
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be between 0 and 1.")

    unique_patient_ids = sorted(set(patient_ids))
    if len(unique_patient_ids) != len(patient_ids):
        raise ValueError("Patient IDs must be unique before splitting.")
    if len(unique_patient_ids) < 2:
        raise ValueError("At least two patients are required for a train/test split.")

    shuffled_patient_ids = unique_patient_ids.copy()
    random.Random(random_seed).shuffle(shuffled_patient_ids)
    test_count = min(
        len(shuffled_patient_ids) - 1,
        max(1, math.ceil(len(shuffled_patient_ids) * test_fraction)),
    )
    test_ids = sorted(shuffled_patient_ids[:test_count])
    train_ids = sorted(shuffled_patient_ids[test_count:])
    return train_ids, test_ids

