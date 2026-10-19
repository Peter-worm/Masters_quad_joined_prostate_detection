import math
import random


def split_patient_ids(
    patient_ids: list[str],
    validation_fraction: float = 0.1,
    test_fraction: float = 0.1,
    random_seed: int = 42,
) -> tuple[list[str], list[str], list[str]]:
    """Return reproducible, disjoint patient-level train/validation/test splits."""
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1.")
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be between 0 and 1.")
    if validation_fraction + test_fraction >= 1:
        raise ValueError("validation_fraction and test_fraction must sum to less than 1.")

    unique_patient_ids = sorted(set(patient_ids))
    if len(unique_patient_ids) != len(patient_ids):
        raise ValueError("Patient IDs must be unique before splitting.")
    if len(unique_patient_ids) < 3:
        raise ValueError("At least three patients are required for train/validation/test splits.")

    shuffled_patient_ids = unique_patient_ids.copy()
    random.Random(random_seed).shuffle(shuffled_patient_ids)
    validation_count = max(1, math.ceil(len(shuffled_patient_ids) * validation_fraction))
    test_count = max(1, math.ceil(len(shuffled_patient_ids) * test_fraction))
    if validation_count + test_count >= len(shuffled_patient_ids):
        raise ValueError(
            "validation_fraction and test_fraction leave fewer than one patient for training."
        )

    validation_ids = sorted(shuffled_patient_ids[:validation_count])
    test_ids = sorted(
        shuffled_patient_ids[validation_count : validation_count + test_count]
    )
    train_ids = sorted(shuffled_patient_ids[validation_count + test_count :])
    return train_ids, validation_ids, test_ids
