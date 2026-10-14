"""Run configured preprocessing steps in order."""

import math
import random
from pathlib import Path

from core.file_manager import preprocess_file_manager
from core.helper import copy_NiFty, patients_transform
from core.transformers.crop_transformer import non_weighted_crop_transformer
from core.transformers.masks_fill_transformer import masks_fill_transformer
from core.transformers.nifti_to_raw_transformer import nifti_to_raw_transformer
from core.transformers.normalizers.min_max_normalizer_transformer import min_max_normalizer_transformer
from core.transformers.normalizers.z_score_normalizer_transformer import z_score_normalizer_transformer
from core.transformers.resample_transformer import resample_transformer


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


def run_preprocessing_pipeline(
    settings: dict,
    file_manager: preprocess_file_manager,
    patient_filter: list[str] | None = None,
) -> None:
    """Run the preprocessing steps defined by a settings dictionary."""
    preprocessed_steps = settings["preprocessed_steps"]
    preprocessing_steps = settings["preprocessing_steps_list"]
    channels = settings["channels"]
    save_intermediate_steps = settings.get("save_intermediate_steps", True)
    create_train_test_split = settings.get("create_train_test_split", True)

    if len(preprocessing_steps) < 2:
        raise ValueError("The preprocessing steps must include a start step and at least one transform.")

    for step, _ in preprocessing_steps[1:]:
        if step not in preprocessed_steps:
            raise ValueError(f"No step configuration for preprocessing step: {step}")

    first_transform = preprocessing_steps[1][0]
    initial_step = preprocessed_steps[first_transform]["start"]
    step_transformers = {}

    if preprocessed_steps.get("min_max_normalization"):
        step_transformers["min_max_normalization"] = min_max_normalizer_transformer(
            channels_to_normalize=settings["min_max_channels_to_normalize"],
            mask_channel=settings["mask_channel"],
        )

    if preprocessed_steps.get("z_score_normalization"):
        step_transformers["z_score_normalization"] = z_score_normalizer_transformer(
            channels_to_normalize=settings["z_score_channels_to_normalize"],
            mask_channel=settings["mask_channel"],
        )

    if preprocessed_steps.get("resampling"):
        step_transformers["resampling"] = resample_transformer(
            target_spacing=settings["target_spacing"],
            crop_size=settings["crop_size"],
            channels=channels,
            reference_channel=settings.get("reference_channel", "t2"),
        )

    if preprocessed_steps.get("nifti_to_raw"):
        step_transformers["nifti_to_raw"] = nifti_to_raw_transformer()

    if preprocessed_steps.get("filling_anatomy_gaps"):
        step_transformers["filling_anatomy_gaps"] = masks_fill_transformer()

    if preprocessed_steps.get("cropping"):
        step_transformers["cropping"] = non_weighted_crop_transformer(settings["crop_size"])

    source_folder = Path(settings["original_data_folder"])
    if patient_filter is None:
        if not source_folder.is_dir():
            raise FileNotFoundError(f"Source data folder does not exist: {source_folder}")
        patients = sorted(path.name for path in source_folder.iterdir() if path.is_dir())
    else:
        patients = patient_filter

    if create_train_test_split:
        train_patients, test_patients = split_patient_ids(
            patients,
            test_fraction=settings.get("test_fraction", 0.2),
            random_seed=settings.get("split_seed", 42),
        )
        patient_groups = {
            "train": train_patients,
            "test": test_patients,
        }
        print(
            f"Patient split: train={len(train_patients)}, test={len(test_patients)} "
            f"(test_fraction={settings.get('test_fraction', 0.2)}, "
            f"seed={settings.get('split_seed', 42)})"
        )
    else:
        patient_groups = {"": patients}

    for split_name, split_patients in patient_groups.items():
        if split_name:
            split_root = Path(file_manager.main_folder) / split_name

            split_file_manager = preprocess_file_manager(
                str(split_root),
                preprocessed_steps,
                channels,
                current_load_step=file_manager.current_load_step,
            )
        else:
            split_file_manager = file_manager

        if save_intermediate_steps:
            for step, _ in preprocessing_steps:
                print(f"[{split_name or 'all'}] Starting step: {step}")

                if step == "start":
                    copy_NiFty(
                        settings["original_data_folder"],
                        split_file_manager,
                        channels,
                        filter=split_patients,
                        step=initial_step,
                    )
                else:
                    if step not in step_transformers:
                        raise ValueError(
                            f"No transformer configured for preprocessing step: {step}"
                        )

                    step_settings = preprocessed_steps[step]
                    patients_transform(
                        split_file_manager,
                        step_settings["start"],
                        step_settings["end"],
                        step_transformers[step],
                        filter=split_patients,
                    )

                print(f"[{split_name or 'all'}] Finished step: {step}")
            continue

        output_step = preprocessed_steps[preprocessing_steps[-1][0]]["end"]
        for patient in split_patients:
            print(f"[{split_name or 'all'}] Preprocessing patient: {patient}")
            patient_data = split_file_manager.load_file_NiFty_external(
                settings["original_data_folder"], patient, channels
            )

            for step, _ in preprocessing_steps[1:]:
                if step not in step_transformers:
                    raise ValueError(
                        f"No transformer configured for preprocessing step: {step}"
                    )

                patient_data = step_transformers[step].execute(patient_data)
                mask_summaries = getattr(
                    step_transformers[step], "last_mask_resampling_summary", None
                )
                if mask_summaries:
                    print(f"Mask resampling summary for {patient}: {mask_summaries}")

            split_file_manager.save_file(output_step, patient, patient_data)
            del patient_data

        print(
            f"Saved {split_name or 'all'} preprocessing output to step: {output_step}"
        )
