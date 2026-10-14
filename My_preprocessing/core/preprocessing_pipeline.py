"""Run configured preprocessing steps in order."""

from pathlib import Path

from core.file_manager import preprocess_file_manager
from core.helper import copy_NiFty, patients_transform
from core.transformers.crop_transformer import non_weighted_crop_transformer
from core.transformers.masks_fill_transformer import masks_fill_transformer
from core.transformers.nifti_to_raw_transformer import nifti_to_raw_transformer
from core.transformers.normalizers.min_max_normalizer_transformer import min_max_normalizer_transformer
from core.transformers.normalizers.z_score_normalizer_transformer import z_score_normalizer_transformer
from core.transformers.resample_transformer import resample_transformer


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

    if save_intermediate_steps:
        for step, _ in preprocessing_steps:
            print(f"Starting step: {step}")

            if step == "start":
                copy_NiFty(
                    settings["original_data_folder"],
                    file_manager,
                    channels,
                    filter=patient_filter,
                    step=initial_step,
                )
            else:
                if step not in step_transformers:
                    raise ValueError(f"No transformer configured for preprocessing step: {step}")

                step_settings = preprocessed_steps[step]
                patients_transform(
                    file_manager,
                    step_settings["start"],
                    step_settings["end"],
                    step_transformers[step],
                    filter=patient_filter,
                )

            print(f"Finished step: {step}")
        return

    source_folder = Path(settings["original_data_folder"])
    if patient_filter is None:
        patients = sorted(path.name for path in source_folder.iterdir() if path.is_dir())
    else:
        patients = patient_filter

    output_step = preprocessed_steps[preprocessing_steps[-1][0]]["end"]
    for patient in patients:
        print(f"Preprocessing patient: {patient}")
        patient_data = file_manager.load_file_NiFty_external(
            settings["original_data_folder"], patient, channels
        )

        for step, _ in preprocessing_steps[1:]:
            if step not in step_transformers:
                raise ValueError(f"No transformer configured for preprocessing step: {step}")

            patient_data = step_transformers[step].execute(patient_data)
            mask_summaries = getattr(
                step_transformers[step], "last_mask_resampling_summary", None
            )
            if mask_summaries:
                print(f"Mask resampling summary for {patient}: {mask_summaries}")

        file_manager.save_file(output_step, patient, patient_data)
        del patient_data

    print(f"Saved final preprocessing output to step: {output_step}")
