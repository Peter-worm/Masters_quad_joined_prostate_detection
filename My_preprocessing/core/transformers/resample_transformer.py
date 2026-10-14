from core.transformers.preprocessing_transformer import preprocessing_transformer
import SimpleITK as sitk
import numpy as np


class resample_transformer(preprocessing_transformer):
    def __init__(
        self,
        crop_size,
        target_spacing=(0.8, 0.8, 3.5),
        channels=("anatomy", "t2", "dwi", "adc", "lesion"),
        mask_channels=("anatomy", "lesion"),
        reference_channel="t2",
    ):
        self.crop_size = crop_size
        self.target_spacing = target_spacing
        self.channels = channels
        self.mask_channels = mask_channels
        self.reference_channel = reference_channel
        self.last_mask_resampling_summary = {}

        if len(target_spacing) != 3 or any(spacing <= 0 for spacing in target_spacing):
            raise ValueError("target_spacing must contain three positive values.")
        if reference_channel not in channels:
            raise ValueError(
                f"Reference channel '{reference_channel}' must be included in channels."
            )

    def execute(self, patient_data):
        if self.reference_channel not in patient_data:
            raise ValueError(
                f"Reference channel '{self.reference_channel}' is missing from patient data."
            )

        reference_image = patient_data[self.reference_channel]
        reference_grid = self._create_reference_grid(reference_image, self.target_spacing)
        self.last_mask_resampling_summary = {}

        for channel in self.channels:
            if channel not in patient_data:
                raise ValueError(f"Missing channel '{channel}' in patient data.")

            image = patient_data[channel]
            is_mask = channel in self.mask_channels
            if is_mask:
                before = self._mask_summary(image)

            patient_data[channel] = self.resample_nii(
                image,
                target_spacing=self.target_spacing,
                is_mask=is_mask,
                reference=reference_grid,
            )

            if is_mask:
                self.last_mask_resampling_summary[channel] = {
                    "before": before,
                    "after": self._mask_summary(patient_data[channel]),
                }

        return patient_data

    @staticmethod
    def _create_reference_grid(image, target_spacing):
        original_spacing = image.GetSpacing()
        original_size = image.GetSize()
        target_size = [
            max(1, int(round(size * spacing / target)))
            for size, spacing, target in zip(original_size, original_spacing, target_spacing)
        ]

        reference_grid = sitk.Image(target_size, sitk.sitkFloat32)
        reference_grid.SetOrigin(image.GetOrigin())
        reference_grid.SetDirection(image.GetDirection())
        reference_grid.SetSpacing(target_spacing)
        return reference_grid

    @staticmethod
    def _mask_summary(image):
        mask_array = sitk.GetArrayViewFromImage(image) > 0
        components = sitk.ConnectedComponent(
            sitk.GetImageFromArray(mask_array.astype(np.uint8))
        )
        component_count = np.count_nonzero(
            np.unique(sitk.GetArrayViewFromImage(components))
        )
        return {
            "foreground_voxels": int(np.count_nonzero(mask_array)),
            "components": int(component_count),
            "nonempty_z_slices": int(np.count_nonzero(np.any(mask_array, axis=(1, 2)))),
        }

    def resample_nii(self, image, target_spacing=None, is_mask=False, reference=None):
        """Resample an image, using a common reference grid when one is supplied."""
        if target_spacing is None:
            target_spacing = self.target_spacing
        if reference is None:
            reference = self._create_reference_grid(image, target_spacing)

        if is_mask:
            interpolator = sitk.sitkNearestNeighbor
            output_pixel_type = image.GetPixelID()
        else:
            interpolator = sitk.sitkLinear
            output_pixel_type = sitk.sitkFloat32

        return sitk.Resample(
            image,
            reference,
            sitk.Transform(),
            interpolator,
            0.0,
            output_pixel_type,
        )
