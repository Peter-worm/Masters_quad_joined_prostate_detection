from core.transformers.preprocessing_transformer import preprocessing_transformer
import SimpleITK as sitk
import numpy as np

class resample_transformer(preprocessing_transformer):
    def __init__(self, crop_size, target_spacing=(0.8, 0.8, 3.5),channels = ['anatomy', 't2', 'dwi', 'adc','lesion'], mask_channels = ['anatomy','lesion']):
        self.crop_size = crop_size
        self.target_spacing = target_spacing
        self.channels = channels
        self.mask_channels = mask_channels

    def execute(self, patient_data):
        for channel in self.channels:
            if channel in self.mask_channels:
                patient_data[channel] = self.resample_nii(patient_data[channel], target_spacing=self.target_spacing, is_mask=True)
            else:
                patient_data[channel] = self.resample_nii(patient_data[channel], target_spacing=self.target_spacing, is_mask=False)
        return patient_data


    def resample_nii(
    self,
    image,
    target_spacing=(0.8, 0.8, 3.5),
    is_mask=False
):
        # Load image
        image
        
        original_spacing = image.GetSpacing()
        original_size = image.GetSize()
    
        # Compute new size
        new_size = [
            int(np.round(osz * ospc / tspc))
            for osz, ospc, tspc in zip(original_size, original_spacing, target_spacing)
        ]
    
        # Choose interpolation
        if is_mask:
            interpolator = sitk.sitkNearestNeighbor
        else:
            interpolator = sitk.sitkBSpline  # or sitkLinear
    
        # Resample
        resampled = sitk.Resample(
            image,
            new_size,
            sitk.Transform(),
            interpolator,
            image.GetOrigin(),
            target_spacing,
            image.GetDirection(),
            0,
            image.GetPixelID()
        )
    
        return resampled

