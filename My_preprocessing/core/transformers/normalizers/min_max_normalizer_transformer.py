import numpy as np
import SimpleITK as sitk

from core.transformers.preprocessing_transformer import preprocessing_transformer


class min_max_normalizer_transformer(preprocessing_transformer):
    def __init__(self, channels_to_normalize=['t2', 'dwi', 'adc'], mask_channel='anatomy'):
        self.channels_to_normalize = channels_to_normalize
        self.mask_channel = mask_channel

    def execute(self, patient_data):
        for channel in self.channels_to_normalize:
            patient_data[channel] = self.min_max_normalize_global(
                patient_data[channel],
                patient_data[self.mask_channel]
            )
        return patient_data

    def min_max_normalize_global(self, img, mask):
        arr = sitk.GetArrayFromImage(img).astype(np.float32)
        mask_arr = sitk.GetArrayFromImage(mask)

        masked_vals = arr[mask_arr > 0]

        vmin = masked_vals.min()
        vmax = masked_vals.max()

        # avoid division by zero
        if vmax == vmin:
            norm_arr = np.zeros_like(arr, dtype=np.float32)
        else:
            norm_arr = (arr - vmin) / (vmax - vmin)

        norm_img = sitk.GetImageFromArray(norm_arr)
        norm_img.CopyInformation(img)

        for key in img.GetMetaDataKeys():
            norm_img.SetMetaData(key, img.GetMetaData(key))

        return norm_img