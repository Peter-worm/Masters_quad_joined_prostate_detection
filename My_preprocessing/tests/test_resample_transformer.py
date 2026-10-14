import unittest

import numpy as np
import SimpleITK as sitk

from core.transformers.masks_fill_transformer import masks_fill_transformer
from core.transformers.resample_transformer import resample_transformer


class ResampleTransformerTests(unittest.TestCase):
    def test_resamples_channels_to_reference_grid_and_preserves_binary_labels(self):
        t2 = sitk.GetImageFromArray(np.ones((5, 6, 7), dtype=np.float32))
        t2.SetSpacing((1.5, 1.5, 2.0))
        t2.SetOrigin((10.0, 20.0, 30.0))

        adc = sitk.GetImageFromArray(np.full((5, 6, 7), 12, dtype=np.int16))
        adc.SetSpacing((1.5, 1.5, 2.0))
        adc.SetOrigin((11.5, 20.0, 30.0))

        lesion_array = np.zeros((5, 6, 7), dtype=np.uint8)
        lesion_array[2, 3, 3] = 1
        lesion = sitk.GetImageFromArray(lesion_array)
        lesion.CopyInformation(t2)

        anatomy_array = np.ones((5, 6, 7), dtype=np.uint8)
        anatomy = sitk.GetImageFromArray(anatomy_array)
        anatomy.CopyInformation(t2)

        transformer = resample_transformer(
            crop_size=(16, 16, 8),
            target_spacing=(0.75, 0.75, 1.0),
            channels=("t2", "adc", "anatomy", "lesion"),
        )
        result = transformer.execute(
            {"t2": t2, "adc": adc, "anatomy": anatomy, "lesion": lesion}
        )

        reference = result["t2"]
        for channel in ("adc", "anatomy", "lesion"):
            self.assertEqual(result[channel].GetSize(), reference.GetSize())
            self.assertEqual(result[channel].GetSpacing(), reference.GetSpacing())
            self.assertEqual(result[channel].GetOrigin(), reference.GetOrigin())
            self.assertEqual(result[channel].GetDirection(), reference.GetDirection())

        output_labels = np.unique(sitk.GetArrayFromImage(result["lesion"]))
        self.assertTrue(set(output_labels).issubset({0, 1}))
        self.assertEqual(sitk.GetArrayFromImage(result["adc"])[0, 0, 0], 0.0)
        self.assertEqual(result["t2"].GetPixelID(), sitk.sitkFloat32)
        self.assertEqual(result["adc"].GetPixelID(), sitk.sitkFloat32)
        self.assertIn("foreground_voxels", transformer.last_mask_resampling_summary["lesion"]["before"])
        self.assertIn("components", transformer.last_mask_resampling_summary["lesion"]["after"])
        self.assertIn("nonempty_z_slices", transformer.last_mask_resampling_summary["lesion"]["after"])

    def test_gap_filling_changes_anatomy_but_not_lesion_labels(self):
        anatomy = np.zeros((7, 7, 3), dtype=np.uint8)
        anatomy[2:5, 2:5, 0] = 1
        anatomy[2:5, 2:5, 2] = 1

        lesion = anatomy.copy()
        lesion_before = lesion.copy()

        result = masks_fill_transformer().execute(
            {"anatomy": anatomy, "lesion": lesion}
        )

        self.assertGreater(int(result["anatomy"][:, :, 1].sum()), 0)
        np.testing.assert_array_equal(result["lesion"], lesion_before)


if __name__ == "__main__":
    unittest.main()
