import unittest

import numpy as np
import SimpleITK as sitk

from core.preprocessing_pipeline import run_preprocessing_pipeline


class FakeFileManager:
    def __init__(self, patient_data):
        self.patient_data = patient_data
        self.loaded_patients = []
        self.saved = []

    def load_file_NiFty_external(self, data_folder, patient, channels):
        self.loaded_patients.append(patient)
        return self.patient_data

    def save_file(self, step, patient, data):
        self.saved.append((step, patient, data))


class PreprocessingPipelineStorageTests(unittest.TestCase):
    def test_final_only_mode_keeps_patient_in_memory_until_final_output(self):
        image = sitk.GetImageFromArray(np.ones((2, 3, 4), dtype=np.float32))
        file_manager = FakeFileManager({"t2": image})
        settings = {
            "original_data_folder": "/unused/source",
            "channels": {"t2": "t2"},
            "save_intermediate_steps": False,
            "create_train_test_split": False,
            "preprocessing_steps_list": [
                ("start", "nifty"),
                ("nifti_to_raw", "raw"),
            ],
            "preprocessed_steps": {
                "nifti_to_raw": {"start": "0_nifty", "end": "1_raw"}
            },
        }

        run_preprocessing_pipeline(settings, file_manager, patient_filter=["patient-1"])

        self.assertEqual(file_manager.loaded_patients, ["patient-1"])
        self.assertEqual(len(file_manager.saved), 1)
        output_step, patient, data = file_manager.saved[0]
        self.assertEqual(output_step, "1_raw")
        self.assertEqual(patient, "patient-1")
        self.assertIsInstance(data["t2"], np.ndarray)
        self.assertEqual(data["t2"].shape, (4, 3, 2))


if __name__ == "__main__":
    unittest.main()
