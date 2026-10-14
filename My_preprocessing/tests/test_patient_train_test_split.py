import tempfile
import unittest
from pathlib import Path

import numpy as np
import SimpleITK as sitk

from core.file_manager import preprocess_file_manager
from core.helper import transform_step_list_to_dictioanry
from core.preprocessing_pipeline import run_preprocessing_pipeline, split_patient_ids


class PatientTrainTestSplitTests(unittest.TestCase):
    def test_split_is_reproducible_disjoint_and_80_20(self):
        patient_ids = [f"patient-{index}" for index in range(10)]

        train_ids, test_ids = split_patient_ids(patient_ids)
        repeated_train_ids, repeated_test_ids = split_patient_ids(patient_ids)

        self.assertEqual((train_ids, test_ids), (repeated_train_ids, repeated_test_ids))
        self.assertEqual(len(train_ids), 8)
        self.assertEqual(len(test_ids), 2)
        self.assertTrue(set(train_ids).isdisjoint(test_ids))
        self.assertEqual(set(train_ids) | set(test_ids), set(patient_ids))

    def test_pipeline_writes_preprocessed_patients_to_separate_split_folders(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source_dir = root / "source"
            output_dir = root / "processed"
            source_dir.mkdir()

            image = sitk.GetImageFromArray(np.ones((2, 3, 4), dtype=np.float32))
            patient_ids = [f"patient-{index}" for index in range(5)]
            for patient_id in patient_ids:
                patient_dir = source_dir / patient_id
                patient_dir.mkdir()
                sitk.WriteImage(image, str(patient_dir / "t2.nii.gz"))

            steps = [("start", "nifty"), ("nifti_to_raw", "raw")]
            step_map = transform_step_list_to_dictioanry(steps)
            file_manager = preprocess_file_manager(
                str(output_dir), step_map, {"t2": "t2"}
            )
            settings = {
                "original_data_folder": str(source_dir),
                "channels": {"t2": "t2"},
                "save_intermediate_steps": False,
                "create_train_test_split": True,
                "test_fraction": 0.2,
                "split_seed": 42,
                "preprocessing_steps_list": steps,
                "preprocessed_steps": step_map,
            }

            run_preprocessing_pipeline(settings, file_manager)

            train_dir = output_dir / "train" / "1_raw"
            test_dir = output_dir / "test" / "1_raw"
            train_output_ids = {path.stem for path in train_dir.glob("*.pkl")}
            test_output_ids = {path.stem for path in test_dir.glob("*.pkl")}

            self.assertEqual(len(train_output_ids), 4)
            self.assertEqual(len(test_output_ids), 1)
            self.assertTrue(train_output_ids.isdisjoint(test_output_ids))
            self.assertEqual(train_output_ids | test_output_ids, set(patient_ids))

    def test_intermediate_outputs_are_also_separated_by_split(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source_dir = root / "source"
            output_dir = root / "processed"
            source_dir.mkdir()

            image = sitk.GetImageFromArray(np.ones((2, 3, 4), dtype=np.float32))
            patient_ids = [f"patient-{index}" for index in range(5)]
            for patient_id in patient_ids:
                patient_dir = source_dir / patient_id
                patient_dir.mkdir()
                sitk.WriteImage(image, str(patient_dir / "t2.nii.gz"))

            steps = [("start", "nifty"), ("nifti_to_raw", "raw")]
            step_map = transform_step_list_to_dictioanry(steps)
            file_manager = preprocess_file_manager(
                str(output_dir), step_map, {"t2": "t2"}
            )
            settings = {
                "original_data_folder": str(source_dir),
                "channels": {"t2": "t2"},
                "save_intermediate_steps": True,
                "create_train_test_split": True,
                "test_fraction": 0.2,
                "split_seed": 42,
                "preprocessing_steps_list": steps,
                "preprocessed_steps": step_map,
            }

            run_preprocessing_pipeline(settings, file_manager)

            for split in ("train", "test"):
                split_root = output_dir / split
                raw_ids = {path.stem for path in (split_root / "1_raw").glob("*.pkl")}
                nifty_ids = {path.name for path in (split_root / "0_nifty").iterdir()}
                self.assertEqual(raw_ids, nifty_ids)


if __name__ == "__main__":
    unittest.main()
