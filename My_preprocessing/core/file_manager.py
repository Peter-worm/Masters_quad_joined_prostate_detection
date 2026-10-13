import pickle
import os
from pathlib import Path
import SimpleITK as sitk

class preprocess_file_manager:
    def __init__(self, main_folder,preprocess_steps,channels,current_load_step = None):
        self.main_folder = main_folder
        self.steps = preprocess_steps
        if current_load_step is not None:
            self.current_load_step = current_load_step
        else:
            self.current_load_step = preprocess_steps[list(preprocess_steps.keys())[0]]['start']
        self.channels = channels

    def load_file(self, step, patient_id):
        path = os.path.join(self.main_folder, step, f"{patient_id}")
        pkl_path = Path(path)
        if pkl_path.is_dir():
            return self.load_file_NifTy(step, patient_id)
        pkl_path = Path(path + ".pkl")
        if pkl_path.is_file():
            return self.load_file_pickle(step, patient_id)
        else:
            raise FileNotFoundError(f"No file found for patient {patient_id} in step {step}.")

    def save_file(self, step, patient_id, data):
        if isinstance(data, dict) and all(isinstance(v, sitk.Image) for v in data.values()):
            self.save_file_Nifty(step, patient_id, data)
        else:
            self.save_file_pickle(step, patient_id, data)


    ## pickle loading
    def load_file_pickle(self, step, patient_id):
        path = os.path.join(self.main_folder, step, f"{patient_id}.pkl")

        with open(path, "rb") as f:
            data = pickle.load(f)

        return data

    def save_file_pickle(self, step, patient_id, data):
        path = os.path.join(self.main_folder, step, f"{patient_id}.pkl")
        
        os.makedirs(os.path.dirname(path), exist_ok=True)

        with open(path, "wb") as f:
            pickle.dump(data, f)
    ##nii.gz
    def load_file_NifTy(self, step, patient_id):
        data_path = os.path.join(self.main_folder, step)

        data = self.load_file_NiFty_external(data_path,patient_id,self.channels)
        return data

    def load_file_NiFty_external(self,data_path,patient_id,channels = None):
        if channels == None:
            channels = self.channels

        scikit_images = {}
        patient_path = data_path / Path(patient_id)
        if patient_path.is_dir():
            for key in channels:
                channel_file = patient_path / f"{channels[key]}.nii.gz"
                image = sitk.ReadImage(str(channel_file))
                scikit_images[key]= image
        return scikit_images

    def save_file_Nifty(self, step, patient_id, data):
        data_path = os.path.join(self.main_folder, step)
        self.save_file_Nifty_external(data_path, patient_id, data, self.channels)


    def save_file_Nifty_external(self, data_path, patient_id, data, channels=None):
        if channels is None:
            channels = self.channels

        patient_path = Path(data_path) / Path(patient_id)
        patient_path.mkdir(parents=True, exist_ok=True)

        for key in channels:
            if key not in data:
                raise ValueError(f"Missing channel '{key}' in data dictionary.")

            channel_file = patient_path / f"{channels[key]}.nii.gz"
            sitk.WriteImage(data[key], str(channel_file))


    def get_file_names(self):
        data_folder = os.path.join(self.main_folder, self.current_load_step)
        folder = Path(data_folder)
        ids = []
        for patient in folder.iterdir():
            ids.append(patient.stem)   # removes .plk
        return ids
    
    def copy_NiFty_to_internal(self, data_folder, channels):
        folder = Path(data_folder)
        patients_folders = [x.name for x in folder.iterdir()]
        for patient in patients_folders:
            data = self.load_file_NiFty_external(data_folder,patient,channels)
            self.save_file_Nifty('raw',patient,data)
        

