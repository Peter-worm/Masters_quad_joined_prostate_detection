import pickle
import numpy as np
import torch
from torch.utils.data import Dataset

def to_numpy(val):
    if isinstance(val, tuple):
        return np.asarray(val[0]), val[1]
    return np.asarray(val), None

class ProstateLesionDataset(Dataset):
    def __init__(self, pickle_files: list, is_train: bool = True):
        self.pickle_files = pickle_files
        self.is_train = is_train

    def __len__(self):
        return len(self.pickle_files)

    def __getitem__(self, idx: int):
        file_path = self.pickle_files[idx]
        with open(file_path, "rb") as f:
            sample = pickle.load(f)

        adc, _ = to_numpy(sample["adc"])
        dwi, _ = to_numpy(sample["dwi"])
        t2, _ = to_numpy(sample["t2"])
        anatomy, _ = to_numpy(sample["anatomy"])
        lesion, _ = to_numpy(sample["lesion"])

        if adc.max() > 10.0:
            adc = adc / 1000.0

        def fit_shape(arr):
            if arr.shape[0] == 160 and arr.shape[2] == 24:
                arr = np.transpose(arr, (2, 0, 1))
            return arr.astype(np.float32)

        adc = fit_shape(adc)
        dwi = fit_shape(dwi)
        t2 = fit_shape(t2)
        anatomy = (fit_shape(anatomy) > 0).astype(np.float32)
        lesion = (fit_shape(lesion) > 0).astype(np.float32)

        if self.is_train:
            if np.random.rand() > 0.5:
                adc = np.flip(adc, axis=2).copy()
                dwi = np.flip(dwi, axis=2).copy()
                t2 = np.flip(t2, axis=2).copy()
                anatomy = np.flip(anatomy, axis=2).copy()
                lesion = np.flip(lesion, axis=2).copy()
            if np.random.rand() > 0.5:
                adc = np.flip(adc, axis=1).copy()
                dwi = np.flip(dwi, axis=1).copy()
                t2 = np.flip(t2, axis=1).copy()
                anatomy = np.flip(anatomy, axis=1).copy()
                lesion = np.flip(lesion, axis=1).copy()

        inputs = np.stack([adc, dwi, t2, anatomy], axis=0)
        target = np.expand_dims(lesion, axis=0)

        return torch.from_numpy(inputs), torch.from_numpy(target)