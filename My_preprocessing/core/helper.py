import SimpleITK as sitk
import numpy as np
from core.file_manager import preprocess_file_manager
from pathlib import Path
from core.transformers.preprocessing_transformer import preprocessing_transformer

def register_and_resample(moving, reference, interpolator=sitk.sitkLinear):
    """Register moving image to reference and resample."""
    
    transform = sitk.CenteredTransformInitializer(
        reference,
        moving,
        sitk.Euler3DTransform(),
        sitk.CenteredTransformInitializerFilter.GEOMETRY
    )

    resampled = sitk.Resample(
        moving,
        reference,
        transform,
        interpolator,
        0.0,
        moving.GetPixelID()
    )

    return resampled

def sikit_to_just_data(scikit_images):     
        ##transform to just data
        patient_data = {}
        for channel in scikit_images:
            patient_data[channel] = sitk.GetArrayFromImage(scikit_images[channel])
            #sus transpose
            patient_data[channel] = np.transpose(patient_data[channel], (2, 1, 0))
        return patient_data


def load_NiFty_and_save_raw_data(data_folder,file_manager: preprocess_file_manager,channels,resample_to = 't2',resample_channels = ['adc','dwi'],filter = None,step = 'raw'):
    folder = Path(data_folder)
    if filter == None:
        patients_folders = [x.name for x in folder.iterdir()]
    else:
         patients_folders = filter

    for patient in patients_folders:
        # print(patient)
        data = file_manager.load_file_NiFty_external(data_folder,patient,channels)
        # data = register_and_resample()
        raw_data = sikit_to_just_data(data)
        file_manager.save_file_pickle(step,patient,raw_data)

def copy_NiFty(data_folder,file_manager: preprocess_file_manager,channels,filter = None,step = 'raw'):
    folder = Path(data_folder)
    if filter == None:
        patients_folders = [x.name for x in folder.iterdir()]
    else:
         patients_folders = filter

    for patient in patients_folders:
        data = file_manager.load_file_NiFty_external(data_folder,patient,channels)
        file_manager.save_file_Nifty(step,patient,data)


def patients_transform(file_manager: preprocess_file_manager,start_step,end_step, transformer:preprocessing_transformer,filter = None):
    patients = file_manager.get_file_names()
    if filter:
        patients = [p for p in patients if p in filter]
    for patient in patients:
        patient_data = file_manager.load_file(start_step,patient)
        data_transformed = transformer.execute(patient_data)
        file_manager.save_file(end_step,patient,data_transformed)

def transform_step_list_to_dictioanry(preprocessing_steps_list):
    preprocessed_steps = {}
    for i,(step, output) in enumerate(preprocessing_steps_list):
        if i == 0:
            continue
        preprocessed_steps[step] = {'start': f"{i-1}_{preprocessing_steps_list[i-1][1]}", 'end': f"{i}_{output}"}
    return preprocessed_steps