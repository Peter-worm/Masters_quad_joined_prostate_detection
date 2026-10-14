from core.file_manager import preprocess_file_manager
import numpy as np
from scipy.ndimage import distance_transform_edt
from core.stat_calc import find_periods

def signed_distance(mask):
    mask = mask.astype(bool)
    outside = distance_transform_edt(~mask)
    inside = distance_transform_edt(mask)
    return outside - inside

def interpolate_shapes(A, B, steps):
    dA = signed_distance(A)
    dB = signed_distance(B)
    result = [] 
    for i in range(1, steps + 1):
        t = i / (steps + 1)
        d = (1 - t) * dA + t * dB
        result.append((d < 0).astype(int))
    return result

def fix_patient_mask_file_to_file(outlier, file_manager: preprocess_file_manager,mask_channel = 'anatomy',step = 'raw'):
    outlier_data = file_manager.load_file(step, outlier)
    outlier_new_data = fix_patient_mask(outlier_data, mask_channel=mask_channel)
    file_manager.save_file(step,outlier,outlier_new_data)


def fix_patient_mask(outlier_data,mask_channel = 'anatomy'):
    prostate = outlier_data[mask_channel]
    valid_layers = np.any(prostate == 1, axis=(0, 1))
    true_indices = np.where(valid_layers)[0]
    i = 0
    periods = find_periods(true_indices)
    while i < len(periods)-1:
        start = periods[i][1]
        end = periods[i+1][0]
        # print(f"{start}  {end}")

        start_layer = prostate[:,:,start]
        end_layer = prostate[:,:,end]
        number_of_interpolation_layers = end-start-1

        new_leayers = interpolate_shapes(start_layer,end_layer,number_of_interpolation_layers)

        j = start+1
        while j <  end:
            prostate[:,:,j] = new_leayers[j-start-1]
            j+=1
        i+=1
    return outlier_data

def find_gaps_in_mask(patients,file_manager,step = 'raw',mask_channel = 'anatomy'):
    outliers = []
    sections_with_prostate = {}
    for patient in patients:
        prostate = file_manager.load_file(step, patient)[mask_channel]
        valid_layers = np.any(prostate == 1, axis=(0, 1))
        true_indices = np.where(valid_layers)[0]
        if len(find_periods(true_indices)) != 1:
            outliers.append(patient)
        periods = find_periods(true_indices)
        if len(periods) >= 0:
            sections_with_prostate[patient] = (periods[0][0], periods[-1][1])
        else:
            raise ValueError(f"Patient {patient} without anatomy data")
    return outliers,sections_with_prostate