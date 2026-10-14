import numpy as np
def find_periods(arr):
    if len(arr) == 0:
        return []

    periods = []
    start = arr[0]
    prev = arr[0]

    for num in arr[1:]:
        if num == prev + 1:
            prev = num
        else:
            periods.append((start, prev))
            start = num
            prev = num

    periods.append((start, prev))
    return periods

def find_centroid_mean(figure):
    coords = np.argwhere(figure == 1)
    centroid = coords.mean(axis=0)
    return centroid

def find_figure_box(figure):
    idx = np.where(figure == 1)
    min_indices = [i.min() for i in idx]
    max_indices = [i.max() for i in idx]
    return min_indices, max_indices

def find_centroid_non_weighted(figure):
    min_indices, max_indices = find_figure_box(figure)
    return [np.mean([min_indices[i],max_indices[i]]) for i in range (0,len(max_indices))]


##i dont know if this should be in this folder...
def find_patients_max_prostate_sizes(patients,file_manager,step = 'raw'):
    maximum_prostate_size = [0,0,0]
    for patient in patients:
        prostate = file_manager.load_file(step, patient)['anatomy']
        borders = find_figure_box(prostate)
        prostate_size= [ borders[1][i] - borders[0][i]+1 for i in range (0,len(borders[0]))]
        for dim, size in enumerate(prostate_size):
            if (size > maximum_prostate_size[dim]):
                maximum_prostate_size[dim] = size
    return maximum_prostate_size
    