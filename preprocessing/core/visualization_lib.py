import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display
from core.file_manager import preprocess_file_manager
import numpy as np
import SimpleITK as sitk

def do_nothing_normalizer(file):
    return file

def normalize_volume(volume):
    vmin = volume.min()
    vmax = volume.max()
    if vmax - vmin == 0:
        return np.zeros_like(volume)
    return (volume - vmin) / (vmax - vmin)

def folder_shower(file_manager: preprocess_file_manager,
    step = 'raw',
    normalizer = do_nothing_normalizer,
    channels = {
        'adc': 'adc',
        'anatomy': 'anatomy',
        'dwi': 'dwi',
        't2': 't2'
}):

    def show_image(patient, channel, slice_idx):
        # print(type(file_manager.load_file(step, patient)))
        plt.figure(figsize=(5,5))
        data = file_manager.load_file(step, patient)[channel]
        if type(data) is not np.ndarray:
            data = sitk.GetArrayFromImage(data)
            data = np.transpose(data, (2, 1, 0))
        plt.imshow(
            normalizer(data[:, :, slice_idx]),
            cmap='gray',
            vmin=0,
            vmax=1
        )
        plt.title(f"Patient: {patient} | Channel: {channel} | Slice: {slice_idx}")
        plt.axis('off')
        plt.show()

    patients = file_manager.get_file_names()

    # Patient Dropdown (NEW)
    patient_dropdown = widgets.Dropdown(
        options=sorted(patients),
        value=sorted(patients)[0],
        description='Patient:'
    )

    # Slice Slider
    slice_slider = widgets.IntSlider(
        value=0,
        min=0,
        max=50,  # You may later want to update this dynamically per patient
        step=1,
        description='Slice:',
        continuous_update=False
    )

    # Channel Dropdown
    channel_dropdown = widgets.Dropdown(
        options=list(channels.keys()),
        value='adc',
        description='Channel:'
    )

    # Interactive display
    ui = widgets.interactive(
        show_image,
        patient=patient_dropdown,
        channel=channel_dropdown,
        slice_idx=slice_slider
    )

    display(ui)




def multiple_steps_viewer(
    file_manager: preprocess_file_manager,
    step_groups=(('raw1','raw2','raw3'), ('proc1','proc2','proc3')),
    normalizer=do_nothing_normalizer,
    channels={
        'adc': 'adc',
        'anatomy': 'anatomy',
        'dwi': 'dwi',
        't2': 't2'
    }
):

    def get_depth(step, patient, channel):
        data = file_manager.load_file(step, patient)[channel]

        if not isinstance(data, np.ndarray):
            data = sitk.GetArrayFromImage(data)
            data = np.transpose(data, (2, 1, 0))

        return data.shape[2]

    def show_images(patient, channel, **slice_indices):
        total_steps = sum(len(g) for g in step_groups)
        plt.figure(figsize=(5 * total_steps, 5))

        plot_idx = 1

        for group_idx, group in enumerate(step_groups):
            slice_idx = slice_indices[f'slice_{group_idx}']

            for step in group:
                data = file_manager.load_file(step, patient)[channel]

                if not isinstance(data, np.ndarray):
                    data = sitk.GetArrayFromImage(data)
                    data = np.transpose(data, (2, 1, 0))

                plt.subplot(1, total_steps, plot_idx)
                plt.imshow(
                    normalizer(data[:, :, slice_idx]),
                    cmap='gray',
                    vmin=0,
                    vmax=1
                )
                plt.title(f"{step}\n(slice {slice_idx})")
                plt.axis('off')

                plot_idx += 1

        plt.suptitle(f"Patient: {patient} | Channel: {channel}")
        plt.show()

    # --- UI ---
    patients = file_manager.get_file_names()

    patient_dropdown = widgets.Dropdown(
        options=sorted(patients),
        value=sorted(patients)[0],
        description='Patient:'
    )

    channel_dropdown = widgets.Dropdown(
        options=list(channels.keys()),
        value='adc',
        description='Channel:'
    )

    # Create sliders
    sliders = {}
    for i in range(len(step_groups)):
        sliders[f'slice_{i}'] = widgets.IntSlider(
            value=0,
            min=0,
            max=1,  # temporary, will update
            step=1,
            description=f'Slice G{i}:',
            continuous_update=False
        )

    # 🔥 Dynamic update per patient/channel
    def update_sliders(*args):
        patient = patient_dropdown.value
        channel = channel_dropdown.value

        for i, group in enumerate(step_groups):
            # use first step in group as reference
            ref_step = group[0]

            try:
                depth = min(get_depth(step, patient, channel) for step in group)
                sliders[f'slice_{i}'].max = depth - 1

                # clamp value if needed
                if sliders[f'slice_{i}'].value > depth - 1:
                    sliders[f'slice_{i}'].value = depth - 1

            except Exception as e:
                print(f"Warning: could not load {ref_step} → {e}")

    # Attach updates
    patient_dropdown.observe(update_sliders, names='value')
    channel_dropdown.observe(update_sliders, names='value')

    # Initial update
    update_sliders()

    ui = widgets.interactive(
        show_images,
        patient=patient_dropdown,
        channel=channel_dropdown,
        **sliders
    )

    display(ui)






def show_transformation(dictionary):
    n = len(dictionary)
    plt.figure(figsize=(5*n, 5))

    for i, (name, img) in enumerate(dictionary.items(), 1):
        plt.subplot(1, n, i)
        plt.imshow(img, cmap='gray', vmin=0, vmax=1)
        plt.title(name)
        plt.axis('off')

    plt.show()