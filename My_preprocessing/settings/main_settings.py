from core.helper import transform_step_list_to_dictioanry

class test_settings:
    def __init__(self):
        self.original_data_folder =  '/home/robakp/Exeriments1/prostate_lesion_detection/rjozwiak-MGR_dataset_correct/MGR_dataset_correct'

        self.target = 'lesion'

        self.channels = {
            'adc' : 'adc',
            'anatomy' : 'anatomy',
            'dwi' : 'dwi',
            't2' : 't2',
            self.target : self.target
        }

        self.file_extention = '.nii.gz'

        self.preprocessing_steps_list = [
            ('start', 'nifty'),
            ('min_max_normalization', 'min_max_normalized'),
            ('z_score_normalization', 'z_score_normalized'),
            ('resampling','resampled'),
            ('nifti_to_raw', 'raw'),
            ('filling_anatomy_gaps', 'anatomy_gap_filled'),
            ('cropping', 'cropped'),

        ]

        self.preprocessed_steps = transform_step_list_to_dictioanry(self.preprocessing_steps_list)

        self.z_score_channels_to_normalize=['t2']
        self.max_min_channels_to_normalize = ['dwi']
        self.mask_channel='anatomy'

        self.crop_size = (160,160,24)
        self.target_spacing=(0.8, 0.8, 3.5)

    def get_setting_dictionary(self):
        return {
            'original_data_folder': self.original_data_folder,
            'channels': self.channels,
            'target': self.target,
            'file_extention': self.file_extention,
            'preprocessing_steps_list': self.preprocessing_steps_list,
            'preprocessed_steps': self.preprocessed_steps,
            'crop_size': self.crop_size,
            'target_spacing': self.target_spacing,
            'z_score_channels_to_normalize': self.z_score_channels_to_normalize,
            'min_max_channels_to_normalize': self.max_min_channels_to_normalize,
            'mask_channel': self.mask_channel
        }