from core.transformers.preprocessing_transformer import preprocessing_transformer
from core.anatomy_gap_fixer import fix_patient_mask

class masks_fill_transformer(preprocessing_transformer):
    def __init__(self, outliers=None, channels=("anatomy",)):
        self.outliers = outliers
        self.channels = channels

    def execute(self, patient_data, channels=None):
        channels_to_fill = self.channels if channels is None else channels
        for channel in channels_to_fill:
            patient_data = fix_patient_mask(patient_data, mask_channel=channel)
        return patient_data