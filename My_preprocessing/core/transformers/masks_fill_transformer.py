from core.transformers.preprocessing_transformer import preprocessing_transformer
from core.anatomy_gap_fixer import fix_patient_mask

class masks_fill_transformer(preprocessing_transformer):
    def __init__(self,outliers = None):
        self.outliers = outliers

    def execute(self, patient_data,channels = ['anatomy','lesion']):
        for channel in channels:
            patient_data = fix_patient_mask(patient_data, mask_channel=channel)
        return patient_data
        