from core.transformers.preprocessing_transformer import preprocessing_transformer
from core.helper import sikit_to_just_data

class nifti_to_raw_transformer(preprocessing_transformer):

    def execute(self, patient_data):
        raw_data = sikit_to_just_data(patient_data)
        return raw_data
        