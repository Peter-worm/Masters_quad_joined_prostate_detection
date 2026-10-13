from abc import ABC, abstractmethod

class preprocessing_transformer(ABC):
    @abstractmethod
    def execute(self, patient_data):
        pass