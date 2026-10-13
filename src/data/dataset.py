"""Starter for Dataset classes tailored to the project's data format."""

from typing import Any

from torch.utils.data import Dataset


class ProjectDataset(Dataset):
    """Add sample and label loading for the selected data format."""

    def __len__(self) -> int:
        raise NotImplementedError("Implement the number of samples in the dataset.")

    def __getitem__(self, index: int) -> Any:
        raise NotImplementedError("Implement loading a sample and its label.")
