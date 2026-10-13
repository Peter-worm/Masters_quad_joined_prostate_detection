"""Model architecture definitions."""

from torch import nn


class ExampleModel(nn.Module):
    """Minimal example model to replace with the project's architecture."""

    def __init__(self, input_features: int, output_classes: int) -> None:
        super().__init__()
        self.classifier = nn.Linear(input_features, output_classes)

    def forward(self, inputs):
        return self.classifier(inputs)
