"""Metrics and result-reporting utilities."""


def accuracy(predictions, targets) -> float:
    """Calculate accuracy for class-label tensors."""
    if predictions.numel() == 0:
        return 0.0
    predicted_classes = predictions.argmax(dim=1)
    return (predicted_classes == targets).float().mean().item()
