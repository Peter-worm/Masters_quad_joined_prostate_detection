"""Local entry point for the project."""

from src.config import ExperimentConfig
from src.models.train import train


def main() -> None:
    config = ExperimentConfig()
    print(f"Project directory: {config.base_dir}")
    print(f"Colab environment: {'yes' if config.is_colab else 'no'}")
    print(f"Raw data: {config.raw_data_dir}")
    print(f"Processed data: {config.processed_data_dir}")
    print(f"Model weights: {config.saved_weights_dir}")
    train(config)


if __name__ == "__main__":
    main()
