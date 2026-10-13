"""Local entry point for the project."""

from src.config import BASE_DIR, IN_COLAB, PROCESSED_DATA_DIR, RAW_DATA_DIR, SAVED_WEIGHTS_DIR


def main() -> None:
    print(f"Project directory: {BASE_DIR}")
    print(f"Colab environment: {'yes' if IN_COLAB else 'no'}")
    print(f"Raw data: {RAW_DATA_DIR}")
    print(f"Processed data: {PROCESSED_DATA_DIR}")
    print(f"Model weights: {SAVED_WEIGHTS_DIR}")
    print("Complete src/data and src/models to start training.")


if __name__ == "__main__":
    main()
