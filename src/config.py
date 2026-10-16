import os
from pathlib import Path

# 1. Environment Detection
IN_COLAB = bool(os.environ.get("COLAB_RELEASE_TAG")) or Path("/content/drive").exists()

# 2. Base Directory Resolution
try:
    _SCRIPT_DIR = Path(__file__).resolve().parent
except NameError:
    # Fallback for interactive environments (Jupyter/Colab notebooks) where __file__ is undefined
    _SCRIPT_DIR = Path.cwd()

_DEFAULT_BASE_DIR = (
    Path("/content/drive/MyDrive/ai_project")
    if IN_COLAB
    else _SCRIPT_DIR.parents[0]  # Assumes config file is inside a subfolder (e.g., src/ or config/); use parent if in root
)

BASE_DIR = Path(os.environ.get("AI_PROJECT_DIR", _DEFAULT_BASE_DIR)).expanduser().resolve()

# Fallback path for local raw dataset
_LOCAL_RAW_DATA_DIR = Path(
    "/home/robakp/Exeriments1/prostate_lesion_detection/rjozwiak-MGR_dataset_correct/MGR_dataset_correct"
)


class Config:
    # Context Flag
    IS_COLAB: bool = IN_COLAB

    # Base Paths
    BASE_DIR: Path = BASE_DIR
    DATA_DIR: Path = BASE_DIR / "data"

    # Raw Data Resolution: Uses env var if set, falls back to local path if it exists locally, otherwise defaults to BASE_DIR / data / raw
    RAW_DATA_DIR: Path = Path(
        os.environ.get(
            "RAW_DATA_DIR",
            _LOCAL_RAW_DATA_DIR if (_LOCAL_RAW_DATA_DIR.exists() and not IN_COLAB) else DATA_DIR / "raw"
        )
    ).resolve()

    # Derived Paths (Using absolute Path objects rather than relative string paths)
    PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
    TRAIN_DIR: Path = PROCESSED_DATA_DIR / "train"
    TEST_DIR: Path = PROCESSED_DATA_DIR / "test"
    MODELS_DIR: Path = BASE_DIR / "models"
    SAVED_WEIGHTS_DIR: Path = Path(os.environ.get("SAVED_WEIGHTS_DIR", BASE_DIR / "saved_weights")).expanduser().resolve()
    LOG_DIR: Path = BASE_DIR / "logs"
    TRAINING_LOG_PATH: Path = LOG_DIR / "training.log"

    # Preprocessing & Volumetric Parameters
    CROP_SIZE = (160, 160, 24)
    TARGET_SPACING = (0.8, 0.8, 3.5)
    IN_CHANNELS = 4
    OUT_CHANNELS = 1
    BASE_FILTERS = 32

    # Training Hyperparameters
    BATCH_SIZE = 2
    EPOCHS = 60
    LR = 2e-4
    NUM_WORKERS = 2 if IN_COLAB else 4  # Colab free tier is prone to shared memory bottlenecks with higher worker counts

    @classmethod
    def setup_directories(cls) -> None:
        """Utility method to ensure necessary directories exist before training."""
        for path in [
            cls.DATA_DIR,
            cls.PROCESSED_DATA_DIR,
            cls.TRAIN_DIR,
            cls.TEST_DIR,
            cls.SAVED_WEIGHTS_DIR,
            cls.LOG_DIR,
        ]:
            path.mkdir(parents=True, exist_ok=True)


# Auto-create output directories upon import
Config.setup_directories()