"""Configure project paths for local development and Google Colab."""

import os
from pathlib import Path

IN_COLAB = bool(os.environ.get("COLAB_RELEASE_TAG")) or Path("/content/drive").exists()
_DEFAULT_BASE_DIR = (
    Path("/content/drive/MyDrive/ai_project")
    if IN_COLAB
    else Path(__file__).resolve().parents[1]
)
BASE_DIR = Path(os.environ.get("AI_PROJECT_DIR", _DEFAULT_BASE_DIR)).expanduser().resolve()
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"
SAVED_WEIGHTS_DIR = MODELS_DIR / "saved_weights"
