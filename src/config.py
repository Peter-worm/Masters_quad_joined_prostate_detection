import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


IN_COLAB = bool(os.environ.get("COLAB_RELEASE_TAG")) or Path("/content/drive").exists()

try:
    _SCRIPT_DIR = Path(__file__).resolve().parent
except NameError:
    _SCRIPT_DIR = Path.cwd()

_DEFAULT_BASE_DIR = (
    Path("/content/drive/MyDrive/ai_project")
    if IN_COLAB
    else _SCRIPT_DIR.parent
)
_LOCAL_RAW_DATA_DIR = Path(
    "/home/robakp/Exeriments1/prostate_lesion_detection/"
    "rjozwiak-MGR_dataset_correct/MGR_dataset_correct"
)


@dataclass
class ExperimentConfig:
    is_colab: bool = IN_COLAB
    base_dir: Path = field(
        default_factory=lambda: Path(
            os.environ.get("AI_PROJECT_DIR", _DEFAULT_BASE_DIR)
        )
    )
    raw_data_dir: Path | None = None

    data_dir: Path = field(init=False)
    processed_data_dir: Path = field(init=False)
    train_dir: Path = field(init=False)
    validation_dir: Path = field(init=False)
    test_dir: Path = field(init=False)
    models_dir: Path = field(init=False)
    saved_weights_dir: Path = field(init=False)
    log_dir: Path = field(init=False)
    training_log_path: Path = field(init=False)

    crop_size: tuple[int, int, int] = (160, 160, 24)
    target_spacing: tuple[float, float, float] = (0.8, 0.8, 3.5)
    in_channels: int = 4
    out_channels: int = 1
    base_filters: int = 32

    batch_size: int = 2
    epochs: int = 60
    lr: float = 2e-4
    num_workers: int = 2 if IN_COLAB else 4
    cross_validation_folds: int = 5
    cross_validation_seed: int = 42

    def __post_init__(self) -> None:
        self.base_dir = self.base_dir.expanduser().resolve()

        if self.raw_data_dir is None:
            env_raw_data = os.environ.get("RAW_DATA_DIR")
            if env_raw_data:
                self.raw_data_dir = Path(env_raw_data).expanduser().resolve()
            elif _LOCAL_RAW_DATA_DIR.exists() and not self.is_colab:
                self.raw_data_dir = _LOCAL_RAW_DATA_DIR
            else:
                self.raw_data_dir = self.base_dir / "data" / "raw"
        else:
            self.raw_data_dir = self.raw_data_dir.expanduser().resolve()

        self.data_dir = self.base_dir / "data"
        self.processed_data_dir = self.data_dir / "processed"
        self.train_dir = self.processed_data_dir / "train"
        self.validation_dir = self.processed_data_dir / "validation"
        self.test_dir = self.processed_data_dir / "test"
        self.models_dir = self.base_dir / "models"
        self.saved_weights_dir = Path(
            os.environ.get("SAVED_WEIGHTS_DIR", self.base_dir / "saved_weights")
        ).expanduser().resolve()
        self.log_dir = self.base_dir / "logs"
        self.training_log_path = self.log_dir / "training.log"

    def setup_directories(self) -> None:
        for path in (
            self.data_dir,
            self.processed_data_dir,
            self.train_dir,
            self.validation_dir,
            self.test_dir,
            self.models_dir,
            self.saved_weights_dir,
            self.log_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)

    def to_dict(self) -> dict[str, Any]:
        return {
            key: str(value) if isinstance(value, Path) else value
            for key, value in asdict(self).items()
        }
