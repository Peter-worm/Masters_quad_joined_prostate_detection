import logging
from pathlib import Path
from time import perf_counter

import torch
from torch.utils.data import DataLoader

from src.config import ExperimentConfig
from src.data.dataset import ProstateLesionDataset
from src.models.training_routines import evaluate, make_model, train_fold
from src.utils.cross_validation import create_folds, split_train_validation
from src.utils.metrics import FocalDiceLoss


LOGGER = logging.getLogger(__name__)


def _configure_logging(config: ExperimentConfig) -> None:
    """Configure console and persistent logging for a training run."""
    if LOGGER.handlers:
        return

    LOGGER.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    LOGGER.addHandler(console_handler)

    file_handler = logging.FileHandler(config.training_log_path)
    file_handler.setFormatter(formatter)
    LOGGER.addHandler(file_handler)
    LOGGER.propagate = False


def train(config: ExperimentConfig | None = None) -> None:
    config = config or ExperimentConfig()
    config.setup_directories()
    _configure_logging(config)
    run_started = perf_counter()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if config.use_cross_validation:
        train_val_files = sorted(config.train_val_dir.glob("**/*.pkl"))
        train_files, validation_files = [], []
    else:
        train_val_files = sorted(config.train_val_dir.glob("**/*.pkl"))
        train_files, validation_files = split_train_validation(
            [str(path) for path in train_val_files],
            validation_fraction=config.validation_fraction,
            random_seed=config.cross_validation_seed,
        )
        train_files = [Path(path) for path in train_files]
        validation_files = [Path(path) for path in validation_files]
    test_files = sorted(config.test_dir.glob("**/*.pkl"))
    if not test_files:
        raise FileNotFoundError(f"Test directory is empty: {config.test_dir}.")

    if config.use_cross_validation:
        if not train_val_files:
            raise FileNotFoundError(
                f"Train/validation directory is empty: {config.train_val_dir}."
            )

        folds = create_folds(
            [str(path) for path in train_val_files],
            n_splits=config.cross_validation_folds,
            random_seed=config.cross_validation_seed,
        )
        fold_scores: list[float] = []
        fold_checkpoints: list[Path] = []

        LOGGER.info(
            "Starting %d-fold cross-validation: device=%s, epochs=%d, "
            "cv_samples=%d, test_samples=%d, batch_size=%d",
            config.cross_validation_folds,
            device,
            config.epochs,
            len(train_val_files),
            len(test_files),
            config.batch_size,
        )

        for fold_number, validation_fold in enumerate(folds, start=1):
            validation_set = set(validation_fold)
            fold_train = [
                Path(path) for path in cv_files if str(path) not in validation_set
            ]
            fold_validation = [Path(path) for path in validation_fold]
            checkpoint, score = train_fold(
                config,
                fold_number,
                fold_train,
                fold_validation,
                device,
            )
            fold_checkpoints.append(checkpoint)
            fold_scores.append(score)
            LOGGER.info("Fold %d complete | best_val_dice=%.4f", fold_number, score)

        mean_score = sum(fold_scores) / len(fold_scores)
        variance = sum((score - mean_score) ** 2 for score in fold_scores) / len(fold_scores)
        std_score = variance**0.5
        best_fold_index = max(range(len(fold_scores)), key=fold_scores.__getitem__)
        best_checkpoint = fold_checkpoints[best_fold_index]
        evaluation_description = (
            "Cross-validation complete | fold_val_dice_mean=%.4f | "
            "fold_val_dice_std=%.4f | best_fold=%d"
        )
        evaluation_values: tuple[object, ...] = (
            mean_score,
            std_score,
            best_fold_index + 1,
        )
    else:
        if not train_files or not validation_files:
            raise FileNotFoundError(
                "Training requires non-empty train and validation directories: "
                f"train/validation pool={config.train_val_dir}."
            )
        LOGGER.info(
            "Starting train/validation training: device=%s, epochs=%d, "
            "train_samples=%d, validation_samples=%d, test_samples=%d, batch_size=%d",
            device,
            config.epochs,
            len(train_files),
            len(validation_files),
            len(test_files),
            config.batch_size,
        )
        best_checkpoint, best_val_dice = train_fold(
            config,
            1,
            train_files,
            validation_files,
            device,
        )
        evaluation_description = "Train/validation training complete | val_dice=%.4f"
        evaluation_values = (best_val_dice,)

    model = make_model(config, device)
    model.load_state_dict(torch.load(best_checkpoint, map_location=device))
    test_loader = DataLoader(
        ProstateLesionDataset(test_files, is_train=False),
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=max(1, config.num_workers // 2),
        pin_memory=torch.cuda.is_available(),
    )
    test_loss, test_dice = evaluate(
        model,
        test_loader,
        FocalDiceLoss(),
        device,
        "Final evaluation [test]",
    )
    torch.save(model.state_dict(), config.saved_weights_dir / "best_prostate_unet3d.pth")
    LOGGER.info(
        f"{evaluation_description} | test_loss=%.4f | test_dice=%.4f | "
        "duration=%.1fs",
        *evaluation_values,
        test_loss,
        test_dice,
        perf_counter() - run_started,
    )
