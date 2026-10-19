import logging
from contextlib import nullcontext
from pathlib import Path
from time import perf_counter

import torch
from tqdm import tqdm
from torch.utils.data import DataLoader

from src.config import ExperimentConfig
from src.data.dataset import ProstateLesionDataset
from src.models.architectures import AnisotropicUNet3D
from src.utils.cross_validation import create_folds
from src.utils.metrics import FocalDiceLoss, calculate_dice_score


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


def _autocast_context():
    if torch.cuda.is_available():
        try:
            return torch.autocast(device_type="cuda")
        except TypeError:
            return torch.cuda.amp.autocast()
    return nullcontext()


def _make_model(config: ExperimentConfig, device: torch.device) -> torch.nn.Module:
    return AnisotropicUNet3D(
        in_channels=config.in_channels,
        out_channels=config.out_channels,
        base_filters=config.base_filters,
    ).to(device)


def _evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: FocalDiceLoss,
    device: torch.device,
    description: str,
) -> tuple[float, float]:
    model.eval()
    loss_total, dice_total = 0.0, 0.0

    with torch.no_grad():
        progress = tqdm(loader, desc=description, unit="batch")
        for inputs, targets in progress:
            inputs, targets = inputs.to(device), targets.to(device)
            with _autocast_context():
                outputs = model(inputs)
                loss = criterion(outputs, targets)

            loss_total += loss.item() * inputs.size(0)
            dice_total += calculate_dice_score(outputs, targets).item() * inputs.size(0)
            progress.set_postfix(loss=f"{loss.item():.4f}")

    return loss_total / len(loader.dataset), dice_total / len(loader.dataset)


def _train_fold(
    config: ExperimentConfig,
    fold_number: int,
    train_files: list[Path],
    validation_files: list[Path],
    device: torch.device,
) -> tuple[Path, float]:
    train_ds = ProstateLesionDataset(train_files, is_train=True)
    validation_ds = ProstateLesionDataset(validation_files, is_train=False)
    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    validation_loader = DataLoader(
        validation_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=max(1, config.num_workers // 2),
        pin_memory=torch.cuda.is_available(),
    )

    model = _make_model(config, device)
    criterion = FocalDiceLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.epochs, eta_min=1e-6
    )
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())
    except AttributeError:
        scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    best_val_dice = float("-inf")
    checkpoint_path = (
        config.saved_weights_dir / f"cross_validation_fold_{fold_number}.pth"
    )

    for epoch in range(1, config.epochs + 1):
        model.train()
        train_loss = 0.0
        progress = tqdm(
            train_loader,
            desc=f"Fold {fold_number}/{config.cross_validation_folds} "
            f"epoch {epoch:02d}/{config.epochs} [train]",
            unit="batch",
        )
        for inputs, targets in progress:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            with _autocast_context():
                outputs = model(inputs)
                loss = criterion(outputs, targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item() * inputs.size(0)
            progress.set_postfix(loss=f"{loss.item():.4f}")

        scheduler.step()
        train_loss /= len(train_loader.dataset)
        val_loss, val_dice = _evaluate(
            model,
            validation_loader,
            criterion,
            device,
            f"Fold {fold_number}/{config.cross_validation_folds} "
            f"epoch {epoch:02d}/{config.epochs} [validation]",
        )
        LOGGER.info(
            "Fold %d epoch %02d/%d | train_loss=%.4f | val_loss=%.4f | "
            "val_dice=%.4f | learning_rate=%.2e",
            fold_number,
            epoch,
            config.epochs,
            train_loss,
            val_loss,
            val_dice,
            optimizer.param_groups[0]["lr"],
        )

        if val_dice > best_val_dice:
            best_val_dice = val_dice
            torch.save(model.state_dict(), checkpoint_path)
            LOGGER.info(
                "Saved fold %d checkpoint: %s (val_dice=%.4f)",
                fold_number,
                checkpoint_path,
                best_val_dice,
            )

    return checkpoint_path, best_val_dice


def train(config: ExperimentConfig | None = None) -> None:
    config = config or ExperimentConfig()
    config.setup_directories()
    _configure_logging(config)
    run_started = perf_counter()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_files = sorted(config.train_dir.glob("**/*.pkl"))
    validation_files = sorted(config.validation_dir.glob("**/*.pkl"))
    test_files = sorted(config.test_dir.glob("**/*.pkl"))
    if not train_files or not validation_files or not test_files:
        raise FileNotFoundError(
            "Cross-validation requires non-empty train, validation, and test "
            f"directories: train={config.train_dir}, validation={config.validation_dir}, "
            f"test={config.test_dir}."
        )

    cv_files = train_files + validation_files
    folds = create_folds(
        [str(path) for path in cv_files],
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
        len(cv_files),
        len(test_files),
        config.batch_size,
    )

    for fold_number, validation_fold in enumerate(folds, start=1):
        validation_set = set(validation_fold)
        fold_train = [Path(path) for path in cv_files if str(path) not in validation_set]
        fold_validation = [Path(path) for path in validation_fold]
        checkpoint, score = _train_fold(
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
    model = _make_model(config, device)
    model.load_state_dict(torch.load(best_checkpoint, map_location=device))
    test_loader = DataLoader(
        ProstateLesionDataset(test_files, is_train=False),
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=max(1, config.num_workers // 2),
        pin_memory=torch.cuda.is_available(),
    )
    test_loss, test_dice = _evaluate(
        model,
        test_loader,
        FocalDiceLoss(),
        device,
        "Final evaluation [test]",
    )
    final_checkpoint = config.saved_weights_dir / "best_prostate_unet3d.pth"
    torch.save(model.state_dict(), final_checkpoint)
    LOGGER.info(
        "Cross-validation complete | fold_val_dice_mean=%.4f | "
        "fold_val_dice_std=%.4f | best_fold=%d | test_loss=%.4f | test_dice=%.4f | "
        "duration=%.1fs",
        mean_score,
        std_score,
        best_fold_index + 1,
        test_loss,
        test_dice,
        perf_counter() - run_started,
    )
