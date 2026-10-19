import logging
from contextlib import nullcontext
from time import perf_counter

import torch
from tqdm import tqdm
from torch.utils.data import DataLoader

from src.config import ExperimentConfig
from src.data.dataset import ProstateLesionDataset
from src.models.architectures import AnisotropicUNet3D
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


def train(config: ExperimentConfig | None = None) -> None:
    config = config or ExperimentConfig()
    config.setup_directories()
    _configure_logging(config)
    run_started = perf_counter()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_files = sorted(config.train_dir.glob("**/*.pkl"))
    validation_files = sorted(config.validation_dir.glob("**/*.pkl"))
    test_files = sorted(config.test_dir.glob("**/*.pkl"))

    if not train_files:
        raise FileNotFoundError(f"No training pickle files found under {config.train_dir}. Check preprocessing output paths.")
    if not validation_files:
        raise FileNotFoundError(
            f"No validation pickle files found under {config.validation_dir}. "
            "Check preprocessing output paths."
        )
    if not test_files:
        raise FileNotFoundError(
            f"No test pickle files found under {config.test_dir}. "
            "Check preprocessing output paths."
        )

    LOGGER.info(
        "Starting training: device=%s, epochs=%d, train_samples=%d, "
        "validation_samples=%d, test_samples=%d, batch_size=%d",
        device,
        config.epochs,
        len(train_files),
        len(validation_files),
        len(test_files),
        config.batch_size,
    )

    train_ds = ProstateLesionDataset(train_files, is_train=True)
    validation_ds = ProstateLesionDataset(validation_files, is_train=False)
    test_ds = ProstateLesionDataset(test_files, is_train=False)

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
    test_loader = DataLoader(
        test_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=max(1, config.num_workers // 2),
        pin_memory=torch.cuda.is_available(),
    )

    model = AnisotropicUNet3D(
        in_channels=config.in_channels,
        out_channels=config.out_channels,
        base_filters=config.base_filters,
    ).to(device)

    criterion = FocalDiceLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs, eta_min=1e-6)

    try:
        scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())
    except AttributeError:
        scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    best_val_dice = float("-inf")

    def autocast_context():
        if torch.cuda.is_available():
            try:
                return torch.autocast(device_type="cuda")
            except TypeError:
                return torch.cuda.amp.autocast()
        return nullcontext()

    for epoch in range(1, config.epochs + 1):
        epoch_started = perf_counter()
        model.train()
        train_loss = 0.0

        train_progress = tqdm(
            train_loader,
            desc=f"Epoch {epoch:02d}/{config.epochs} [train]",
            unit="batch",
        )
        for inputs, targets in train_progress:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()

            with autocast_context():
                outputs = model(inputs)
                loss = criterion(outputs, targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item() * inputs.size(0)
            train_progress.set_postfix(loss=f"{loss.item():.4f}")

        train_loss /= len(train_loader.dataset)
        scheduler.step()

        model.eval()
        val_loss, val_dice = 0.0, 0.0

        with torch.no_grad():
            validation_progress = tqdm(
                validation_loader,
                desc=f"Epoch {epoch:02d}/{config.epochs} [validation]",
                unit="batch",
            )
            for inputs, targets in validation_progress:
                inputs, targets = inputs.to(device), targets.to(device)
                with autocast_context():
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)

                val_loss += loss.item() * inputs.size(0)
                val_dice += calculate_dice_score(outputs, targets).item() * inputs.size(0)
                validation_progress.set_postfix(loss=f"{loss.item():.4f}")

        val_loss /= len(validation_loader.dataset)
        val_dice /= len(validation_loader.dataset)

        LOGGER.info(
            "Epoch %02d/%d | train_loss=%.4f | val_loss=%.4f | val_dice=%.4f | "
            "learning_rate=%.2e | duration=%.1fs",
            epoch,
            config.epochs,
            train_loss,
            val_loss,
            val_dice,
            optimizer.param_groups[0]["lr"],
            perf_counter() - epoch_started,
        )

        if val_dice > best_val_dice:
            best_val_dice = val_dice
            save_path = config.saved_weights_dir / "best_prostate_unet3d.pth"
            torch.save(model.state_dict(), save_path)
            LOGGER.info("Saved checkpoint: %s (val_dice=%.4f)", save_path, best_val_dice)

    best_checkpoint = config.saved_weights_dir / "best_prostate_unet3d.pth"
    model.load_state_dict(torch.load(best_checkpoint, map_location=device))
    model.eval()
    test_loss, test_dice = 0.0, 0.0

    with torch.no_grad():
        test_progress = tqdm(
            test_loader,
            desc="Final evaluation [test]",
            unit="batch",
        )
        for inputs, targets in test_progress:
            inputs, targets = inputs.to(device), targets.to(device)
            with autocast_context():
                outputs = model(inputs)
                loss = criterion(outputs, targets)

            test_loss += loss.item() * inputs.size(0)
            test_dice += calculate_dice_score(outputs, targets).item() * inputs.size(0)
            test_progress.set_postfix(loss=f"{loss.item():.4f}")

    test_loss /= len(test_loader.dataset)
    test_dice /= len(test_loader.dataset)
    LOGGER.info(
        "Final test evaluation | test_loss=%.4f | test_dice=%.4f",
        test_loss,
        test_dice,
    )
    LOGGER.info(
        "Training complete in %.1fs | best_val_dice=%.4f | test_dice=%.4f | log_file=%s",
        perf_counter() - run_started,
        best_val_dice,
        test_dice,
        config.training_log_path,
    )
