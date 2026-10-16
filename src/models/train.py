import glob
import logging
import os
from contextlib import nullcontext
from time import perf_counter

import torch
from tqdm import tqdm
from torch.utils.data import DataLoader

from src.config import Config
from src.data.dataset import ProstateLesionDataset
from src.models.architectures import AnisotropicUNet3D
from src.utils.metrics import FocalDiceLoss, calculate_dice_score


LOGGER = logging.getLogger(__name__)


def _configure_logging() -> None:
    """Configure console and persistent logging for a training run."""
    if LOGGER.handlers:
        return

    LOGGER.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    LOGGER.addHandler(console_handler)

    file_handler = logging.FileHandler(Config.TRAINING_LOG_PATH)
    file_handler.setFormatter(formatter)
    LOGGER.addHandler(file_handler)
    LOGGER.propagate = False


def train() -> None:
    _configure_logging()
    run_started = perf_counter()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(Config.SAVED_WEIGHTS_DIR, exist_ok=True)

    train_files = sorted(glob.glob(os.path.join(Config.TRAIN_DIR, "**", "*.pkl"), recursive=True))
    test_files = sorted(glob.glob(os.path.join(Config.TEST_DIR, "**", "*.pkl"), recursive=True))

    if not train_files:
        raise FileNotFoundError(f"No training pickle files found under {Config.TRAIN_DIR}. Check preprocessing output paths.")
    if not test_files:
        raise FileNotFoundError(f"No validation pickle files found under {Config.TEST_DIR}. Check preprocessing output paths.")

    LOGGER.info(
        "Starting training: device=%s, epochs=%d, train_samples=%d, validation_samples=%d, batch_size=%d",
        device,
        Config.EPOCHS,
        len(train_files),
        len(test_files),
        Config.BATCH_SIZE,
    )

    train_ds = ProstateLesionDataset(train_files, is_train=True)
    test_ds = ProstateLesionDataset(test_files, is_train=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=max(1, Config.NUM_WORKERS // 2),
        pin_memory=torch.cuda.is_available(),
    )

    model = AnisotropicUNet3D(
        in_channels=Config.IN_CHANNELS,
        out_channels=Config.OUT_CHANNELS,
        base_filters=Config.BASE_FILTERS,
    ).to(device)

    criterion = FocalDiceLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.EPOCHS, eta_min=1e-6)

    try:
        scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())
    except AttributeError:
        scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    best_val_dice = 0.0

    def autocast_context():
        if torch.cuda.is_available():
            try:
                return torch.autocast(device_type="cuda")
            except TypeError:
                return torch.cuda.amp.autocast()
        return nullcontext()

    for epoch in range(1, Config.EPOCHS + 1):
        epoch_started = perf_counter()
        model.train()
        train_loss = 0.0

        train_progress = tqdm(
            train_loader,
            desc=f"Epoch {epoch:02d}/{Config.EPOCHS} [train]",
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
                test_loader,
                desc=f"Epoch {epoch:02d}/{Config.EPOCHS} [validation]",
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

        val_loss /= len(test_loader.dataset)
        val_dice /= len(test_loader.dataset)

        LOGGER.info(
            "Epoch %02d/%d | train_loss=%.4f | val_loss=%.4f | val_dice=%.4f | "
            "learning_rate=%.2e | duration=%.1fs",
            epoch,
            Config.EPOCHS,
            train_loss,
            val_loss,
            val_dice,
            optimizer.param_groups[0]["lr"],
            perf_counter() - epoch_started,
        )

        if val_dice > best_val_dice:
            best_val_dice = val_dice
            save_path = os.path.join(Config.SAVED_WEIGHTS_DIR, "best_prostate_unet3d.pth")
            torch.save(model.state_dict(), save_path)
            LOGGER.info("Saved checkpoint: %s (val_dice=%.4f)", save_path, best_val_dice)

    LOGGER.info(
        "Training complete in %.1fs | best_val_dice=%.4f | log_file=%s",
        perf_counter() - run_started,
        best_val_dice,
        Config.TRAINING_LOG_PATH,
    )
