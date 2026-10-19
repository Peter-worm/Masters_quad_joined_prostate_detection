import logging
from contextlib import nullcontext
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import ExperimentConfig
from src.data.dataset import ProstateLesionDataset
from src.models.architectures import AnisotropicUNet3D
from src.utils.metrics import FocalDiceLoss, calculate_dice_score


LOGGER = logging.getLogger("src.models.train")


def _autocast_context():
    if torch.cuda.is_available():
        try:
            return torch.autocast(device_type="cuda")
        except TypeError:
            return torch.cuda.amp.autocast()
    return nullcontext()


def make_model(config: ExperimentConfig, device: torch.device) -> torch.nn.Module:
    return AnisotropicUNet3D(
        in_channels=config.in_channels,
        out_channels=config.out_channels,
        base_filters=config.base_filters,
    ).to(device)


def evaluate(
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


def train_fold(
    config: ExperimentConfig,
    fold_number: int,
    train_files: list[Path],
    validation_files: list[Path],
    device: torch.device,
) -> tuple[Path, float]:
    train_loader = DataLoader(
        ProstateLesionDataset(train_files, is_train=True),
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    validation_loader = DataLoader(
        ProstateLesionDataset(validation_files, is_train=False),
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=max(1, config.num_workers // 2),
        pin_memory=torch.cuda.is_available(),
    )

    model = make_model(config, device)
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
    checkpoint_name = (
        f"cross_validation_fold_{fold_number}.pth"
        if config.use_cross_validation
        else "train_validation.pth"
    )
    checkpoint_path = config.saved_weights_dir / checkpoint_name

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
        val_loss, val_dice = evaluate(
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
