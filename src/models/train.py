import glob
import os
from contextlib import nullcontext

import torch
from torch.utils.data import DataLoader

from src.config import Config
from src.data.dataset import ProstateLesionDataset
from src.models.architectures import AnisotropicUNet3D
from src.utils.metrics import FocalDiceLoss, calculate_dice_score


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(Config.SAVED_WEIGHTS_DIR, exist_ok=True)

    train_files = sorted(glob.glob(os.path.join(Config.TRAIN_DIR, "**", "*.pkl"), recursive=True))
    test_files = sorted(glob.glob(os.path.join(Config.TEST_DIR, "**", "*.pkl"), recursive=True))

    if not train_files:
        raise FileNotFoundError(f"No training pickle files found under {Config.TRAIN_DIR}. Check preprocessing output paths.")
    if not test_files:
        raise FileNotFoundError(f"No validation pickle files found under {Config.TEST_DIR}. Check preprocessing output paths.")

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
        model.train()
        train_loss = 0.0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()

            with autocast_context():
                outputs = model(inputs)
                loss = criterion(outputs, targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item() * inputs.size(0)

        train_loss /= len(train_loader.dataset)
        scheduler.step()

        model.eval()
        val_loss, val_dice = 0.0, 0.0

        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                with autocast_context():
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)

                val_loss += loss.item() * inputs.size(0)
                val_dice += calculate_dice_score(outputs, targets).item() * inputs.size(0)

        val_loss /= len(test_loader.dataset)
        val_dice /= len(test_loader.dataset)

        print(f"Epoch {epoch:02d}/{Config.EPOCHS} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Dice: {val_dice:.4f}")

        if val_dice > best_val_dice:
            best_val_dice = val_dice
            save_path = os.path.join(Config.SAVED_WEIGHTS_DIR, "best_prostate_unet3d.pth")
            torch.save(model.state_dict(), save_path)
            print(f"--> Saved Checkpoint to {save_path} (Val Dice: {best_val_dice:.4f})")