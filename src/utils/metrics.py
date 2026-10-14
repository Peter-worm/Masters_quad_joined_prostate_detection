import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalDiceLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, smooth=1e-5):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, inputs, targets):
        probs = torch.sigmoid(inputs)
        bce = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_loss = (self.alpha * (1 - p_t) ** self.gamma * bce).mean()

        inputs_flat = probs.view(-1)
        targets_flat = targets.view(-1)
        intersection = (inputs_flat * targets_flat).sum()
        dice_loss = 1.0 - (2.0 * intersection + self.smooth) / (
            inputs_flat.sum() + targets_flat.sum() + self.smooth
        )

        return focal_loss + dice_loss

def calculate_dice_score(preds, targets, smooth=1e-5):
    preds_binary = (torch.sigmoid(preds) > 0.5).float()
    preds_flat = preds_binary.view(-1)
    targets_flat = targets.view(-1)
    intersection = (preds_flat * targets_flat).sum()
    return (2.0 * intersection + smooth) / (preds_flat.sum() + targets_flat.sum() + smooth)