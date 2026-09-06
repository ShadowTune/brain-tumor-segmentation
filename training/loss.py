"""
Loss functions for BraTS 3D segmentation.

ImprovedLoss = 0.4 * DiceLoss + 0.3 * WeightedCE + 0.3 * TverskyFocalLoss
  + deep supervision (weighted auxiliary heads)

Why this combination?
  DiceLoss:         directly optimizes the Dice metric; handles class imbalance.
  WeightedCE:       provides dense per-voxel gradients; class weights [0.25,2.5,1.5,3.0]
                    upweight the small ET class (label 3).
  TverskyFocal:     Tversky loss with focal modulation penalises false negatives
                    heavily (beta=0.7) — critical for small ET which is easy to miss.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TverskyFocalLoss(nn.Module):
    """
    Tversky loss with focal modulation.

    Tversky index:  TI = TP / (TP + α·FP + β·FN)
    Tversky loss:   1 - TI
    Focal version:  (1 - TI)^γ

    alpha=0.3, beta=0.7: penalises false negatives more (avoids missing ET).
    gamma=2.0: focuses training on hard examples (same as focal loss).
    """

    def __init__(self, num_classes=4, alpha=0.3, beta=0.7, gamma=2.0, smooth=1e-6):
        super().__init__()
        self.nc    = num_classes
        self.alpha = alpha
        self.beta  = beta
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, probs, target_oh):
        """
        Args:
            probs:     softmax probabilities (B, C, D, H, W)
            target_oh: one-hot target       (B, C, D, H, W)
        """
        dims = (0, 2, 3, 4)
        TP  = (probs * target_oh).sum(dims)
        FP  = (probs * (1 - target_oh)).sum(dims)
        FN  = ((1 - probs) * target_oh).sum(dims)
        TI  = (TP + self.smooth) / (TP + self.alpha * FP + self.beta * FN + self.smooth)
        # Skip background class (index 0) — only average over foreground
        return ((1 - TI).pow(self.gamma)[1:]).mean()


class ImprovedLoss(nn.Module):
    """
    Combined segmentation loss with deep supervision.

    Main loss = 0.4·Dice + 0.3·WeightedCE + 0.3·TverskyFocal
    If model returns a tuple (main, aux5, aux4, aux3):
        total = main_loss + 0.2·aux5 + 0.2·aux4 + 0.1·aux3
    """

    def __init__(self, num_classes=4, smooth=1e-6,
                 class_weights=(0.25, 2.5, 1.5, 3.0)):
        super().__init__()
        self.nc      = num_classes
        self.smooth  = smooth
        cw           = torch.tensor(class_weights, dtype=torch.float32)
        self.ce      = nn.CrossEntropyLoss(weight=cw)
        self.tversky = TverskyFocalLoss(num_classes)

    def to(self, device):
        self.ce = nn.CrossEntropyLoss(weight=self.ce.weight.to(device))
        return super().to(device)

    def _dice(self, probs, target_oh):
        dims  = (0, 2, 3, 4)
        inter = (probs * target_oh).sum(dims)
        union = probs.sum(dims) + target_oh.sum(dims)
        return 1.0 - ((2.0 * inter + self.smooth) / (union + self.smooth))[1:].mean()

    def main_loss(self, logits, target, target_oh):
        probs = F.softmax(logits, dim=1)
        return (
            0.4 * self._dice(probs, target_oh)
            + 0.3 * self.ce(logits, target)
            + 0.3 * self.tversky(probs, target_oh)
        )

    def forward(self, outputs, target):
        # One-hot encoding computed once, shared across all loss terms
        with torch.no_grad():
            target_oh = (
                F.one_hot(target, self.nc)
                .permute(0, 4, 1, 2, 3)
                .float()
            )

        if isinstance(outputs, tuple):
            main, aux5, aux4, aux3 = outputs
            loss  = self.main_loss(main,  target, target_oh)
            loss += 0.2 * self.main_loss(aux5, target, target_oh)
            loss += 0.2 * self.main_loss(aux4, target, target_oh)
            loss += 0.1 * self.main_loss(aux3, target, target_oh)
            return loss

        return self.main_loss(outputs, target, target_oh)
