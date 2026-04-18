"""
models/losses.py
----------------
Loss functions and training-time metric helpers for lung nodule segmentation.

BCEDiceLoss
    Weighted sum of binary cross-entropy (with optional positive-class
    reweighting for class imbalance), soft Dice loss, and focal loss.

dice_coef_from_logits
    Fast, no-grad Dice coefficient for logging during training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BCEDiceLoss(nn.Module):
    """Combined BCE + Dice + Focal loss.

    Parameters
    ----------
    pos_weight   : BCE positive-class weight (handles foreground/background
                   imbalance; set higher when nodules are small)
    bce_weight   : weight applied to the BCE term
    dice_weight  : weight applied to the soft-Dice term
    focal_weight : weight applied to the focal term
    focal_gamma  : focusing parameter γ for the focal term
    smooth       : Laplace smoothing for the Dice denominator
    """

    def __init__(
        self,
        pos_weight:   float = 15.0,
        bce_weight:   float = 1.0,
        dice_weight:  float = 1.0,
        focal_weight: float = 0.5,
        focal_gamma:  float = 2.0,
        smooth:       float = 1e-3,
    ) -> None:
        super().__init__()
        self.pos_weight   = pos_weight
        self.bce_weight   = bce_weight
        self.dice_weight  = dice_weight
        self.focal_weight = focal_weight
        self.focal_gamma  = focal_gamma
        self.smooth       = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        pos_w = torch.tensor(
            [self.pos_weight], dtype=logits.dtype, device=logits.device
        )

        # ── BCE term ──────────────────────────────────────────────────────────
        bce = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_w)

        # ── Focal term ────────────────────────────────────────────────────────
        probs_f   = torch.sigmoid(logits)
        bce_pixel = F.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        )
        p_t   = probs_f * targets + (1 - probs_f) * (1 - targets)
        focal = ((1 - p_t) ** self.focal_gamma * bce_pixel).mean()

        # ── Soft-Dice term ────────────────────────────────────────────────────
        probs  = torch.sigmoid(logits)
        p_flat = probs.view(probs.size(0), -1)
        t_flat = targets.view(targets.size(0), -1)
        inter  = (p_flat * t_flat).sum(dim=1)
        den    = p_flat.sum(dim=1) + t_flat.sum(dim=1)
        dice   = (
            1.0 - (2.0 * inter + self.smooth) / (den + self.smooth)
        ).mean()

        return (
            self.bce_weight   * bce
            + self.dice_weight  * dice
            + self.focal_weight * focal
        )


@torch.no_grad()
def dice_coef_from_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.3,
    smooth: float = 1e-6,
) -> float:
    """Compute batch-mean Dice from raw logits (no gradient).

    Used for logging during training; not for final evaluation
    (see :func:`utils.metrics.compute_metrics` for that).
    """
    probs   = torch.sigmoid(logits)
    preds   = (probs >= threshold).float()
    preds   = preds.view(preds.size(0), -1)
    targets = targets.view(targets.size(0), -1)
    inter   = (preds * targets).sum(dim=1)
    den     = preds.sum(dim=1) + targets.sum(dim=1)
    return ((2.0 * inter + smooth) / (den + smooth)).mean().item()
