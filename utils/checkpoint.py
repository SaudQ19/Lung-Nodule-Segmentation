"""
utils/checkpoint.py
-------------------
Helpers for saving and loading PyTorch model checkpoints.

Checkpoint format
-----------------
A checkpoint is a dict with the following keys:

    epoch              : int  — epoch index when saved (0-based)
    global_step        : int  — total optimizer steps
    model_state_dict   : OrderedDict — model weights
    optimizer_state_dict : OrderedDict
    history            : dict — training curves (optional)
"""

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    global_step: int,
    history: Optional[dict] = None,
) -> None:
    """Save model + optimizer state to *path*.

    Parameters
    ----------
    path        : destination file path (.pt / .pth)
    model       : the model whose weights to save
    optimizer   : optimizer whose state to save
    epoch       : current epoch (saved so training can resume)
    global_step : cumulative optimizer steps
    history     : optional dict of training-curve lists
    """
    torch.save(
        {
            "epoch":                epoch,
            "global_step":          global_step,
            "model_state_dict":     model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "history":              history,
        },
        path,
    )


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    map_location: str | torch.device = "cpu",
    strict: bool = True,
) -> tuple:
    """Load weights (and optionally optimizer state) from *path*.

    Parameters
    ----------
    path         : checkpoint file to load
    model        : model to load weights into (modified in-place)
    optimizer    : if provided, load optimizer state too
    map_location : passed to ``torch.load``
    strict       : if ``False``, silently skip mismatched keys

    Returns
    -------
    (epoch, global_step, history) — defaults to (0, 0, None) for legacy
    checkpoints that store only the state dict.
    """
    ckpt = torch.load(path, map_location=map_location)

    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"], strict=strict)
        if optimizer is not None and "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        epoch       = ckpt.get("epoch",       0)
        global_step = ckpt.get("global_step", 0)
        history     = ckpt.get("history",     None)
    else:
        # Legacy format: bare state dict
        model.load_state_dict(ckpt, strict=strict)
        epoch, global_step, history = 0, 0, None

    return epoch, global_step, history
