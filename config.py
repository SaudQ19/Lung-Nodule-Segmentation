#this file

"""
config.py
---------
Central configuration dataclass for the Lung Nodule Segmentation pipeline.
All hyperparameters, dataset paths, and OACM settings live here.
Edit this file (or pass overrides at runtime) — do not scatter magic numbers
across source files.
"""

from dataclasses import dataclass
from typing import Tuple


@dataclass
class Config:
    # ── Data paths ───────────────────────────────────────────────────────────
    data_root:   str = "data/lidc_dataset"
    train_split: str = "train"
    val_split:   str = "test"

    # ── CT windowing (Yang et al. 2024, Section 3.1.1) ───────────────────────
    wc: float = -450.0   # window centre
    ww: float = 1000.0   # window width

    # ── 2.5D context ─────────────────────────────────────────────────────────
    # Each sample is a stack of (2*context_slices + 1) adjacent CT slices.
    # Default: prev2 + current + next2 = 5 input channels.
    context_slices: int = 2
    target_size: Tuple[int, int] = (256, 256)

    # ── Training ─────────────────────────────────────────────────────────────
    batch_size:    int   = 16
    num_workers:   int   = 0
    epochs:        int   = 18       # epochs before early-stop / custom weights
    total_epochs:  int   = 40       # hard ceiling if training from scratch
    lr:            float = 3e-4
    weight_decay:  float = 1e-4
    grad_clip:     float = 1.0

    # ── Loss weights ─────────────────────────────────────────────────────────
    pos_weight:   float = 1e2   # BCE positive-class weight (nodule/background imbalance)
    dice_weight:  float = 1.0
    bce_weight:   float = 1.0
    focal_weight: float = 1.0

    # ── U-Net capacity ───────────────────────────────────────────────────────
    unet_base: int = 128            # base channel width; scales encoder/decoder

    # ── Checkpointing ────────────────────────────────────────────────────────
    out_dir:             str  = "checkpoints"
    save_every_iters:    int  = 100
    save_every_epochs:   int  = 10
    load_custom_weights: bool = True
    custom_weights_path: str  = "weights/model_best.pt"
    model_weights_path:  str  = "checkpoints/model_best.pt"

    # ── Stage 2 — OACM hyperparameters (Eq. 2 / Algorithm 1 of Yang et al.) ──
    # Soft-masking variant (pass U-Net probability map directly)
    lam:   float = 210.0    # data-fidelity balance  λ
    theta: float = 2.0      # ADMM penalty parameter  θ
    tau:   float = 0.001    # Gaussian kernel variance  τ
    l_max: int   = 50       # outer iterations  L_max
    k_max: int   = 1        # inner ADMM iterations  K_max
    eps:   float = 1e-5     # convergence tolerance

    # Hard-masking variant (binarise probability map first)
    lam2:   float = 104.0
    theta2: float = 3.0
    tau2:   float = 0.01
    l_max2: int   = 50
    k_max2: int   = 1

    # ── Inference ────────────────────────────────────────────────────────────
    cnn_threshold:  float = 0.4   # probability threshold for Stage 1 binary mask
    n_vis_samples:  int   = 5     # number of samples shown in visualisation plots

    # ── Reproducibility ──────────────────────────────────────────────────────
    seed: int = 42
