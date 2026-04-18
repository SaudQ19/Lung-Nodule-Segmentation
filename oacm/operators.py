"""
oacm/operators.py
-----------------
Low-level GPU tensor operators used by the Optimized Active Contour Model
(OACM) — Yang et al., Medical Physics 2024, Equation 2 / Algorithm 1.

All functions operate on 2-D (H, W) PyTorch tensors unless stated otherwise.
They are kept separate from the refinement loop so they can be unit-tested
and reused independently.

Mathematical reference
----------------------
Energy functional (Eq. 2):

  E(u, c1, c2) = λ ∫[(I-c1)²u + (I-c2)²(1-u)] dx
               + √(π/τ) ∫ u G_τ*(1-u) dx
               + ∫ β(I)|∇²u| dx

ADMM subproblems (Eqs. 24-26):
  ū   : closed-form sign of ψ gradient  (Eq. 30)
  p   : soft-thresholding               (Eq. 48 / Theorem 3)
  q   : gradient ascent                 (Eq. 49)
"""

import numpy as np
import torch
import torch.nn.functional as F


# ── Gaussian kernel (heat kernel, Eq. 3) ─────────────────────────────────────

def make_gaussian_kernel(tau: float, size: int, device: torch.device) -> torch.Tensor:
    """Build a normalised Gaussian kernel G_τ as a (1,1,H,W) conv weight.

    Parameters
    ----------
    tau    : kernel variance (controls spatial reach of the heat term)
    size   : kernel spatial size (forced odd; minimum 3)
    device : target device

    Returns
    -------
    Tensor of shape (1, 1, size, size)
    """
    size = max(int(size) | 1, 3)   # ensure odd and >= 3
    ax   = torch.arange(-(size // 2), size // 2 + 1, dtype=torch.float32)
    xx, yy = torch.meshgrid(ax, ax, indexing="ij")
    k = torch.exp(-(xx**2 + yy**2) / (4 * tau + 1e-12)) / (4 * np.pi * tau + 1e-12)
    k = k / (k.sum() + 1e-12)
    return k.unsqueeze(0).unsqueeze(0).to(device)


# ── Finite-difference gradient ────────────────────────────────────────────────

def gradient(u: torch.Tensor, axis: int) -> torch.Tensor:
    """Central-difference gradient of *u* (H, W) along *axis* (0=y, 1=x)."""
    g = torch.zeros_like(u)
    if axis == 0:
        g[1:-1] = (u[2:] - u[:-2]) / 2.0
        g[0]    = u[1] - u[0]
        g[-1]   = u[-1] - u[-2]
    else:
        g[:, 1:-1] = (u[:, 2:] - u[:, :-2]) / 2.0
        g[:, 0]    = u[:, 1] - u[:, 0]
        g[:, -1]   = u[:, -1] - u[:, -2]
    return g


# ── Hessian and its Frobenius norm (Eq. 4) ────────────────────────────────────

def hessian_components(u: torch.Tensor):
    """Return (u_xx, u_xy, u_yx, u_yy) using central differences."""
    gx  = gradient(u, axis=1)
    gy  = gradient(u, axis=0)
    uxx = gradient(gx, axis=1)
    uxy = gradient(gx, axis=0)
    uyx = gradient(gy, axis=1)
    uyy = gradient(gy, axis=0)
    return uxx, uxy, uyx, uyy


def hessian_frobenius(u: torch.Tensor) -> torch.Tensor:
    """||∇²u||_F — Frobenius norm of the Hessian (Eq. 4)."""
    uxx, uxy, uyx, uyy = hessian_components(u)
    return torch.sqrt(uxx**2 + uxy**2 + uyx**2 + uyy**2 + 1e-12)


# ── Edge-adaptive weight β(I) (Eq. 5) ────────────────────────────────────────

def beta_weight(I: torch.Tensor) -> torch.Tensor:
    """β(I) = 1 / sqrt(1 + |∇I|²)  —  suppresses regularisation at edges."""
    Ix = gradient(I, axis=1)
    Iy = gradient(I, axis=0)
    return 1.0 / torch.sqrt(1.0 + Ix**2 + Iy**2 + 1e-12)


# ── Mean intensity updates (Eqs. 9-10) ───────────────────────────────────────

def update_c1_c2(u: torch.Tensor, I: torch.Tensor):
    """Global intensity means inside (c1) and outside (c2) the contour."""
    c1 = (I * u).sum() / (u.sum() + 1e-12)
    c2 = (I * (1.0 - u)).sum() / ((1.0 - u).sum() + 1e-12)
    return c1.item(), c2.item()


# ── Adjoint of the Hessian: div² ──────────────────────────────────────────────

def div2_operator(p: torch.Tensor) -> torch.Tensor:
    """Adjoint of ∇²; maps p (4, H, W) → scalar field (H, W).

    p[0..3] correspond to (u_xx, u_xy, u_yx, u_yy) channels.
    """
    d2_xx = gradient(gradient(p[0], axis=1), axis=1)
    d2_xy = gradient(gradient(p[1], axis=0), axis=1)
    d2_yx = gradient(gradient(p[2], axis=1), axis=0)
    d2_yy = gradient(gradient(p[3], axis=0), axis=0)
    return d2_xx + d2_xy + d2_yx + d2_yy


# ── Soft-thresholding (Theorem 3 / Eq. 48) ───────────────────────────────────

def soft_threshold_2d(S: torch.Tensor, thr: torch.Tensor) -> torch.Tensor:
    """Vector soft-thresholding operator.

    S   : (4, H, W) — stacked Hessian channels
    thr : scalar or broadcastable tensor

    Returns M* = max{|S| - thr, 0} ∘ S / |S|
    """
    S_frob = torch.sqrt((S**2).sum(dim=0, keepdim=True) + 1e-12)
    scale  = torch.clamp(S_frob - thr, min=0.0) / S_frob
    return scale * S


# ── Same-padded 2-D convolution ───────────────────────────────────────────────

def conv2d_same(u: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """2-D convolution with 'same' (half) padding.

    u      : (H, W)
    kernel : (1, 1, kH, kW)

    Returns (H, W) tensor.
    """
    kH, kW = kernel.shape[2], kernel.shape[3]
    pad    = (kH // 2, kW // 2)
    return (
        F.conv2d(u.unsqueeze(0).unsqueeze(0), kernel, padding=pad)
        .squeeze(0)
        .squeeze(0)
    )


# ── ψ gradient (Eq. 29) ───────────────────────────────────────────────────────

def psi_gradient(
    u_bar: torch.Tensor,
    p: torch.Tensor,
    q: torch.Tensor,
    I: torch.Tensor,
    c1: float,
    c2: float,
    G_tau: torch.Tensor,
    lam: float,
    tau: float,
    theta: float,
) -> torch.Tensor:
    """Compute ψ(ū) — the sub-gradient used to update ū (Eq. 29).

    Parameters
    ----------
    u_bar : current contour iterate  (H, W)
    p     : auxiliary variable       (4, H, W)
    q     : Lagrange multiplier      (4, H, W)
    I     : normalised CT image      (H, W)
    c1    : mean intensity inside contour
    c2    : mean intensity outside contour
    G_tau : Gaussian kernel          (1, 1, kH, kW)
    lam   : data-fidelity weight  λ
    tau   : kernel variance       τ
    theta : ADMM penalty          θ
    """
    scale     = float(np.sqrt(np.pi / (tau + 1e-12)))
    heat_term = scale * conv2d_same(1.0 - 2.0 * u_bar, G_tau)
    data_term = lam * ((I - c1) ** 2 - (I - c2) ** 2)
    uxx, uxy, uyx, uyy = hessian_components(u_bar)
    nabla2_u  = torch.stack([uxx, uxy, uyx, uyy], dim=0)
    reg_term  = theta * div2_operator(nabla2_u - p - q / (theta + 1e-12))
    return heat_term + data_term + reg_term
