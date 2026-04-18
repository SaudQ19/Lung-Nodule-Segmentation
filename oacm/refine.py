"""
oacm/refine.py
--------------
Stage 2 refinement: Optimized Active Contour Model (OACM).

Given a CT image I and an initial binary/soft mask u⁰ produced by the U-Net,
`oacm_refine` runs Algorithm 1 from Yang et al. 2024 to iteratively minimize
the energy functional E(u, c1, c2) defined in Equation 2 of the paper.

The refinement is restricted to a padded ROI bounding box around the initial
mask to reduce computation; the result is composited back into a full-size
output mask.

Supports two initialisation modes
----------------------------------
Soft masking : pass the raw U-Net probability map as *u0_np*
Hard masking : binarise the probability map externally before calling this
               function (e.g. u0_np = (prob > threshold).astype(float))
"""

from typing import Optional, Tuple

import numpy as np
import torch

from oacm.operators import (
    beta_weight,
    conv2d_same,
    div2_operator,
    hessian_components,
    make_gaussian_kernel,
    psi_gradient,
    soft_threshold_2d,
    update_c1_c2,
)


# ── Helper: ROI bounding box ──────────────────────────────────────────────────

def get_roi_bbox(
    mask_np: np.ndarray,
    pad: int = 24,
) -> Optional[Tuple[int, int, int, int]]:
    """Return (y0, y1, x0, x1) bounding box of *mask_np* with *pad* pixels,
    or ``None`` if the mask is empty.
    """
    rows = np.any(mask_np, axis=1)
    cols = np.any(mask_np, axis=0)
    if not rows.any():
        return None
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    H, W = mask_np.shape
    y0 = max(0, rmin - pad)
    y1 = min(H, rmax + pad + 1)
    x0 = max(0, cmin - pad)
    x1 = min(W, cmax + pad + 1)
    return y0, y1, x0, x1


# ── Main refinement function ──────────────────────────────────────────────────

def oacm_refine(
    I_np: np.ndarray,
    u0_np: np.ndarray,
    lam: float,
    theta: float,
    tau: float,
    l_max: int,
    k_max: int,
    eps: float,
    roi_pad: int = 24,
    device: torch.device = torch.device("cpu"),
) -> np.ndarray:
    """Refine an initial segmentation mask using the OACM (Algorithm 1).

    Parameters
    ----------
    I_np    : normalised CT slice  (H, W)  float32, values in [0, 1]
    u0_np   : initial mask          (H, W)  float32 (soft or hard)
    lam     : data-fidelity balance  λ
    theta   : ADMM penalty parameter θ
    tau     : Gaussian kernel variance τ
    l_max   : maximum outer iterations L_max
    k_max   : inner ADMM iterations per outer step K_max
    eps     : convergence tolerance
    roi_pad : padding around the ROI bounding box (pixels)
    device  : torch device for GPU-accelerated operations

    Returns
    -------
    np.ndarray of shape (H, W), float32, values in {0, 1}
    """
    # Trivial early-exit: empty mask
    if u0_np.sum() == 0:
        return u0_np.astype(np.float32)

    bbox = get_roi_bbox(u0_np > 0.5, pad=roi_pad)
    if bbox is None:
        return u0_np.astype(np.float32)
    y0, y1, x0, x1 = bbox

    # Crop to ROI
    I_roi  = I_np[y0:y1, x0:x1]
    u0_roi = u0_np[y0:y1, x0:x1]

    with torch.no_grad():
        I   = torch.from_numpy(I_roi.astype(np.float32)).to(device)
        u   = torch.from_numpy(u0_roi.astype(np.float32)).to(device)
        H, W = I.shape

        # Binarise initial mask
        u = (u > 0.5).float()

        # Gaussian kernel size: larger τ → wider kernel
        k_size = max(int(6 * np.sqrt(tau * 1000)) | 1, 3) if tau < 0.1 else 15
        G_tau  = make_gaussian_kernel(tau, size=k_size, device=device)
        beta   = beta_weight(I)   # (H, W) edge-adaptive weight

        c1, c2 = update_c1_c2(u, I)

        # Initialise ADMM variables
        p = torch.zeros(4, H, W, dtype=torch.float32, device=device)
        q = torch.zeros(4, H, W, dtype=torch.float32, device=device)

        # ── Outer loop (L_max iterations) ────────────────────────────────────
        for _ in range(l_max):
            u_prev_outer = u.clone()
            u_bar        = u.clone()

            # ── Inner ADMM loop (K_max steps) ──────────────────────────────
            for _ in range(k_max):
                # Step 1: update ū  (Eq. 30 — threshold sign of ψ)
                psi   = psi_gradient(u_bar, p, q, I, c1, c2,
                                     G_tau, lam, tau, theta)
                u_bar = (psi <= 0).float()

                # Step 2: update p  (Eq. 48 — soft-thresholding)
                uxx, uxy, uyx, uyy = hessian_components(u_bar)
                nabla2_u = torch.stack([uxx, uxy, uyx, uyy], dim=0)
                p = soft_threshold_2d(
                    nabla2_u - q / (theta + 1e-12),
                    beta.unsqueeze(0) / (theta + 1e-12),
                )

                # Step 3: update q  (Eq. 49 — gradient ascent)
                uxx2, uxy2, uyx2, uyy2 = hessian_components(u_bar)
                nabla2_u_new = torch.stack([uxx2, uxy2, uyx2, uyy2], dim=0)
                q = q + theta * (p - nabla2_u_new)

            u = u_bar
            c1, c2 = update_c1_c2(u, I)

            # Convergence check
            diff = (u - u_prev_outer).abs().sum()
            if diff <= eps * (u_prev_outer.abs().sum() + 1e-12):
                break

        roi_result = u.cpu().numpy().astype(np.float32)

    # Composite ROI result back into full-size output
    u_out = np.zeros_like(I_np, dtype=np.float32)
    u_out[y0:y1, x0:x1] = roi_result
    return u_out
