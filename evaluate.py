"""
evaluate.py
-----------
End-to-end evaluation script for the two-stage lung nodule segmentation pipeline.

Stage 1 — 2.5-D U-Net inference
    Loads trained weights, runs the network over the validation set, and
    computes per-sample segmentation metrics.

Stage 2A — OACM refinement (soft masking)
    Passes the raw U-Net probability map as the initial contour to the OACM.

Stage 2B — OACM refinement (hard masking)
    Binarises the probability map at ``cfg.cnn_threshold`` before OACM.

Outputs (saved to ``cfg.out_dir``):
    bootstrap_distributions.png
    zoomed_comparison_contours.png
    per_patient_dice.png

Usage
-----
    python evaluate.py
"""

import warnings
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import label as nd_label
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from config import Config
from data.dataset import LIDCDataset, build_index
from models.unet import UNet2p5D
from oacm.refine import oacm_refine
from utils.checkpoint import load_checkpoint
from utils.metrics import aggregate_metrics, compute_metrics, print_metrics_table
from utils.visualization import (
    plot_bootstrap_distributions,
    plot_per_patient_dice,
    visualise_zoomed_comparison,
    visualize_predictions,
)

warnings.filterwarnings("ignore")


# ── Post-processing ───────────────────────────────────────────────────────────

def clean_mask(mask: np.ndarray, min_size: int = 20) -> np.ndarray:
    """Remove connected components smaller than *min_size* pixels."""
    labeled, n = nd_label(mask.astype(int))
    cleaned = np.zeros_like(mask, dtype=np.float32)
    for i in range(1, n + 1):
        component = labeled == i
        if component.sum() >= min_size:
            cleaned[component] = 1.0
    return cleaned


# ── Stage 1 inference ─────────────────────────────────────────────────────────

@torch.no_grad()
def run_cnn_inference(model, loader, device, context_slices: int,
                      threshold: float = 0.3):
    """Run U-Net over *loader* and return aligned result lists.

    Returns
    -------
    cnn_probs   : list of (H, W) float32 probability maps
    cnn_preds   : list of (H, W) bool binary masks (cleaned)
    ct_centers  : list of (H, W) float32 centre-slice CT images
    gt_masks    : list of (H, W) bool ground-truth masks
    """
    model.eval()
    cnn_probs, cnn_preds, ct_centers, gt_masks = [], [], [], []

    for x, y in tqdm(loader, desc="Stage 1 — U-Net inference", leave=True):
        x     = x.to(device)
        probs = torch.sigmoid(model(x)).squeeze(1).cpu().numpy()
        preds = probs >= threshold
        cts   = x[:, context_slices, :, :].cpu().numpy()
        gts   = y.squeeze(1).numpy().astype(bool)

        for b in range(len(probs)):
            clean_pred = clean_mask(preds[b], min_size=35)
            cnn_probs.append(probs[b].astype(np.float32))
            cnn_preds.append(clean_pred.astype(bool))
            ct_centers.append(cts[b].astype(np.float32))
            gt_masks.append(gts[b])

    return cnn_probs, cnn_preds, ct_centers, gt_masks


# ── Stage 2 refinement ────────────────────────────────────────────────────────

def run_oacm_refinement(ct_centers, u0_list, device, lam, theta, tau,
                         l_max, k_max, eps, desc="Stage 2 — OACM"):
    """Apply OACM to each sample in *u0_list* and return refined predictions."""
    preds   = []
    results = []
    for i in tqdm(range(len(u0_list)), desc=desc):
        u_refined = oacm_refine(
            ct_centers[i], u0_list[i],
            lam=lam, theta=theta, tau=tau,
            l_max=l_max, k_max=k_max, eps=eps,
            device=device,
        )
        preds.append(u_refined.astype(bool))
        results.append(u_refined)
    return preds, results


# ── Main evaluation ───────────────────────────────────────────────────────────

def evaluate(cfg: Config) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Dataset ────────────────────────────────────────────────────────────
    data_root = Path(cfg.data_root)
    val_dir   = data_root / cfg.val_split
    assert val_dir.exists(), f"Val dir not found: {val_dir}"

    val_items  = build_index(val_dir)
    val_ds     = LIDCDataset(val_items, wc=cfg.wc, ww=cfg.ww,
                              context=cfg.context_slices,
                              target_size=cfg.target_size, split="val")
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                            num_workers=cfg.num_workers, pin_memory=True)
    print(f"Val items: {len(val_items)}")

    # ── Load U-Net weights ─────────────────────────────────────────────────
    in_ch = 2 * cfg.context_slices + 1
    model = UNet2p5D(in_channels=in_ch, base=cfg.unet_base).to(device)

    loaded = False
    for wpath in [cfg.custom_weights_path, cfg.model_weights_path]:
        p = Path(wpath)
        if p.exists():
            try:
                load_checkpoint(p, model, map_location=device, strict=False)
                print(f"[Stage 1] Weights loaded from: {p}")
                loaded = True
                break
            except Exception as ex:
                print(f"[WARN] Could not load {p}: {ex}")

    if not loaded:
        print("[WARN] No weights found — using random model weights.")

    # ── Stage 1: U-Net inference ───────────────────────────────────────────
    cnn_probs, cnn_preds, ct_centers, gt_masks = run_cnn_inference(
        model, val_loader, device,
        context_slices=cfg.context_slices,
        threshold=cfg.cnn_threshold,
    )

    print("\nComputing Stage 1 metrics...")
    stage1_results = [
        compute_metrics(cnn_preds[i], gt_masks[i])
        for i in tqdm(range(len(cnn_preds)), desc="Stage 1 metrics")
    ]
    stage1_agg = aggregate_metrics(stage1_results)
    print("\n[Stage 1 — U-Net] Aggregate Metrics:")
    for k, (m, s) in stage1_agg.items():
        print(f"  {k:>15}: {m:.4f} ± {s:.4f}")

    # ── Stage 2A: OACM with soft probability map ───────────────────────────
    print(f"\n[Stage 2A] OACM (soft)  λ={cfg.lam}  θ={cfg.theta}  τ={cfg.tau}")
    cnn_probs_list = list(cnn_probs)   # already float32 probability maps
    oacm_preds_soft, oacm_raw_soft = run_oacm_refinement(
        ct_centers, cnn_probs_list, device,
        lam=cfg.lam, theta=cfg.theta, tau=cfg.tau,
        l_max=cfg.l_max, k_max=cfg.k_max, eps=cfg.eps,
        desc="Stage 2A — OACM (soft)",
    )
    stage2a_results = [
        compute_metrics(oacm_preds_soft[i], gt_masks[i])
        for i in range(len(oacm_preds_soft))
    ]
    stage2a_agg = aggregate_metrics(stage2a_results)

    # ── Stage 2B: OACM with hard binary mask ──────────────────────────────
    print(f"\n[Stage 2B] OACM (hard)  λ={cfg.lam2}  θ={cfg.theta2}  τ={cfg.tau2}")
    cnn_probs_hard = [(p > cfg.cnn_threshold).astype(float) for p in cnn_probs]
    oacm_preds_hard, _ = run_oacm_refinement(
        ct_centers, cnn_probs_hard, device,
        lam=cfg.lam2, theta=cfg.theta2, tau=cfg.tau2,
        l_max=cfg.l_max2, k_max=cfg.k_max2, eps=cfg.eps,
        desc="Stage 2B — OACM (hard)",
    )
    stage2b_results = [
        compute_metrics(oacm_preds_hard[i], gt_masks[i])
        for i in range(len(oacm_preds_hard))
    ]
    stage2b_agg = aggregate_metrics(stage2b_results)

    # ── Quantitative comparison table ─────────────────────────────────────
    print("\n" + "=" * 88)
    print("   QUANTITATIVE EVALUATION: Stage 1 (U-Net) vs Stage 2A vs Stage 2B")
    print("=" * 88)
    print(f"{'Metric':>15} | {'Stage1':>10} | {'Stage2A':>10} | {'Stage2B':>10}")
    print("-" * 60)
    for k in stage1_agg.keys():
        m1, _  = stage1_agg[k]
        m2a, _ = stage2a_agg[k]
        m2b, _ = stage2b_agg[k]
        print(f"{k:>15} | {m1:10.4f} | {m2a:10.4f} | {m2b:10.4f}")

    print("\nRelative improvement over Stage 1:")
    for k in ["dice", "jaccard", "kappa"]:
        m1, _  = stage1_agg[k]; m2a, _ = stage2a_agg[k]; m2b, _ = stage2b_agg[k]
        print(f"  {k:>12}:  Stage2A {100*(m2a-m1)/(m1+1e-9):+.2f}%   "
              f"Stage2B {100*(m2b-m1)/(m1+1e-9):+.2f}%")

    h1, _ = stage1_agg["hausdorff"]; h2a, _ = stage2a_agg["hausdorff"]
    h2b, _ = stage2b_agg["hausdorff"]
    print(f"  {'hausdorff':>12}:  Stage2A {100*(h2a-h1)/(h1+1e-9):+.2f}% (↓ better)  "
          f"Stage2B {100*(h2b-h1)/(h1+1e-9):+.2f}% (↓ better)")

    # ── Visualisations ─────────────────────────────────────────────────────
    # Use Stage 2A predictions for visualisation
    plot_bootstrap_distributions(
        stage1_results, stage2a_results,
        save_path=str(out_dir / "bootstrap_distributions.png"),
    )

    visualise_zoomed_comparison(
        ct_centers, cnn_preds, oacm_preds_soft, gt_masks,
        n=cfg.n_vis_samples, pad=20,
        save_path=str(out_dir / "zoomed_comparison_contours.png"),
    )

    plot_per_patient_dice(
        val_items, stage1_results, stage2a_results,
        save_path=str(out_dir / "per_patient_dice.png"),
    )

    print(f"\nAll outputs saved to: {out_dir}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    cfg = Config()
    evaluate(cfg)
