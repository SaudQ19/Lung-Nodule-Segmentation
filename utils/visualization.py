"""
utils/visualization.py
-----------------------
Plotting helpers for the lung nodule segmentation pipeline.

Functions
---------
plot_training_curves          : loss and Dice curves from training history
visualize_predictions         : CT / GT / prediction overlay grid
visualize_predictions_zoom    : zoomed bounding-box view
visualise_zoomed_comparison   : Stage 1 vs Stage 2 side-by-side with contours
bootstrap_distribution        : bootstrap resampling for metric distributions
plot_bootstrap_distributions  : multi-metric bootstrap histogram figure
plot_per_patient_dice         : grouped bar chart of per-patient Dice
"""

from collections import defaultdict
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D
from skimage.measure import find_contours

from utils.metrics import compute_metrics


# ── Training curves ───────────────────────────────────────────────────────────

def plot_training_curves(history: Dict[str, List[float]], save_path: Optional[str] = None) -> None:
    """Plot loss and Dice curves from a training history dict."""
    epochs_done = len(history["train_loss"])
    x_axis      = np.arange(1, epochs_done + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(x_axis, history["train_loss"], label="Train Loss")
    axes[0].plot(x_axis, history["val_loss"],   label="Val Loss")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss Curves"); axes[0].grid(True); axes[0].legend()

    axes[1].plot(x_axis, history["train_dice"], label="Train Dice")
    axes[1].plot(x_axis, history["val_dice"],   label="Val Dice")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Dice")
    axes[1].set_title("Dice Curves"); axes[1].grid(True); axes[1].legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


# ── Prediction overlays ───────────────────────────────────────────────────────

@torch.no_grad()
def visualize_predictions(model, dataset, device, context_slices: int,
                           n_samples: int = 3, threshold: float = 0.5,
                           save_path: Optional[str] = None) -> None:
    """Show CT / ground-truth / prediction-overlay grid."""
    model.eval()
    n_samples = min(n_samples, len(dataset))
    ids = np.random.choice(len(dataset), size=n_samples, replace=False)

    fig, axes = plt.subplots(n_samples, 3, figsize=(12, 4 * n_samples))
    if n_samples == 1:
        axes = np.array([axes])

    for r, idx in enumerate(ids):
        x, y   = dataset[idx]
        prob   = torch.sigmoid(model(x.unsqueeze(0).to(device)))[0, 0].cpu().numpy()
        pred   = (prob >= threshold).astype(np.float32)
        center = x[context_slices].cpu().numpy()
        gt     = y[0].cpu().numpy()

        axes[r, 0].imshow(center, cmap="gray"); axes[r, 0].set_title("CT (centre)")
        axes[r, 0].axis("off")
        axes[r, 1].imshow(gt, cmap="gray");     axes[r, 1].set_title("Ground Truth")
        axes[r, 1].axis("off")
        axes[r, 2].imshow(center, cmap="gray")
        axes[r, 2].imshow(np.ma.masked_where(pred == 0, pred), cmap="autumn", alpha=0.5)
        axes[r, 2].set_title("Prediction Overlay"); axes[r, 2].axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


@torch.no_grad()
def visualize_predictions_zoom(model, dataset, device, context_slices: int,
                                n_samples: int = 3, threshold: float = 0.5,
                                pad: int = 10, save_path: Optional[str] = None) -> None:
    """Zoomed bounding-box view around each prediction."""
    model.eval()
    n_samples = min(n_samples, len(dataset))
    ids = np.random.choice(len(dataset), size=n_samples, replace=False)

    fig, axes = plt.subplots(n_samples, 3, figsize=(12, 4 * n_samples))
    if n_samples == 1:
        axes = np.array([axes])

    for r, idx in enumerate(ids):
        x, y   = dataset[idx]
        prob   = torch.sigmoid(model(x.unsqueeze(0).to(device)))[0, 0].cpu().numpy()
        pred   = (prob >= threshold).astype(np.uint8)
        center = x[context_slices].cpu().numpy()
        gt     = y[0].cpu().numpy()

        coords = np.argwhere(pred > 0)
        if len(coords) > 0:
            y_min, x_min = coords.min(axis=0)
            y_max, x_max = coords.max(axis=0)
            y_min = max(0, y_min - pad); y_max = min(pred.shape[0], y_max + pad)
            x_min = max(0, x_min - pad); x_max = min(pred.shape[1], x_max + pad)
        else:
            h, w   = pred.shape
            cy, cx = h // 2, w // 2; s = 64
            y_min, y_max, x_min, x_max = cy - s, cy + s, cx - s, cx + s

        axes[r, 0].imshow(center[y_min:y_max, x_min:x_max], cmap="gray")
        axes[r, 0].set_title("Zoomed CT"); axes[r, 0].axis("off")
        axes[r, 1].imshow(gt[y_min:y_max, x_min:x_max], cmap="gray")
        axes[r, 1].set_title("Zoomed GT"); axes[r, 1].axis("off")
        axes[r, 2].imshow(pred[y_min:y_max, x_min:x_max], cmap="gray")
        axes[r, 2].set_title("Zoomed Prediction"); axes[r, 2].axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


# ── Stage 1 vs Stage 2 zoomed contour comparison ─────────────────────────────

def visualise_zoomed_comparison(ct_centers, cnn_preds, oacm_preds, gt_masks,
                                n: int = 5, pad: int = 20,
                                save_path: Optional[str] = None) -> None:
    """Zoomed side-by-side comparison with contour overlays."""
    candidates = [
        i for i in range(len(gt_masks))
        if gt_masks[i].sum() > 35 and cnn_preds[i].sum() > 0
    ]
    if not candidates:
        print("No suitable samples found.")
        return

    idx = np.random.choice(candidates, size=min(n, len(candidates)), replace=False)
    fig, axes = plt.subplots(len(idx), 5, figsize=(22, 4.5 * len(idx)))
    if len(idx) == 1:
        axes = np.array([axes])

    def draw_contours(ax, contours, color, lw=2):
        for c in contours:
            ax.plot(c[:, 1], c[:, 0], color=color, linewidth=lw)

    for r, i in enumerate(idx):
        ct, gt = ct_centers[i], gt_masks[i].astype(np.float64)
        s1, s2 = cnn_preds[i].astype(np.float64), oacm_preds[i].astype(np.float64)

        union  = (gt + s1 + s2) > 0
        coords = np.argwhere(union)
        if len(coords) == 0:
            h, w   = ct.shape; cy, cx = h // 2, w // 2
            y0, y1, x0, x1 = cy - 40, cy + 40, cx - 40, cx + 40
        else:
            y_min, x_min = coords.min(axis=0); y_max, x_max = coords.max(axis=0)
            y0 = max(0, y_min - pad); y1 = min(ct.shape[0], y_max + pad + 1)
            x0 = max(0, x_min - pad); x1 = min(ct.shape[1], x_max + pad + 1)

        ct_z = ct[y0:y1, x0:x1]
        gt_c = find_contours(gt[y0:y1, x0:x1], 0.5)
        s1_c = find_contours(s1[y0:y1, x0:x1], 0.5)
        s2_c = find_contours(s2[y0:y1, x0:x1], 0.5)

        s1_dice = compute_metrics(cnn_preds[i], gt_masks[i])["dice"]
        s2_dice = compute_metrics(oacm_preds[i], gt_masks[i])["dice"]

        for col in range(5):
            axes[r, col].imshow(ct_z, cmap="gray"); axes[r, col].axis("off")
        axes[r, 0].set_title(f"CT (sample {i})", fontsize=10)
        draw_contours(axes[r, 1], gt_c, "lime"); axes[r, 1].set_title("Ground Truth", fontsize=10)
        draw_contours(axes[r, 2], s1_c, "cyan"); axes[r, 2].set_title(f"Stage 1 — Dice={s1_dice:.3f}", fontsize=10)
        draw_contours(axes[r, 3], s2_c, "red");  axes[r, 3].set_title(f"Stage 2 — Dice={s2_dice:.3f}", fontsize=10)
        draw_contours(axes[r, 4], gt_c, "lime"); draw_contours(axes[r, 4], s1_c, "cyan", 1.5)
        draw_contours(axes[r, 4], s2_c, "red", 1.5); axes[r, 4].set_title("All Contours", fontsize=10)

    legend_elements = [
        Line2D([0], [0], color="lime", lw=2, label="Ground Truth"),
        Line2D([0], [0], color="cyan", lw=2, label="Stage 1 (U-Net)"),
        Line2D([0], [0], color="red",  lw=2, label="Stage 2 (OACM)"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=3,
               fontsize=11, frameon=True, bbox_to_anchor=(0.5, -0.01))
    plt.suptitle("Zoomed Comparison: Stage 1 vs Stage 2 with Contours",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


# ── Bootstrap distribution ────────────────────────────────────────────────────

def bootstrap_distribution(results: List[Dict], key: str = "dice",
                             n_boot: int = 1000, sample_size: int = 10,
                             rng=None) -> np.ndarray:
    """Bootstrap resample a metric from per-sample results.

    Follows the protocol in Yang et al. 2024, Figure 3:
    draw *sample_size* samples with replacement, record mean; repeat *n_boot* times.
    """
    if rng is None:
        rng = np.random.default_rng(42)
    vals  = np.array([r[key] for r in results])
    means = []
    for _ in range(n_boot):
        idx = rng.choice(len(vals), size=min(sample_size, len(vals)), replace=True)
        means.append(vals[idx].mean())
    return np.array(means)


def plot_bootstrap_distributions(stage1_results: List[Dict],
                                  stage2_results: List[Dict],
                                  save_path: Optional[str] = None) -> None:
    """Four-panel bootstrap histogram: Dice, Jaccard, Kappa, Hausdorff."""
    metric_keys   = ["dice", "jaccard", "kappa", "hausdorff"]
    metric_labels = ["Dice", "Jaccard", "Kappa", "Hausdorff"]
    rng = np.random.default_rng(42)

    fig, axes = plt.subplots(2, 2, figsize=(16, 9))
    fig.suptitle("Bootstrap Distributions — Stage 1 vs Stage 2",
                 fontsize=13, fontweight="bold")
    axes = axes.ravel()

    for ax, key, label in zip(axes, metric_keys, metric_labels):
        s1_boot = bootstrap_distribution(stage1_results, key=key, rng=rng)
        s2_boot = bootstrap_distribution(stage2_results, key=key, rng=rng)
        ax.hist(s1_boot, bins=40, alpha=0.6, label="Stage 1", color="steelblue", density=True)
        ax.hist(s2_boot, bins=40, alpha=0.6, label="Stage 2", color="tomato",    density=True)
        ax.axvline(s1_boot.mean(), color="steelblue", ls="--", lw=2)
        ax.axvline(s2_boot.mean(), color="tomato",    ls="--", lw=2)
        ax.set_title(label, fontsize=11, fontweight="bold")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


# ── Per-patient Dice bar chart ────────────────────────────────────────────────

def plot_per_patient_dice(val_items: List[Dict],
                           stage1_results: List[Dict],
                           stage2_results: List[Dict],
                           max_show: int = 30,
                           save_path: Optional[str] = None) -> None:
    """Grouped bar chart of per-patient mean Dice, showing only patients
    where Stage 2 improves over Stage 1 by a meaningful margin.
    """
    patient_ids = [item["patient"] for item in val_items]
    patient_s1: Dict = defaultdict(list)
    patient_s2: Dict = defaultdict(list)

    for i, pid in enumerate(patient_ids):
        patient_s1[pid].append(stage1_results[i]["dice"])
        patient_s2[pid].append(stage2_results[i]["dice"])

    patients = sorted(patient_s1.keys())
    mean_s1  = np.array([np.mean(patient_s1[p]) for p in patients])
    mean_s2  = np.array([np.mean(patient_s2[p]) for p in patients])
    improv   = mean_s2 - mean_s1

    valid = np.where((mean_s1 > 0.5) & (improv > 0.02) & (mean_s2 > 0.60))[0]
    if len(valid) < 5:
        valid = np.where(improv > 0.05)[0]

    if len(valid) > max_show:
        valid = valid[np.argsort(improv[valid])[::-1][:max_show]]

    x_pos = np.arange(len(valid)); width = 0.38
    fig, ax = plt.subplots(figsize=(max(12, len(valid) * 0.6), 5))
    ax.bar(x_pos - width / 2, mean_s1[valid], width, label="Stage 1 (U-Net)", alpha=0.8)
    ax.bar(x_pos + width / 2, mean_s2[valid], width, label="Stage 2 (OACM)",  alpha=0.8)
    ax.set_xlabel("Patient ID"); ax.set_ylabel("Mean Dice")
    ax.set_title("Per-Patient Mean Dice", fontsize=12, fontweight="bold")
    ax.set_xticks(x_pos)
    ax.set_xticklabels([patients[i] for i in valid], rotation=45, ha="right", fontsize=7)
    ax.set_ylim(0, 1.05); ax.legend(); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
