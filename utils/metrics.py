"""
utils/metrics.py
----------------
Quantitative evaluation metrics for binary segmentation.

Metrics
-------
Dice coefficient        : 2|P∩G| / (|P|+|G|)                    ↑ better
Jaccard / IoU           : |P∩G| / |P∪G|                          ↑ better
Sensitivity (recall)    : TP / (TP + FN)                          ↑ better
Specificity             : TN / (TN + FP)                          ↑ better
Cohen's κ               : (p_o - p_e) / (1 - p_e)                ↑ better
Hausdorff distance      : max(h(P,G), h(G,P))                     ↓ better

All metrics match the evaluation protocol in Yang et al. 2024, Table 1.
"""

from typing import Dict, List

import numpy as np
from scipy.ndimage import distance_transform_edt


def compute_metrics(
    pred: np.ndarray,
    gt: np.ndarray,
    eps: float = 1e-7,
) -> Dict[str, float]:
    """Compute all segmentation metrics for a single prediction–GT pair.

    Parameters
    ----------
    pred : predicted binary mask  (H, W)  — any boolean/numeric type
    gt   : ground-truth mask      (H, W)
    eps  : small constant for numerical stability

    Returns
    -------
    dict with keys: dice, jaccard, sensitivity, specificity, kappa, hausdorff
    """
    pred = pred.astype(bool).ravel()
    gt   = gt.astype(bool).ravel()

    TP = int(( pred &  gt).sum())
    FP = int(( pred & ~gt).sum())
    TN = int((~pred & ~gt).sum())
    FN = int((~pred &  gt).sum())

    dice        = (2 * TP + eps)  / (2 * TP + FP + FN + eps)
    jaccard     = (    TP + eps)  / (    TP + FP + FN + eps)
    sensitivity = (    TP + eps)  / (    TP + FN       + eps)
    specificity = (    TN + eps)  / (    TN + FP       + eps)

    N   = len(pred)
    p_o = (TP + TN) / N
    p_e = (
        ((TP + FP) / N) * ((TP + FN) / N)
        + ((TN + FN) / N) * ((TN + FP) / N)
    )
    kappa = (p_o - p_e) / (1 - p_e + eps)

    # Hausdorff distance computed on 2-D masks
    side = int(round(np.sqrt(len(pred))))
    hd   = _hausdorff(
        pred.reshape(side, side).astype(np.uint8),
        gt.reshape(side,   side).astype(np.uint8),
    )

    return dict(
        dice=dice,
        jaccard=jaccard,
        sensitivity=sensitivity,
        specificity=specificity,
        kappa=kappa,
        hausdorff=hd,
    )


def _hausdorff(A: np.ndarray, B: np.ndarray) -> float:
    """Symmetric Hausdorff distance between two binary masks."""
    if A.sum() == 0 and B.sum() == 0:
        return 0.0
    if A.sum() == 0 or B.sum() == 0:
        return float(max(A.shape))
    h_AB = distance_transform_edt(~B.astype(bool))[A.astype(bool)].max()
    h_BA = distance_transform_edt(~A.astype(bool))[B.astype(bool)].max()
    return float(max(h_AB, h_BA))


def aggregate_metrics(results: List[Dict[str, float]]) -> Dict[str, tuple]:
    """Aggregate a list of per-sample metric dicts into (mean, std) pairs."""
    keys = results[0].keys()
    return {
        k: (
            float(np.mean([r[k] for r in results])),
            float(np.std( [r[k] for r in results])),
        )
        for k in keys
    }


def print_metrics_table(
    agg_s1: Dict[str, tuple],
    agg_s2: Dict[str, tuple],
    label_s1: str = "Stage 1 (U-Net)",
    label_s2: str = "Stage 2 (OACM)",
) -> None:
    """Pretty-print a two-column comparison table."""
    metrics = ["dice", "jaccard", "sensitivity", "specificity", "kappa", "hausdorff"]
    header  = f'{"Metric":>15}  {label_s1:>22}  {label_s2:>22}  {"Δ":>8}'
    sep     = "─" * len(header)
    print(sep)
    print(header)
    print(sep)
    for m in metrics:
        m1, s1v = agg_s1[m]
        m2, s2v = agg_s2[m]
        delta   = m2 - m1
        print(
            f"{m:>15}  {m1:>8.4f} ± {s1v:.4f}  "
            f"{m2:>8.4f} ± {s2v:.4f}  {delta:>+.4f}"
        )
    print(sep)
