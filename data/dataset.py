"""
data/dataset.py
---------------
Dataset utilities and PyTorch Dataset implementation for the LIDC-IDRI
lung nodule segmentation task.

Expected directory layout
--------------------------
<data_root>/
  train/
    <patient_id>/
      <slice>_ct.npy
      <slice>_mask.npy
      ...
  test/
    <patient_id>/
      ...

Each `*_ct.npy` file is a single 2-D CT slice (H×W, float or int).
The corresponding `*_mask.npy` is its binary segmentation mask.
"""

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from data.augmentation import augment


# ── Normalisation ─────────────────────────────────────────────────────────────

def wl_normalize(image: np.ndarray,
                 wc: float = -450.0,
                 ww: float = 1000.0) -> np.ndarray:
    """Min-max normalise a CT slice to [0, 1].

    The window-centre / window-width parameters are kept for API
    compatibility; the current implementation uses the per-slice
    min/max (robust to varying HU ranges across datasets).
    """
    x = image.astype(np.float32)
    lo, hi = x.min(), x.max()
    if hi - lo < 1e-6:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


# ── Index builders ────────────────────────────────────────────────────────────

def sorted_ct_mask_pairs(
    patient_dir: Path,
) -> Tuple[List[Path], List[Path]]:
    """Return parallel lists of CT and mask file paths for one patient,
    keeping only slices for which both files exist.
    """
    ct_files = sorted(patient_dir.glob("*_ct.npy"))
    kept_ct, mask_files = [], []
    for ct in ct_files:
        mk = ct.with_name(ct.name.replace("_ct.npy", "_mask.npy"))
        if mk.exists():
            kept_ct.append(ct)
            mask_files.append(mk)
    return kept_ct, mask_files


def build_index(split_dir: Path) -> List[Dict]:
    """Build a flat list of (patient, ct_files, mask_files, center_idx) dicts.

    One entry per slice per patient; the center_idx points to the slice
    that will be the segmentation target — its neighbours supply context.
    """
    items: List[Dict] = []
    for pdir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
        ct_files, mask_files = sorted_ct_mask_pairs(pdir)
        if not ct_files:
            continue
        for i in range(len(ct_files)):
            items.append(
                {
                    "patient":    pdir.name,
                    "ct_files":   ct_files,
                    "mask_files": mask_files,
                    "center_idx": i,
                }
            )
    return items


# ── Tensor helpers ────────────────────────────────────────────────────────────

def resize_tensor_2d(
    x: torch.Tensor,
    target_hw: Tuple[int, int],
    mode: str = "bilinear",
) -> torch.Tensor:
    """Resize a (1, C, H, W) tensor to *target_hw* using *mode* interpolation."""
    align = False if mode in ("bilinear", "bicubic") else None
    return F.interpolate(x, size=target_hw, mode=mode, align_corners=align)


# ── Dataset ───────────────────────────────────────────────────────────────────

class LIDCDataset(Dataset):
    """2.5-D slice dataset for LIDC-IDRI lung nodule segmentation.

    Each sample is a stack of `(2*context + 1)` adjacent CT slices centred
    on the target slice, resized to `target_size`.  During training,
    spatial and photometric augmentations are applied.

    Parameters
    ----------
    items       : output of :func:`build_index`
    wc, ww      : CT window centre and width (passed to :func:`wl_normalize`)
    context     : number of neighbouring slices on each side
    target_size : spatial size (H, W) after resizing
    split       : ``'train'`` enables augmentation; anything else disables it
    """

    def __init__(
        self,
        items: List[Dict],
        wc: float = -450.0,
        ww: float = 1000.0,
        context: int = 2,
        target_size: Tuple[int, int] = (256, 256),
        split: str = "train",
    ) -> None:
        self.items       = items
        self.wc, self.ww = wc, ww
        self.context     = context
        self.target_size = target_size
        self.split       = split

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        item      = self.items[idx]
        c         = item["center_idx"]
        ct_files  = item["ct_files"]
        mask_files = item["mask_files"]

        # Stack (2*context + 1) neighbouring slices as channels
        channels = []
        for delta in range(-self.context, self.context + 1):
            j  = int(np.clip(c + delta, 0, len(ct_files) - 1))
            ct = np.load(ct_files[j]).astype(np.float32)
            channels.append(wl_normalize(ct, self.wc, self.ww))
        x = np.stack(channels, axis=0)               # (2C+1, H, W)

        m = np.load(mask_files[c]).astype(np.float32)
        y = (m > 0).astype(np.float32)[None, ...]    # (1, H, W)

        if self.split == "train":
            x, y = augment(x, y)

        x_t = torch.from_numpy(x)
        y_t = torch.from_numpy(y)

        if self.target_size is not None:
            x_t = resize_tensor_2d(x_t.unsqueeze(0), self.target_size).squeeze(0)
            y_t = resize_tensor_2d(
                y_t.unsqueeze(0), self.target_size, mode="nearest"
            ).squeeze(0)

        return x_t, y_t
