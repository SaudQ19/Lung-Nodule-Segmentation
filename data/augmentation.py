"""
data/augmentation.py
--------------------
Spatial and photometric augmentations applied to (image, mask) pairs during
training.  All operations are performed on raw NumPy arrays so they can run
on CPU DataLoader workers without GPU overhead.

Expected input shapes
---------------------
image : (C, H, W)  float32  — C context slices, values in [0, 1]
mask  : (1, H, W)  float32  — binary segmentation target
"""

import random
import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import rotate as nd_rotate


def augment(image: np.ndarray, mask: np.ndarray):
    """Apply a random chain of augmentations to an image–mask pair.

    Parameters
    ----------
    image : np.ndarray, shape (C, H, W), float32
    mask  : np.ndarray, shape (1, H, W), float32

    Returns
    -------
    image, mask : augmented copies (same shapes and dtype)
    """
    # ── Horizontal flip ──────────────────────────────────────────────────────
    if random.random() < 0.5:
        image = np.flip(image, axis=2).copy()
        mask  = np.flip(mask,  axis=2).copy()

    # ── Vertical flip ────────────────────────────────────────────────────────
    if random.random() < 0.5:
        image = np.flip(image, axis=1).copy()
        mask  = np.flip(mask,  axis=1).copy()

    # ── 90° rotation multiples ───────────────────────────────────────────────
    if random.random() < 0.5:
        k = random.randint(1, 3)
        image = np.rot90(image, k, axes=(1, 2)).copy()
        mask  = np.rot90(mask,  k, axes=(1, 2)).copy()

    # ── Small random rotation (±15°) ─────────────────────────────────────────
    if random.random() < 0.3:
        angle = np.random.uniform(-15, 15)
        image = np.stack(
            [nd_rotate(image[c], angle, axes=(0, 1), reshape=False, order=1)
             for c in range(image.shape[0])],
            axis=0,
        )
        mask = nd_rotate(mask[0], angle, axes=(0, 1), reshape=False, order=0)[None]

    # ── Intensity shift & scale ──────────────────────────────────────────────
    if random.random() < 0.4:
        shift = np.random.uniform(-0.1, 0.1)
        scale = np.random.uniform(0.85, 1.15)
        image = np.clip(image * scale + shift, 0.0, 1.0)

    # ── Additive Gaussian noise ──────────────────────────────────────────────
    if random.random() < 0.3:
        noise = np.random.normal(0, 0.02, image.shape).astype(np.float32)
        image = np.clip(image + noise, 0.0, 1.0)

    # ── Random zoom-crop (spatial size preserved via bilinear resize) ─────────
    if random.random() < 0.3:
        _, H, W = image.shape
        scale_f = np.random.uniform(0.85, 1.0)
        new_h, new_w = int(H * scale_f), int(W * scale_f)
        y0 = random.randint(0, H - new_h)
        x0 = random.randint(0, W - new_w)
        crop_img  = image[:, y0:y0 + new_h, x0:x0 + new_w]
        crop_mask = mask[:,  y0:y0 + new_h, x0:x0 + new_w]
        crop_img_t  = torch.from_numpy(crop_img).unsqueeze(0)
        crop_mask_t = torch.from_numpy(crop_mask).unsqueeze(0)
        image = (
            F.interpolate(crop_img_t, size=(H, W), mode="bilinear", align_corners=False)
            .squeeze(0)
            .numpy()
        )
        mask = (
            F.interpolate(crop_mask_t, size=(H, W), mode="nearest")
            .squeeze(0)
            .numpy()
        )

    return image, mask
