# Lung Nodule Segmentation — 2.5D U-Net + OACM

A two-stage segmentation pipeline for lung nodules in CT images, combining a
2.5-D U-Net (Stage 1) with an Optimized Active Contour Model (Stage 2, OACM).

> **Reference:** Yang et al., *"Few-shot segmentation framework for lung nodules
> via an optimized active contour model"*, Medical Physics, 2024.
> DOI: [10.1002/mp.16933](https://doi.org/10.1002/mp.16933)


---

## Overview

| Stage | Module | Role |
|-------|--------|------|
| 1 | 2.5-D U-Net | Coarse binary segmentation from multi-slice CT context |
| 2 | OACM | Energy-minimisation refinement driven by Stage 1 output |

The OACM energy functional:

$$E(u, c_1, c_2) = \lambda \int_\Omega \left[(I-c_1)^2 u + (I-c_2)^2(1-u)\right]dx + \sqrt{\frac{\pi}{\tau}} \int_\Omega u\, G_\tau*(1-u)\,dx + \int_\Omega \beta(I)|\nabla^2 u|\,dx$$

It is solved by the ADMM algorithm (Algorithm 1). Stage 2 supports two
initialisation modes:

- **Soft masking** — the raw U-Net probability map is passed as $u^0$
- **Hard masking** — the probability map is binarised at `cfg.cnn_threshold` before OACM

---

## Repository Structure

```
lung_nodule_seg/
│
├── config.py                   # All hyperparameters in one place
├── train.py                    # Stage 1 training entry point
├── evaluate.py                 # Full pipeline evaluation entry point
├── requirements.txt
│
├── data/
│   ├── __init__.py
│   ├── dataset.py              # LIDCDataset + index builder + normalisation
│   └── augmentation.py         # Spatial & photometric augmentations
│
├── models/
│   ├── __init__.py
│   ├── unet.py                 # UNet2p5D architecture
│   └── losses.py               # BCEDiceLoss + training-time Dice metric
│
├── oacm/
│   ├── __init__.py
│   ├── operators.py            # Low-level OACM tensor operators (GPU)
│   └── refine.py               # oacm_refine() — Algorithm 1 loop
│
└── utils/
    ├── __init__.py
    ├── metrics.py              # Dice, Jaccard, Sensitivity, κ, Hausdorff
    ├── checkpoint.py           # save_checkpoint / load_checkpoint
    └── visualization.py        # Training curves, contour overlays, bootstrap plots
```

---

## Dataset

The code is built and tested on [LIDC-IDRI](https://wiki.cancerimagingarchive.net/display/Public/LIDC-IDRI).

### Expected Directory Layout

```
data/lidc_dataset/
  train/
    <patient_id>/
      0000_ct.npy
      0000_mask.npy
      0001_ct.npy
      0001_mask.npy
      ...
  test/
    <patient_id>/
      ...
```

Each `*_ct.npy` is a single 2-D CT slice saved as a NumPy array (H × W, any
numeric dtype). The matching `*_mask.npy` is its binary segmentation mask.


### Training Split (Paper Protocol)

| Dataset | Train | Test |
|---------|-------|------|
| LIDC-IDRI | 80 patients | 20 patients |
---
*Owing to the few-shot learning paradigm, the study is restricted to a maximum of 100 patients.

## Installation

```bash
git clone https://github.com/<your-username>/lung-nodule-segmentation.git
cd lung-nodule-segmentation
pip install -r requirements.txt
```

Python ≥ 3.10 and PyTorch ≥ 2.0 are required. CUDA is strongly recommended
for both training and OACM refinement.

---

## Quick Start

### 1 — Configure paths

Edit `config.py`:

```python
data_root            = "data/lidc_dataset"   # dataset root
out_dir              = "checkpoints"          # where to save model weights
custom_weights_path  = "weights/model_best.pt"  # optional pretrained weights
```

### 2 — Train (Stage 1)

```bash
python train.py
```

The best-Dice checkpoint is saved to `checkpoints/model_best.pt`.

### 3 — Evaluate (Stage 1 + Stage 2)

```bash
python evaluate.py
```

Printed output includes a metric table comparing Stage 1 and both Stage 2
variants. Figures are saved to `checkpoints/`.

---

## Configuration Reference (`config.py`)

### Data

| Parameter | Default | Description |
|-----------|---------|-------------|
| `data_root` | `"data/lidc_dataset"` | Root of the dataset |
| `train_split` | `"train"` | Subdirectory name for training data |
| `val_split` | `"test"` | Subdirectory name for validation/test data |
| `wc` | `-450.0` | CT window centre |
| `ww` | `1000.0` | CT window width |
| `context_slices` | `2` | Neighbouring slices stacked per side (→ 5 channels) |
| `target_size` | `(256, 256)` | Spatial size after resize |

### Training

| Parameter | Default | Description |
|-----------|---------|-------------|
| `batch_size` | `16` | Training batch size |
| `epochs` | `40` | Training epochs |
| `lr` | `3e-4` | Initial learning rate (AdamW) |
| `weight_decay` | `1e-4` | L2 regularisation |
| `grad_clip` | `1.0` | Gradient norm clip |
| `unet_base` | `128` | Base channel width of U-Net |

### Loss Weights

| Parameter | Default | Description |
|-----------|---------|-------------|
| `pos_weight` | `100` | BCE positive-class weight (nodule/background imbalance) |
| `bce_weight` | `1.0` | Weight for BCE term |
| `dice_weight` | `1.0` | Weight for soft-Dice term |
| `focal_weight` | `1.0` | Weight for focal term |

### OACM (Soft Masking — Stage 2A)

| Parameter | Default | Paper symbol | Description |
|-----------|---------|-------------|-------------|
| `lam` | `210` | λ | Data-fidelity balance |
| `theta` | `2` | θ | ADMM penalty |
| `tau` | `0.001` | τ | Gaussian kernel variance |
| `l_max` | `50` | L_max | Outer iterations |
| `k_max` | `1` | K_max | Inner ADMM iterations |
| `eps` | `1e-5` | ε | Convergence tolerance |

### OACM (Hard Masking — Stage 2B)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `lam2` | `104` | Data-fidelity balance |
| `theta2` | `3` | ADMM penalty |
| `tau2` | `0.01` | Gaussian kernel variance |

---

## Model Architecture

### 2.5-D U-Net (`models/unet.py`)

```
Input: (B, 5, 256, 256)  — 5-channel 2.5-D CT stack

Encoder:
  ConvBlock(5  → 128)  → MaxPool
  ConvBlock(128 → 256) → MaxPool
  ConvBlock(256 → 512) → MaxPool

Bottleneck:
  ConvBlock(512 → 1024, dropout=0.3)

Decoder:
  UpConv(1024 → 512) + skip → ConvBlock(1024 → 512)
  UpConv(512  → 256) + skip → ConvBlock(512  → 256)
  UpConv(256  → 128) + skip → ConvBlock(256  → 128)

Head: Conv1×1(128 → 1)  →  raw logits

Output: (B, 1, 256, 256)
```

Each `ConvBlock` = `Conv3×3 → InstanceNorm2d → ReLU → Conv3×3 → InstanceNorm2d → ReLU`.

### Loss Function (`models/losses.py`)

```
L = w_bce · L_BCE  +  w_dice · L_Dice  +  w_focal · L_Focal
```

- **BCE** — binary cross-entropy with per-class positive reweighting (`pos_weight`)
- **Soft-Dice** — differentiable Dice on sigmoid probabilities
- **Focal** — down-weights easy negatives via (1 − p_t)^γ

---

## Evaluation Metrics

| Metric | Symbol | Range | ↑/↓ |
|--------|--------|-------|-----|
| Dice coefficient | DSC | [0, 1] | ↑ |
| Jaccard / IoU | JS | [0, 1] | ↑ |
| Cohen's κ | κ | [-1, 1] | ↑ |
| Hausdorff distance | HD | [0, ∞) | ↓ |

---


### Quantitative Evaluation: Stage 1 (U-Net) vs Soft-Masking vs Hard-Masking

| Metric     | Stage 1 (U-Net) | Soft-Masking | Hard-Masking | Soft vs Stage1 | Hard vs Stage1 |
|------------|-----------------|--------------|--------------|----------------|----------------|
| Dice       | 0.6498          | 0.7366       | 0.7328       | +13.36% ↑      | +12.77% ↑      |
| Jaccard    | 0.5691          | 0.6387       | 0.6308       | +12.22% ↑      | +10.84% ↑      |
| Kappa      | 0.6495          | 0.7362       | 0.7324       | +13.35% ↑      | +12.77% ↑      |
| Hausdorff  | 53.9123         | 28.0149      | 31.4648      | −48.04% ↓*     | −41.64% ↓*     |

* Lower is better for Hausdorff distance.
---

## Outputs

Running `evaluate.py` produces the following files in `cfg.out_dir`:

| File | Description |
|------|-------------|
| `bootstrap_distributions.png` | Bootstrap metric distributions (Fig. 3 style) |
| `zoomed_comparison_contours.png` | Zoomed CT with GT / Stage 1 / Stage 2 contours |
| `per_patient_dice.png` | Per-patient mean Dice bar chart |
| `training_curves.png` | Loss and Dice curves (from `train.py`) |

---

## Pretrained Weights

Place any pretrained checkpoint at the path specified by
`cfg.custom_weights_path` (default: `weights/model_best.pt`). The loader
accepts both full checkpoint dicts (with optimiser state) and bare state dicts.

---

## Limitations

1. The U-Net backbone is a relatively simple architecture; deeper or
   attention-based encoders may extract richer features.
2. The OACM processes slices independently and may struggle when multiple
   nodules are present in the same slice.

---

## Citation

```bibtex
@article{yang2024fssf,
  title   = {Few-shot segmentation framework for lung nodules via an
             optimized active contour model},
  author  = {Yang, Lin and Shao, Dan and Huang, Zhenxing and Geng, Mengxiao
             and Zhang, Na and Chen, Long and Wang, Xi and Liang, Dong
             and Pang, Zhi-Feng and Hu, Zhanli},
  journal = {Medical Physics},
  volume  = {51},
  pages   = {2788--2805},
  year    = {2024},
  doi     = {10.1002/mp.16933}
}
```

---

## License

This project is released under the MIT License.
