"""
train.py
--------
Stage 1 training script: 2.5-D U-Net on the LIDC-IDRI dataset (or any
dataset following the expected directory layout — see data/dataset.py).

Usage
-----
    python train.py

Edit config.py to change paths, hyperparameters, and loss weights.
A pre-trained checkpoint can be supplied via ``cfg.custom_weights_path``
to fine-tune or resume training.

Outputs
-------
checkpoints/
    model_best.pt   — best validation-Dice checkpoint
    model_last.pt   — final epoch checkpoint
    epoch_N.pt      — periodic epoch checkpoints
    iter_N.pt       — periodic iteration checkpoints (optional)
"""

import random
import warnings
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from config import Config
from data.dataset import LIDCDataset, build_index
from models.losses import BCEDiceLoss, dice_coef_from_logits
from models.unet import UNet2p5D
from utils.checkpoint import load_checkpoint, save_checkpoint
from utils.visualization import plot_training_curves

warnings.filterwarnings("ignore")


# ── Reproducibility ───────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ── Single epoch ──────────────────────────────────────────────────────────────

def run_epoch(model, loader, criterion, device, optimizer=None, scaler=None,
              grad_clip: float = 1.0):
    """Run one training or validation epoch.

    Pass ``optimizer`` (and optionally ``scaler``) to enable training mode.
    """
    is_train = optimizer is not None
    model.train(is_train)
    running_loss = running_dice = n_batches = 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        if is_train:
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                logits = model(x)
                loss   = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            with torch.no_grad(), torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                logits = model(x)
                loss   = criterion(logits, y)

        running_loss += loss.item()
        running_dice += dice_coef_from_logits(logits, y)
        n_batches    += 1

    n = max(n_batches, 1)
    return running_loss / n, running_dice / n


# ── Training orchestrator ─────────────────────────────────────────────────────

def train(cfg: Config) -> tuple:
    """Build data loaders, model, and run the training loop.

    Returns
    -------
    (model, history) — trained model and training-curve dict
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    set_seed(cfg.seed)

    # ── Data ──────────────────────────────────────────────────────────────────
    data_root = Path(cfg.data_root)
    train_dir = data_root / cfg.train_split
    val_dir   = data_root / cfg.val_split

    assert train_dir.exists(), f"Train dir not found: {train_dir}"
    assert val_dir.exists(),   f"Val dir not found:   {val_dir}"

    train_items = build_index(train_dir)
    val_items   = build_index(val_dir)
    print(f"Train items: {len(train_items)},  Val items: {len(val_items)}")

    train_ds = LIDCDataset(train_items, wc=cfg.wc, ww=cfg.ww,
                           context=cfg.context_slices,
                           target_size=cfg.target_size, split="train")
    val_ds   = LIDCDataset(val_items,   wc=cfg.wc, ww=cfg.ww,
                           context=cfg.context_slices,
                           target_size=cfg.target_size, split="val")

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=cfg.num_workers, pin_memory=True,
                              drop_last=False)
    val_loader   = DataLoader(val_ds,   batch_size=cfg.batch_size, shuffle=False,
                              num_workers=cfg.num_workers, pin_memory=True,
                              drop_last=False)

    # ── Model, loss, optimizer ────────────────────────────────────────────────
    in_ch = 2 * cfg.context_slices + 1
    model     = UNet2p5D(in_channels=in_ch, base=cfg.unet_base).to(device)
    criterion = BCEDiceLoss(pos_weight=cfg.pos_weight,
                             bce_weight=cfg.bce_weight,
                             dice_weight=cfg.dice_weight,
                             focal_weight=cfg.focal_weight)
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler    = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.epochs, eta_min=1e-6
    )

    history      = {"train_loss": [], "val_loss": [], "train_dice": [], "val_dice": []}
    start_epoch  = global_step = 0
    best_val_dice = -1.0
    out_dir       = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Optional weight loading ───────────────────────────────────────────────
    loaded = False
    for wpath in [cfg.custom_weights_path, cfg.model_weights_path]:
        p = Path(wpath)
        if p.exists():
            print(f"[Weights] Loading from: {p}")
            try:
                e, g, h = load_checkpoint(p, model, optimizer=optimizer,
                                          map_location=device, strict=False)
                start_epoch, global_step = e, g
                if h is not None:
                    history = h
                if history.get("val_dice"):
                    best_val_dice = max(history["val_dice"])
                print(f"  Resumed from epoch {start_epoch}, best_dice={best_val_dice:.4f}")
                loaded = True
            except Exception as ex:
                print(f"  [WARN] Could not load {p}: {ex}")
            break

    if not loaded:
        print("[Weights] No checkpoint found — training from scratch.")

    # Skip training if weights already satisfy the epoch budget
    if cfg.load_custom_weights and loaded and start_epoch >= cfg.epochs:
        print("Skipping training (weights already satisfy epoch budget).")
        return model, history

    # ── Training loop ─────────────────────────────────────────────────────────
    for epoch in range(start_epoch, cfg.epochs):
        train_loss_sum = train_dice_sum = n_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.epochs}", leave=False)
        for x, y in pbar:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                logits = model(x)
                loss   = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            with torch.no_grad():
                batch_dice = dice_coef_from_logits(logits, y)

            train_loss_sum += loss.item()
            train_dice_sum += batch_dice
            n_batches      += 1
            global_step    += 1

            if cfg.save_every_iters > 0 and global_step % cfg.save_every_iters == 0:
                save_checkpoint(out_dir / f"iter_{global_step}.pt",
                                model, optimizer, epoch, global_step, history)

            pbar.set_postfix(loss=f"{loss.item():.4f}", dice=f"{batch_dice:.4f}")

        train_loss = train_loss_sum / max(n_batches, 1)
        train_dice = train_dice_sum / max(n_batches, 1)
        val_loss, val_dice = run_epoch(model, val_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_dice"].append(train_dice)
        history["val_dice"].append(val_dice)

        print(
            f"Epoch {epoch+1:03d} | "
            f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
            f"train_dice={train_dice:.4f}  val_dice={val_dice:.4f}  "
            f"lr={optimizer.param_groups[0]['lr']:.2e}"
        )

        scheduler.step()

        if val_dice > best_val_dice:
            best_val_dice = val_dice
            save_checkpoint(out_dir / "model_best.pt",
                            model, optimizer, epoch + 1, global_step, history)
            print(f"  ✓ New best val_dice={best_val_dice:.4f} — checkpoint saved.")

        if cfg.save_every_epochs > 0 and (epoch + 1) % cfg.save_every_epochs == 0:
            save_checkpoint(out_dir / f"epoch_{epoch+1}.pt",
                            model, optimizer, epoch + 1, global_step, history)

    save_checkpoint(out_dir / "model_last.pt",
                    model, optimizer, cfg.epochs, global_step, history)
    print(f"Training complete.  Checkpoints saved to: {out_dir}")
    return model, history


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    cfg = Config()
    model, history = train(cfg)
    plot_training_curves(history, save_path=str(Path(cfg.out_dir) / "training_curves.png"))
