"""SegFormer training script — reproducible, ClearML-tracked.

This is the Python-script equivalent of segformer_training_colab_result.ipynb.
Same architecture, same hyperparameters, same loss, same augmentation. The
only differences are:

  1. ClearML auto-tracks every run (metrics, hparams, plots, console, git state)
     via the ``mlops/clearml_tracking.py`` helpers.
  2. Runnable from the Makefile (``make train-segformer``) so report numbers
     are tied to a deterministic command line.
  3. Best checkpoint registered as a ClearML OutputModel so the cascade
     pipeline can pull it by Task ID at inference time — closing the loop
     between training and serving.

Usage
-----
    python mlops/train_segformer.py \\
        --s1-dir    data/sen1floods11/S1 \\
        --label-dir data/sen1floods11/Labels \\
        --splits-dir data/sen1floods11/splits \\
        --ckpt-dir  checkpoints/

The trained checkpoint is saved to ``<ckpt-dir>/segformer_flood_best.pt`` and
also uploaded to the ClearML Task as the canonical output model.
"""

from __future__ import annotations

import argparse
import copy
import random
from pathlib import Path

import numpy as np
import rasterio
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from transformers import SegformerConfig, SegformerForSemanticSegmentation

# ClearML tracking helpers (init / log_epoch / register_model).
from mlops.clearml_tracking import init_clearml_task, log_epoch, register_model

# Constants from EDA — must match benchmark_cascade.py and the notebook.
VV_MEAN, VV_STD = -10.41, 4.14
VH_MEAN, VH_STD = -17.14, 4.68


# ─────────────────────────────────────────────────────────────────────────────
# Dataset (identical to the notebook's class)
# ─────────────────────────────────────────────────────────────────────────────
class Sen1Floods11Dataset(Dataset):
    """Loads S1 SAR (VV, VH) + LabelHand. Masks invalid pixels (-1)."""

    def __init__(self, csv_path: Path, s1_dir: Path, label_dir: Path, augment: bool = False):
        self.s1_dir, self.label_dir, self.augment = s1_dir, label_dir, augment
        self.pairs: list[tuple[str, str]] = []
        with open(csv_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                s1_name, lbl_name = [s.strip() for s in line.split(",")]
                self.pairs.append((s1_name, lbl_name))

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        s1_name, lbl_name = self.pairs[idx]
        with rasterio.open(self.s1_dir / s1_name) as src:
            s1 = src.read().astype(np.float32)              # (2, H, W)
        with rasterio.open(self.label_dir / lbl_name) as src:
            label = src.read(1).astype(np.float32)          # (H, W)

        s1[0] = (s1[0] - VV_MEAN) / VV_STD
        s1[1] = (s1[1] - VH_MEAN) / VH_STD
        s1 = np.nan_to_num(s1, nan=0.0)

        valid_mask = (label != -1).astype(np.float32)
        label = np.clip(label, 0, 1)

        if self.augment:
            s1, label, valid_mask = self._augment(s1, label, valid_mask)

        return {
            "image":      torch.from_numpy(s1.copy()),
            "label":      torch.from_numpy(label.copy()).unsqueeze(0),
            "valid_mask": torch.from_numpy(valid_mask.copy()).unsqueeze(0),
            "chip_id":    s1_name.replace("_S1Hand.tif", ""),
        }

    @staticmethod
    def _augment(image: np.ndarray, label: np.ndarray, valid: np.ndarray):
        if random.random() > 0.5:
            image = np.flip(image, axis=2); label = np.flip(label, axis=1); valid = np.flip(valid, axis=1)
        if random.random() > 0.5:
            image = np.flip(image, axis=1); label = np.flip(label, axis=0); valid = np.flip(valid, axis=0)
        k = random.randint(0, 3)
        if k > 0:
            image = np.rot90(image, k, axes=(1, 2))
            label = np.rot90(label, k, axes=(0, 1))
            valid = np.rot90(valid, k, axes=(0, 1))
        return image, label, valid


# ─────────────────────────────────────────────────────────────────────────────
# Model + loss
# ─────────────────────────────────────────────────────────────────────────────
def build_segformer(device: torch.device) -> nn.Module:
    """Reproduces the architecture from segformer_training_colab_result.ipynb:
    MiT-B2 encoder pretrained on ImageNet, 2-channel patch embed re-init,
    binary segmentation head."""
    config = SegformerConfig.from_pretrained(
        "nvidia/mit-b2",
        num_labels=1,
        num_channels=2,
        id2label={0: "flood"},
        label2id={"flood": 0},
    )
    model = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/mit-b2", config=config, ignore_mismatched_sizes=True,
    )
    return model.to(device)


def segformer_forward(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Upsample logits from H/4 × W/4 back to the input resolution."""
    out = model(pixel_values=x).logits
    return F.interpolate(out, size=x.shape[-2:], mode="bilinear", align_corners=False)


class CombinedLoss(nn.Module):
    """Dice + BCE with masking for invalid (-1) pixels — same as the notebook."""

    def __init__(self, dice_w: float = 0.5, bce_w: float = 0.5, smooth: float = 1.0):
        super().__init__()
        self.dw, self.bw, self.smooth = dice_w, bce_w, smooth

    def forward(self, logits, targets, valid):
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        bce_masked = (bce * valid).sum() / (valid.sum() + 1e-8)
        probs  = torch.sigmoid(logits) * valid
        target = targets * valid
        inter  = (probs * target).sum()
        dice   = (2.0 * inter + self.smooth) / (probs.sum() + target.sum() + self.smooth)
        return self.dw * (1 - dice) + self.bw * bce_masked


def compute_iou(logits, targets, valid, threshold: float = 0.5) -> float:
    preds = (torch.sigmoid(logits) > threshold).float() * valid
    targs = targets * valid
    inter = (preds * targs).sum()
    union = preds.sum() + targs.sum() - inter
    return float((inter / (union + 1e-8)).item())


# ─────────────────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, criterion, device, scaler):
    model.train()
    loss_acc, iou_acc = 0.0, 0.0
    for batch in loader:
        x  = batch["image"].to(device)
        y  = batch["label"].to(device)
        vm = batch["valid_mask"].to(device)
        optimizer.zero_grad()
        with autocast():
            logits = segformer_forward(model, x)
            loss = criterion(logits, y, vm)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        loss_acc += loss.item()
        iou_acc  += compute_iou(logits.float(), y, vm)
    n = len(loader)
    return loss_acc / n, iou_acc / n


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    loss_acc, iou_acc = 0.0, 0.0
    for batch in loader:
        x  = batch["image"].to(device)
        y  = batch["label"].to(device)
        vm = batch["valid_mask"].to(device)
        with autocast():
            logits = segformer_forward(model, x)
            loss = criterion(logits, y, vm)
        loss_acc += loss.item()
        iou_acc  += compute_iou(logits.float(), y, vm)
    n = len(loader)
    return loss_acc / n, iou_acc / n


@torch.no_grad()
def evaluate_iou(model, loader, device, threshold: float = 0.5) -> dict:
    """Aggregate IoU/F1 — used at end-of-training to populate the model
    registry tags on ClearML."""
    model.eval()
    tp = fp = fn = tn = 0
    for batch in loader:
        x  = batch["image"].to(device)
        y  = batch["label"].to(device)
        vm = batch["valid_mask"].to(device)
        logits = segformer_forward(model, x)
        preds  = (torch.sigmoid(logits) > threshold).float()
        v = vm.bool()
        p = preds[v]; t = y[v]
        tp += int(((p == 1) & (t == 1)).sum()); fp += int(((p == 1) & (t == 0)).sum())
        fn += int(((p == 0) & (t == 1)).sum()); tn += int(((p == 0) & (t == 0)).sum())
    iou = tp / max(tp + fp + fn, 1)
    f1  = 2 * tp / max(2 * tp + fp + fn, 1)
    return {"iou": iou, "f1": f1}


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--s1-dir",    required=True)
    p.add_argument("--label-dir", required=True)
    p.add_argument("--splits-dir", required=True)
    p.add_argument("--ckpt-dir", default="checkpoints")
    # Hyperparameters — defaults reproduce segformer_training_colab_result.ipynb.
    p.add_argument("--batch-size",    type=int,   default=16)
    p.add_argument("--epochs",        type=int,   default=100)
    p.add_argument("--lr",            type=float, default=6e-5)
    p.add_argument("--weight-decay",  type=float, default=0.01)
    p.add_argument("--patience",      type=int,   default=15)
    p.add_argument("--lr-patience",   type=int,   default=7)
    p.add_argument("--num-workers",   type=int,   default=4)
    p.add_argument("--seed",          type=int,   default=42)
    args = p.parse_args()

    # ── Reproducibility ─────────────────────────────────────────────────────
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # ── ClearML: every hparam captured & searchable ─────────────────────────
    task = init_clearml_task(
        model_name="segformer-mit-b2",
        hparams={
            "encoder":       "nvidia/mit-b2",
            "in_channels":   2,
            "loss":          "dice+bce (0.5/0.5)",
            "optimizer":     "AdamW",
            "scheduler":     "ReduceLROnPlateau (factor=0.5)",
            "lr":            args.lr,
            "weight_decay":  args.weight_decay,
            "batch_size":    args.batch_size,
            "epochs":        args.epochs,
            "patience":      args.patience,
            "lr_patience":   args.lr_patience,
            "seed":          args.seed,
            "vv_mean":       VV_MEAN, "vv_std": VV_STD,
            "vh_mean":       VH_MEAN, "vh_std": VH_STD,
        },
    )

    # ── Data ─────────────────────────────────────────────────────────────────
    s1_dir, label_dir, splits_dir = Path(args.s1_dir), Path(args.label_dir), Path(args.splits_dir)
    train_ds = Sen1Floods11Dataset(splits_dir / "flood_train_data.csv", s1_dir, label_dir, augment=True)
    val_ds   = Sen1Floods11Dataset(splits_dir / "flood_valid_data.csv", s1_dir, label_dir, augment=False)
    test_ds  = Sen1Floods11Dataset(splits_dir / "flood_test_data.csv",  s1_dir, label_dir, augment=False)

    loader_kwargs = dict(
        batch_size=args.batch_size, num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=2 if args.num_workers > 0 else None,
        pin_memory=True,
    )
    train_loader = DataLoader(train_ds, shuffle=True,  **loader_kwargs)
    val_loader   = DataLoader(val_ds,   shuffle=False, **loader_kwargs)
    test_loader  = DataLoader(test_ds,  shuffle=False, **loader_kwargs)

    print(f"Train: {len(train_ds)}  Val: {len(val_ds)}  Test: {len(test_ds)}")

    # ── Model + optimizer ───────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    model = build_segformer(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=args.lr_patience,
    )
    criterion = CombinedLoss()
    scaler    = GradScaler()

    # ── Training loop ───────────────────────────────────────────────────────
    best_val_iou, best_epoch = 0.0, 0
    best_weights = None
    no_improve = 0
    print(f"{'Epoch':>5} {'TrLoss':>9} {'VlLoss':>9} {'TrIoU':>9} {'VlIoU':>9} {'LR':>10}")
    print("-" * 55)
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_iou = train_one_epoch(model, train_loader, optimizer, criterion, device, scaler)
        vl_loss, vl_iou = validate(model, val_loader, criterion, device)
        lr_now = optimizer.param_groups[0]["lr"]

        log_epoch(epoch, {
            "train_loss": tr_loss, "val_loss": vl_loss,
            "train_iou":  tr_iou,  "val_iou":  vl_iou,
            "lr":         lr_now,
        })

        marker = ""
        if vl_iou > best_val_iou:
            best_val_iou, best_epoch = vl_iou, epoch
            best_weights = copy.deepcopy(model.state_dict())
            no_improve, marker = 0, " *"
        else:
            no_improve += 1
        print(f"{epoch:>5} {tr_loss:>9.4f} {vl_loss:>9.4f} {tr_iou:>9.4f} {vl_iou:>9.4f} {lr_now:>10.2e}{marker}")
        scheduler.step(vl_iou)
        if no_improve >= args.patience:
            print(f"\nEarly stopping at epoch {epoch} (no improvement for {args.patience} epochs)")
            break

    # ── Final evaluation on the test set ────────────────────────────────────
    model.load_state_dict(best_weights)
    test_metrics = evaluate_iou(model, test_loader, device)
    print(f"\nBest val IoU: {best_val_iou:.4f} @ epoch {best_epoch}")
    print(f"Test IoU: {test_metrics['iou']:.4f}  Test F1: {test_metrics['f1']:.4f}")

    # ── Save + register the best checkpoint ─────────────────────────────────
    ckpt_dir = Path(args.ckpt_dir); ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / "segformer_flood_best.pt"
    torch.save({
        "model_state_dict": best_weights,
        "epoch":            best_epoch,
        "val_iou":          best_val_iou,
        "test_iou":         test_metrics["iou"],
        "test_f1":          test_metrics["f1"],
        "config": {
            "model": "segformer-mit-b2",
            "vv_mean": VV_MEAN, "vv_std": VV_STD,
            "vh_mean": VH_MEAN, "vh_std": VH_STD,
        },
    }, ckpt_path)
    print(f"Saved checkpoint → {ckpt_path}")

    register_model(
        weights_path=ckpt_path,
        metadata={
            "best_val_iou": best_val_iou,
            "best_epoch":   best_epoch,
            "test_iou":     test_metrics["iou"],
            "test_f1":      test_metrics["f1"],
        },
        name="segformer_flood_best",
    )
    print(f"Registered as ClearML output model. Task ID: {task.id}")


if __name__ == "__main__":
    main()
