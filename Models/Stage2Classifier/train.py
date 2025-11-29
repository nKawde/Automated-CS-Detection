#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import math
import json
import argparse
from pathlib import Path
from typing import Tuple, Dict, List

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.cuda.amp import GradScaler, autocast

import torchvision.transforms as T
from torchvision import models

from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_recall_curve, roc_curve, confusion_matrix
)
import matplotlib.pyplot as plt
plt.switch_backend("Agg")


# ------------------------ Utilities ------------------------

def set_seed(seed: int = 52, deterministic: bool = False):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def ensure_dir(p: str):
    Path(p).mkdir(parents=True, exist_ok=True)


def class_counts_from_csv(csv_path: str) -> Dict[int, int]:
    df = pd.read_csv(csv_path)
    if "label" not in df.columns:
        raise ValueError(f"{csv_path} must contain a 'label' column.")
    counts = df["label"].value_counts().to_dict()
    return {int(k): int(v) for k, v in counts.items()}


# ------------------------ Dataset ------------------------

class CSFrameDataset(Dataset):
    def __init__(self, csv_path: str, image_size: int = 260, is_train: bool = True):
        self.df = pd.read_csv(csv_path)
        if "path" not in self.df.columns or "label" not in self.df.columns:
            raise ValueError(f"{csv_path} must have columns: path,label")

        self.paths = self.df["path"].astype(str).tolist()
        self.labels = self.df["label"].astype(int).tolist()

        # EfficientNet-B2 nominal input is 260 px
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)

        if is_train:
            self.transform = T.Compose([
                T.Resize(int(image_size * 1.15)),
                T.RandomResizedCrop(image_size, scale=(0.7, 1.0), ratio=(0.9, 1.1)),
                T.RandomHorizontalFlip(p=0.5),
                T.RandomVerticalFlip(p=0.2),
                T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05, hue=0.02),
                T.ToTensor(),
                T.Normalize(mean, std),
            ])
        else:
            self.transform = T.Compose([
                T.Resize(image_size + 16),
                T.CenterCrop(image_size),
                T.ToTensor(),
                T.Normalize(mean, std),
            ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        y = self.labels[idx]
        try:
            img = Image.open(path).convert("RGB")
        except Exception as e:
            raise RuntimeError(f"Failed to open image: {path} ({e})")
        x = self.transform(img)
        y = torch.tensor([float(y)], dtype=torch.float32)
        return x, y


# ------------------------ Model ------------------------

def build_model(pretrained: bool = False) -> nn.Module:
    # Use torchvision EfficientNet-B2; set weights=None to avoid internet hits
    if pretrained:
        try:
            weights = models.EfficientNet_B2_Weights.IMAGENET1K_V1
        except Exception:
            weights = None
    else:
        weights = None

    net = models.efficientnet_b2(weights=weights)
    in_feats = net.classifier[1].in_features
    net.classifier[1] = nn.Linear(in_feats, 1)  # binary logit
    return net


# ------------------------ Sampler & Loss ------------------------

def make_weighted_sampler(labels: List[int]) -> WeightedRandomSampler:
    # inverse-frequency weights across the two classes
    labels_np = np.array(labels, dtype=np.int64)
    class_counts = np.bincount(labels_np, minlength=2)
    # Avoid division by zero
    class_counts = np.where(class_counts == 0, 1, class_counts)
    class_weights = 1.0 / class_counts
    sample_weights = class_weights[labels_np]
    sample_weights = torch.from_numpy(sample_weights).float()
    return WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)


def make_pos_weight_for_bce(labels: List[int]) -> torch.Tensor:
    labels_np = np.array(labels, dtype=np.int64)
    neg = (labels_np == 0).sum()
    pos = (labels_np == 1).sum()
    # pos_weight multiplies positive examples' loss
    # Common heuristic: pos_weight = neg/pos
    if pos == 0:
        pos_weight = 1.0
    else:
        pos_weight = float(neg) / float(pos)
    return torch.tensor([pos_weight], dtype=torch.float32)


# ------------------------ Metrics & Plots ------------------------

@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    all_logits, all_targets = [], []

    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        logits = model(xb)
        all_logits.append(logits.detach().cpu().numpy())
        all_targets.append(yb.detach().cpu().numpy())

    logits = np.concatenate(all_logits, axis=0).squeeze(1)
    y_true = np.concatenate(all_targets, axis=0).squeeze(1).astype(int)
    probs = 1.0 / (1.0 + np.exp(-logits))

    # AUCs are robust to imbalance
    try:
        roc_auc = roc_auc_score(y_true, probs)
    except Exception:
        roc_auc = float("nan")
    try:
        pr_auc = average_precision_score(y_true, probs)
    except Exception:
        pr_auc = float("nan")

    # Choose threshold to maximize F1 on this evaluation split
    best_thr, best_f1, best_p, best_r = optimal_threshold(y_true, probs)

    return dict(
        roc_auc=float(roc_auc),
        pr_auc=float(pr_auc),
        best_thr=float(best_thr),
        best_f1=float(best_f1),
        best_precision=float(best_p),
        best_recall=float(best_r),
    )


def optimal_threshold(y_true: np.ndarray, p: np.ndarray) -> Tuple[float, float, float, float]:
    pr, rc, thr = precision_recall_curve(y_true, p)
    # precision_recall_curve returns thresholds of len-1 vs pr/rc len
    best_f1, best_thr, best_p, best_r = 0.0, 0.5, 0.0, 0.0
    for i in range(len(thr)):
        f1 = 2 * pr[i] * rc[i] / (pr[i] + rc[i] + 1e-8)
        if f1 > best_f1:
            best_f1 = f1
            best_thr = thr[i]
            best_p = pr[i]
            best_r = rc[i]
    return best_thr, best_f1, best_p, best_r


def plot_curves(y_true: np.ndarray, probs: np.ndarray, out_dir: str, split_name: str, thr: float):
    ensure_dir(out_dir)

    # PR Curve
    pr, rc, _ = precision_recall_curve(y_true, probs)
    ap = average_precision_score(y_true, probs)
    plt.figure()
    plt.plot(rc, pr, label=f"AP={ap:.4f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall ({split_name})")
    plt.legend(loc="best")
    plt.grid(True)
    plt.savefig(os.path.join(out_dir, f"{split_name}_pr_curve.png"), dpi=160)
    plt.close()

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_true, probs)
    auc = roc_auc_score(y_true, probs)
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC={auc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title(f"ROC ({split_name})")
    plt.legend(loc="best")
    plt.grid(True)
    plt.savefig(os.path.join(out_dir, f"{split_name}_roc_curve.png"), dpi=160)
    plt.close()

    # Confusion Matrix at chosen threshold
    y_pred = (probs >= thr).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    plt.figure()
    plt.imshow(cm, interpolation='nearest')
    plt.title(f"Confusion Matrix ({split_name}) @thr={thr:.3f}")
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ["Neg", "Pos"])
    plt.yticks(tick_marks, ["Neg", "Pos"])
    thresh_val = cm.max() / 2.0
    for i in range(2):
        for j in range(2):
            plt.text(j, i, format(cm[i, j], 'd'),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh_val else "black")
    plt.tight_layout()
    plt.ylabel('True')
    plt.xlabel('Pred')
    plt.savefig(os.path.join(out_dir, f"{split_name}_confusion_matrix.png"), dpi=160)
    plt.close()


@torch.no_grad()
def dump_eval_details(model, loader, device, out_dir: str, split_name: str, thr: float):
    model.eval()
    logits_all, y_all = [], []
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        logits = model(xb)
        logits_all.append(logits.detach().cpu().numpy())
        y_all.append(yb.detach().cpu().numpy())

    logits = np.concatenate(logits_all, axis=0).squeeze(1)
    y_true = np.concatenate(y_all, axis=0).squeeze(1).astype(int)
    probs = 1.0 / (1.0 + np.exp(-logits))
    y_pred = (probs >= thr).astype(int)

    # save CSV
    df = pd.DataFrame({"prob": probs, "true": y_true, "pred": y_pred})
    df.to_csv(os.path.join(out_dir, f"{split_name}_preds.csv"), index=False)

    # plots
    plot_curves(y_true, probs, out_dir, split_name, thr)


# ------------------------ Training ------------------------

def train_one_epoch(model, loader, optimizer, scaler, criterion, device, epoch, log_interval=100):
    model.train()
    running = 0.0
    for step, (xb, yb) in enumerate(loader, 1):
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with autocast():
            logits = model(xb)
            loss = criterion(logits, yb)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running += loss.item()
        if (step % log_interval) == 0:
            print(f"Epoch {epoch} | Step {step}/{len(loader)} | Loss {running/log_interval:.4f}")
            running = 0.0


def cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps, min_lr_ratio=0.01):
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step + 1) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        # cosine from 1 -> min_lr_ratio
        return min_lr_ratio + 0.5 * (1 - min_lr_ratio) * (1 + math.cos(math.pi * progress))

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def fit(args):
    set_seed(args.seed, deterministic=args.deterministic)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    ensure_dir(args.out_dir)

    # Datasets
    train_ds = CSFrameDataset(args.train_csv, image_size=args.image_size, is_train=True)
    val_ds   = CSFrameDataset(args.val_csv,   image_size=args.image_size, is_train=False)
    test_ds  = CSFrameDataset(args.test_csv,  image_size=args.image_size, is_train=False)

    # Sampler (to fight imbalance)
    train_labels = pd.read_csv(args.train_csv)["label"].astype(int).tolist()
    sampler = make_weighted_sampler(train_labels)

    # Loaders
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, sampler=sampler,
        num_workers=args.workers, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True
    )

    # Model
    model = build_model(pretrained=args.pretrained).to(device)

    # Loss with pos_weight
    pos_weight = make_pos_weight_for_bce(train_labels).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # Optimizer & Scheduler
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * args.warmup_frac)
    scheduler = cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps, min_lr_ratio=args.min_lr_ratio)

    scaler = GradScaler(enabled=not args.no_amp)

    # Early stopping on val PR-AUC
    best_val_pr = -np.inf
    no_improve = 0
    best_ckpt = os.path.join(args.out_dir, "best.pt")

    history = []

    global_step = 0
    for epoch in range(1, args.epochs + 1):
        train_one_epoch(model, train_loader, optimizer, scaler, criterion, device, epoch, args.log_interval)

        # Step LR for full epoch
        for _ in range(len(train_loader)):
            scheduler.step()
            global_step += 1

        # Evaluate on val
        val_metrics = evaluate(model, val_loader, device)
        print(f"[Val] epoch={epoch} | PR-AUC={val_metrics['pr_auc']:.6f} | ROC-AUC={val_metrics['roc_auc']:.6f} "
              f"| best_thr={val_metrics['best_thr']:.3f} | F1={val_metrics['best_f1']:.4f} "
              f"(P={val_metrics['best_precision']:.4f}, R={val_metrics['best_recall']:.4f})")

        history.append({"epoch": epoch, "val": val_metrics})

        # Save best
        improved = val_metrics["pr_auc"] > best_val_pr + 1e-6
        if improved:
            best_val_pr = val_metrics["pr_auc"]
            no_improve = 0
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scaler_state": scaler.state_dict(),
                "val_metrics": val_metrics,
                "args": vars(args),
            }, best_ckpt)
            print(f"[Checkpoint] Saved new best to {best_ckpt}")
        else:
            no_improve += 1
            print(f"[EarlyStop] no_improve={no_improve}/{args.patience}")

        if no_improve >= args.patience:
            print("[EarlyStop] patience reached.")
            break

    # Load best and run final evals
    if os.path.exists(best_ckpt):
        ckpt = torch.load(best_ckpt, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        print(f"[LoadBest] epoch={ckpt['epoch']} | best Val PR-AUC={ckpt['val_metrics']['pr_auc']:.6f}")

    # Dump curves and preds for val & test
    with torch.no_grad():
        # Use best threshold from validation to report both splits
        val_best_thr = evaluate(model, val_loader, device)["best_thr"]

        dump_eval_details(model, val_loader, device, args.out_dir, "val",  val_best_thr)
        dump_eval_details(model, test_loader, device, args.out_dir, "test", val_best_thr)

    # Save history
    with open(os.path.join(args.out_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)

    print("[Done] Metrics and plots saved to:", args.out_dir)


# ------------------------ CLI ------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Train EfficientNet-B2 for CS binary classification")
    p.add_argument("--train_csv", type=str, required=True)
    p.add_argument("--val_csv",   type=str, required=True)
    p.add_argument("--test_csv",  type=str, required=True)
    p.add_argument("--out_dir",   type=str, default="outputs_b2")

    p.add_argument("--image_size", type=int, default=260)  # B2 nominal
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--epochs",     type=int, default=30)
    p.add_argument("--lr",         type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)

    p.add_argument("--warmup_frac", type=float, default=0.05, help="fraction of total steps for LR warmup")
    p.add_argument("--min_lr_ratio", type=float, default=0.05, help="cosine min LR factor vs base LR")

    p.add_argument("--pretrained", action="store_true", help="use torchvision ImageNet weights if available locally")
    p.add_argument("--cpu", action="store_true", help="force CPU")
    p.add_argument("--no_amp", action="store_true", help="disable mixed precision")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--log_interval", type=int, default=100)

    p.add_argument("--seed", type=int, default=52)
    p.add_argument("--deterministic", action="store_true")
    p.add_argument("--patience", type=int, default=8, help="early stopping patience on Val PR-AUC")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    fit(args)
