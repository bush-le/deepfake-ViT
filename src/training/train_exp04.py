"""
EXP-04: Universal Balanced Dataset V4 Deepfake Detection ViT Training Pipeline.
Strictly conforms to:
- LOGGING_CHECKPOINT_RULES.md (Resumability, full-state checkpointing, metrics logging)
- FOLDER_STRUCTURE.md
- Strict Zero-Leakage Universal Splits:
    - data/splits/train_v4_universal_balanced.csv (50,000 samples, 25k Real / 25k Fake across all 38 methods)
    - data/splits/val_v4_universal_balanced.csv   (5,000 samples, 2.5k Real / 2.5k Fake)
    - data/splits/test_v4_universal_balanced.csv  (5,000 samples, 2.5k Real / 2.5k Fake)
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
)

# Project setup
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.dinov3_vit import load_dinov3
from src.training.losses import LabelSmoothingCrossEntropy
from src.training.ema import ModelEMA
from src.eval.tta import predict_batch_with_tta

# 1. Advanced Augmentation Pipeline
def get_train_transforms():
    return T.Compose([
        T.Resize((256, 256), interpolation=T.InterpolationMode.BICUBIC),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomApply([T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05)], p=0.5),
        T.RandomApply([T.GaussianBlur(kernel_size=(3, 5), sigma=(0.1, 2.0))], p=0.3),
        T.RandomApply([T.RandomAdjustSharpness(sharpness_factor=2.0)], p=0.3),
        T.RandomApply([T.RandomAutocontrast()], p=0.2),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

def get_eval_transforms():
    return T.Compose([
        T.Resize((256, 256), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

# 2. Dataset Class
class DeepfakeDataset(Dataset):
    def __init__(self, csv_path, transform=None, max_samples=None):
        df = pd.read_csv(csv_path)
        if max_samples and len(df) > max_samples:
            df = df.sample(n=max_samples, random_state=42).reset_index(drop=True)
        self.paths = df["path"].values
        self.labels = df["label"].values
        self.methods = df["method"].values if "method" in df.columns else np.array(["unknown"] * len(df))
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        label = self.labels[idx]
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            img = Image.new("RGB", (256, 256), (0, 0, 0))
        if self.transform:
            tensor = self.transform(img)
        else:
            tensor = T.ToTensor()(img)
        return tensor, torch.tensor(label, dtype=torch.long), idx

# 3. Enhanced Model Architecture
class EnhancedDinoViTClassifier(nn.Module):
    def __init__(self, backbone: nn.Module, num_classes: int = 2, hidden_dim: int = 384, dropout: float = 0.2):
        super().__init__()
        self.backbone = backbone
        embed_dim = backbone.embed_dim
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x)
        return self.head(feat)

def build_llrd_param_groups(model: EnhancedDinoViTClassifier, base_lr: float, head_lr: float, decay_rate: float = 0.8, weight_decay: float = 0.05):
    param_groups = []
    param_groups.append({
        "params": [p for p in model.head.parameters() if p.requires_grad],
        "lr": head_lr,
        "weight_decay": weight_decay,
        "name": "head"
    })
    
    num_layers = len(model.backbone.layer)
    for layer_idx in range(num_layers - 1, -1, -1):
        layer_lr = base_lr * (decay_rate ** (num_layers - 1 - layer_idx))
        block = model.backbone.layer[layer_idx]
        param_groups.append({
            "params": [p for p in block.parameters() if p.requires_grad],
            "lr": layer_lr,
            "weight_decay": weight_decay,
            "name": f"layer_{layer_idx}"
        })
        
    if hasattr(model.backbone, "norm"):
        param_groups.append({
            "params": [p for p in model.backbone.norm.parameters() if p.requires_grad],
            "lr": base_lr,
            "weight_decay": weight_decay,
            "name": "norm"
        })
        
    embed_lr = base_lr * (decay_rate ** num_layers)
    embed_params = [p for p in model.backbone.embeddings.parameters() if p.requires_grad]
    if embed_params:
        param_groups.append({
            "params": embed_params,
            "lr": embed_lr,
            "weight_decay": weight_decay,
            "name": "embeddings"
        })
        
    return param_groups

@torch.no_grad()
def evaluate_model(model, loader, device, criterion=None, use_tta=False):
    model.eval()
    all_y, all_prob, all_idx = [], [], []
    total_loss, n_batches = 0.0, 0

    for x, y, idxs in loader:
        x, y_dev = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        if use_tta:
            probs = predict_batch_with_tta(model, x, use_flips=True, use_multi_lighting=True)
        else:
            with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                logits = model(x)
                if criterion:
                    loss = criterion(logits, y_dev)
                    total_loss += loss.item()
                    n_batches += 1
                probs = F.softmax(logits.float(), dim=-1)[:, 1]

        all_y.extend(y.tolist())
        all_prob.extend(probs.float().cpu().numpy().tolist())
        all_idx.extend(idxs.tolist())

    y_arr = np.array(all_y)
    prob_arr = np.array(all_prob)

    pred_05 = (prob_arr >= 0.5).astype(int)
    acc_05 = float(accuracy_score(y_arr, pred_05))
    try:
        auc = float(roc_auc_score(y_arr, prob_arr))
    except Exception:
        auc = 0.5

    try:
        fpr, tpr, thresholds = roc_curve(y_arr, prob_arr)
        j_scores = tpr - fpr
        opt_idx = np.argmax(j_scores)
        opt_tau = float(thresholds[opt_idx])
    except Exception:
        opt_tau = 0.5
        
    pred_opt = (prob_arr >= opt_tau).astype(int)
    acc_opt = float(accuracy_score(y_arr, pred_opt))
    val_loss = (total_loss / max(1, n_batches)) if (criterion and not use_tta) else 0.0

    return {
        'loss': val_loss,
        'accuracy': acc_05,
        'roc_auc': auc,
        'precision': float(precision_score(y_arr, pred_05, zero_division=0)),
        'recall': float(recall_score(y_arr, pred_05, zero_division=0)),
        'f1': float(f1_score(y_arr, pred_05, zero_division=0)),
        'opt_tau': opt_tau,
        'accuracy_opt': acc_opt,
        'f1_opt': float(f1_score(y_arr, pred_opt, zero_division=0)),
        'probs': prob_arr,
        'labels': y_arr,
        'indices': all_idx,
        'cm': confusion_matrix(y_arr, pred_05, labels=[0, 1]).tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description="EXP-04 Universal Balanced ViT Training")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--base-lr", type=float, default=1e-5)
    parser.add_argument("--head-lr", type=float, default=1e-3)
    parser.add_argument("--decay-rate", type=float, default=0.8)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--output-dir", type=str, default=str(PROJECT_ROOT / "experiments" / "checkpoints" / "exp04_dinov3_v4universal"))
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = PROJECT_ROOT / "experiments" / "plots"
    results_dir = PROJECT_ROOT / "experiments" / "results"
    plots_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 80)
    print(f"🚀 EXP-04: UNIVERSAL BALANCED V4 DINOv3 ViT TRAINING")
    print(f"Device: {device} | Output Dir: {output_dir}")
    print("=" * 80)

    # Load Universal V4 Splits
    train_csv = PROJECT_ROOT / "data" / "splits" / "train_v4_universal_balanced.csv"
    val_csv   = PROJECT_ROOT / "data" / "splits" / "val_v4_universal_balanced.csv"
    test_csv  = PROJECT_ROOT / "data" / "splits" / "test_v4_universal_balanced.csv"

    print(f"Loading Train V4 Dataset from: {train_csv.name}")
    train_dataset = DeepfakeDataset(train_csv, transform=get_train_transforms())
    val_dataset   = DeepfakeDataset(val_csv, transform=get_eval_transforms())
    test_dataset  = DeepfakeDataset(test_csv, transform=get_eval_transforms())

    print(f"  • Train V4: {len(train_dataset):,} samples")
    print(f"  • Val V4:   {len(val_dataset):,} samples")
    print(f"  • Test V4:  {len(test_dataset):,} samples\\n")

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0), drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size * 2, shuffle=False,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda")
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size * 2, shuffle=False,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda")
    )

    # Build Model
    print("Loading Pretrained DINOv3 ViT Backbone...")
    backbone = load_dinov3(pretrained=True, num_classes=0)
    backbone.unfreeze_top_k_layers(3)
    model = EnhancedDinoViTClassifier(backbone, num_classes=2, hidden_dim=384, dropout=0.2).to(device)

    criterion = LabelSmoothingCrossEntropy(smoothing=args.label_smoothing).to(device)
    eval_criterion = nn.CrossEntropyLoss().to(device)

    param_groups = build_llrd_param_groups(
        model, base_lr=args.base_lr, head_lr=args.head_lr,
        decay_rate=args.decay_rate, weight_decay=args.weight_decay
    )
    optimizer = torch.optim.AdamW(param_groups)

    total_steps = len(train_loader) * args.epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-7)

    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == "cuda"))
    model_ema = ModelEMA(model, decay=0.999)

    start_epoch = 1
    best_val_auc = 0.0
    history = []

    # Checkpoint Resume
    latest_ckpt_path = output_dir / "latest_checkpoint.pt"
    best_ckpt_path = output_dir / "best_checkpoint.pt"

    if args.resume and latest_ckpt_path.exists():
        print(f"Resuming from {latest_ckpt_path}...")
        ckpt = torch.load(latest_ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        model_ema.module.load_state_dict(ckpt["ema_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        scheduler.load_state_dict(ckpt["scheduler_state"])
        scaler.load_state_dict(ckpt["scaler_state"])
        start_epoch = ckpt["epoch"] + 1
        best_val_auc = ckpt["best_val_auc"]
        history = ckpt.get("history", [])
        print(f"Resumed at Epoch {start_epoch}, Best Val AUC so far: {best_val_auc:.4f}")

    # Training Loop
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        train_loss, train_correct, total_train = 0.0, 0, 0
        t0 = time.time()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", leave=True)
        for step, (images, labels, _) in enumerate(pbar):
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                logits = model(images)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            scheduler.step()
            model_ema.update(model)

            train_loss += loss.item() * len(labels)
            preds = logits.argmax(dim=-1)
            train_correct += (preds == labels).sum().item()
            total_train += len(labels)

            current_loss = train_loss / total_train
            current_acc = train_correct / total_train * 100.0
            pbar.set_postfix({"loss": f"{current_loss:.4f}", "acc": f"{current_acc:.2f}%", "lr": f"{optimizer.param_groups[0]['lr']:.2e}"})

        epoch_time = time.time() - t0
        train_loss = train_loss / total_train
        train_acc = train_correct / total_train * 100.0

        val_metrics = evaluate_model(model_ema.module, val_loader, device, criterion=eval_criterion)
        val_acc = val_metrics["accuracy"] * 100.0
        val_auc = val_metrics["roc_auc"]
        val_f1 = val_metrics["f1"]
        opt_acc = val_metrics["accuracy_opt"] * 100.0

        print(f"Epoch {epoch:02d} ({epoch_time:.1f}s) | Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_metrics['loss']:.4f} Acc: {val_acc:.2f}% (Opt: {opt_acc:.2f}%) | "
              f"Val AUC: {val_auc:.4f} | F1: {val_f1:.4f}")

        epoch_record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_metrics["loss"],
            "val_acc": val_acc,
            "val_auc": val_auc,
            "val_f1": val_f1,
            "val_precision": val_metrics["precision"],
            "val_recall": val_metrics["recall"],
            "opt_tau": val_metrics["opt_tau"],
            "opt_acc": opt_acc,
            "epoch_time": epoch_time
        }
        history.append(epoch_record)

        state = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "ema_state": model_ema.module.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "scaler_state": scaler.state_dict(),
            "best_val_auc": best_val_auc,
            "config": vars(args),
            "history": history
        }
        torch.save(state, latest_ckpt_path)

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            state["best_val_auc"] = best_val_auc
            torch.save(state, best_ckpt_path)
            print(f"  ⭐ New Best Checkpoint saved! Val AUC: {best_val_auc:.4f}")

    # Final Evaluation on Test V4
    print("\n" + "=" * 80)
    print("🎯 FINAL EVALUATION ON INDEPENDENT TEST V4 UNIVERSAL SET")
    print("=" * 80)

    best_ckpt = torch.load(best_ckpt_path, map_location=device)
    model.load_state_dict(best_ckpt["ema_state"])

    print("\nEvaluating on Test V4 Universal Balanced (5,000 samples)...")
    test_v4_std = evaluate_model(model, test_loader, device, criterion=eval_criterion, use_tta=False)
    test_v4_tta = evaluate_model(model, test_loader, device, criterion=eval_criterion, use_tta=True)

    print(f"  • Test V4 Standard : Acc: {test_v4_std['accuracy']*100:.2f}% | AUC: {test_v4_std['roc_auc']:.4f} | F1: {test_v4_std['f1']:.4f}")
    print(f"  • Test V4 + TTA    : Acc: {test_v4_tta['accuracy']*100:.2f}% | AUC: {test_v4_tta['roc_auc']:.4f} | F1: {test_v4_tta['f1']:.4f}")

    # Per-Domain Breakdown on Test V4
    print("\nComputing Per-Method Breakdown on Test V4...")
    test_df = pd.read_csv(test_csv)
    test_methods = test_df["method"].values
    per_method_records = []

    for m in sorted(list(set(test_methods))):
        mask = (test_methods == m)
        if mask.sum() > 0:
            m_labels = test_v4_tta["labels"][mask]
            m_probs = test_v4_tta["probs"][mask]
            m_preds = (m_probs >= test_v4_tta["opt_tau"]).astype(int)
            m_acc = accuracy_score(m_labels, m_preds) * 100.0
            per_method_records.append({
                "Method / Domain": m,
                "Type": "Real" if "Real" in m else "Fake",
                "Sample Count": int(mask.sum()),
                "Accuracy (TTA)": f"{m_acc:.2f}%",
                "Mean Prob Fake": f"{np.mean(m_probs):.4f}"
            })

    df_method_perf = pd.DataFrame(per_method_records).sort_values(by=["Type", "Method / Domain"]).reset_index(drop=True)
    method_perf_csv = results_dir / "exp04_test_v4_method_breakdown.csv"
    df_method_perf.to_csv(method_perf_csv, index=False)
    print(f"\n💾 Saved Method Breakdown to: {method_perf_csv}")
    print(df_method_perf.to_string(index=False))

    summary = {
        "best_val_auc": best_val_auc,
        "test_v4_std_acc": test_v4_std["accuracy"],
        "test_v4_std_auc": test_v4_std["roc_auc"],
        "test_v4_tta_acc": test_v4_tta["accuracy"],
        "test_v4_tta_auc": test_v4_tta["roc_auc"],
        "history": history
    }
    with open(output_dir / "eval_results.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n🎉 EXP-04 Training & Comprehensive Evaluation Complete!")


if __name__ == "__main__":
    main()
