"""
================================================================================
EXP-05: V5 Combined Universal + Kaggle Boost DINOv3 ViT Training Pipeline
Author: Hoang Tuan (Workspace: /workspace/hoangtuan/deepfake-ViT)
Conforms to: LOGGING_CHECKPOINT_RULES.md, FOLDER_STRUCTURE.md
Dataset: 
  - Train: data/splits/train_v5_combined_universal_kaggle_boost.csv (54,000 samples)
  - Val:   data/splits/val_v5_combined_universal_kaggle_boost.csv (6,000 samples)
  - Test:  data/splits/test_balanced_fixed_zero_leakage.csv (2,354 samples, 40+ methods)
================================================================================
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path
from datetime import datetime

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
PROJECT_ROOT = Path("/workspace/hoangtuan/deepfake-ViT")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.dinov3_vit import build_dinov3_classifier
from src.training.losses import LabelSmoothingCrossEntropy
from src.training.ema import ModelEMA


def get_train_transforms(img_size: int = 256):
    return T.Compose([
        T.Resize((img_size, img_size), interpolation=T.InterpolationMode.BICUBIC),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomApply([T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05)], p=0.5),
        T.RandomApply([T.GaussianBlur(kernel_size=(3, 5), sigma=(0.1, 2.0))], p=0.3),
        T.RandomApply([T.RandomAdjustSharpness(sharpness_factor=2.0)], p=0.3),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def get_eval_transforms(img_size: int = 256):
    return T.Compose([
        T.Resize((img_size, img_size), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


class DeepfakeDataset(Dataset):
    def __init__(self, csv_path_or_df, transform=None):
        if isinstance(csv_path_or_df, (str, Path)):
            self.df = pd.read_csv(csv_path_or_df)
        else:
            self.df = csv_path_or_df.reset_index(drop=True)
            
        self.paths = self.df["path"].values
        self.labels = self.df["label"].values
        self.methods = self.df["method"].values if "method" in self.df.columns else np.array(["unknown"] * len(self.df))
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


def get_parameter_groups(model, base_lr=2e-5, head_lr=5e-4, weight_decay=0.05):
    backbone_params = []
    head_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "head" in name:
            head_params.append(param)
        else:
            backbone_params.append(param)
            
    return [
        {"params": backbone_params, "lr": base_lr, "weight_decay": weight_decay},
        {"params": head_params, "lr": head_lr, "weight_decay": weight_decay * 0.1}
    ]


def train_epoch(model, loader, criterion, optimizer, device, ema=None):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, targets, _ in tqdm(loader, desc="Training", leave=False):
        images, targets = images.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        optimizer.zero_grad()

        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=True):
            outputs = model(images)
            loss = criterion(outputs, targets)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if ema:
            ema.update(model)

        total_loss += loss.item() * targets.size(0)
        _, preds = torch.max(outputs.float(), 1)
        correct += (preds == targets).sum().item()
        total += targets.size(0)

    return total_loss / max(1, total), correct / max(1, total)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds, all_probs, all_targets = [], [], []

    for images, targets, _ in tqdm(loader, desc="Evaluating", leave=False):
        images, targets = images.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=True):
            outputs = model(images)
            loss = criterion(outputs, targets) if criterion else torch.tensor(0.0)

        outputs_f32 = outputs.float()
        probs = F.softmax(outputs_f32, dim=1)[:, 1]
        _, preds = torch.max(outputs_f32, 1)

        total_loss += loss.item() * targets.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())
        all_targets.extend(targets.cpu().numpy())

    all_preds = np.array(all_preds)
    all_probs = np.nan_to_num(np.array(all_probs), nan=0.5)
    all_targets = np.array(all_targets)

    acc = accuracy_score(all_targets, all_preds)
    prec = precision_score(all_targets, all_preds, zero_division=0)
    rec = recall_score(all_targets, all_preds, zero_division=0)
    f1 = f1_score(all_targets, all_preds, zero_division=0)
    try:
        auc = roc_auc_score(all_targets, all_probs) if len(np.unique(all_targets)) > 1 else 0.5
    except Exception:
        auc = 0.5
    cm = confusion_matrix(all_targets, all_preds).tolist()

    return {
        "loss": total_loss / max(1, len(all_targets)),
        "acc": float(acc),
        "prec": float(prec),
        "rec": float(rec),
        "f1": float(f1),
        "auc": float(auc),
        "cm": cm,
        "probs": all_probs,
        "preds": all_preds,
        "targets": all_targets,
    }


def run_training(args):
    print("=" * 80)
    print("🚀 EXP-05: TRAINING DINOv3 ViT ON V5 COMBINED DATASET (60,000 SAMPLES)")
    print("=" * 80)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # 1. Setup Run Directories
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{ts}_v5_combined_universal"
    run_dir = PROJECT_ROOT / "experiments" / "runs" / run_name
    ckpt_dir = run_dir / "checkpoints"
    logs_dir = run_dir / "logs"
    metrics_dir = run_dir / "metrics"
    for d in [ckpt_dir, logs_dir, metrics_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Global checkpoint dir
    global_ckpt_dir = PROJECT_ROOT / "experiments" / "checkpoints" / "exp05_v5_combined"
    global_ckpt_dir.mkdir(parents=True, exist_ok=True)

    print(f"📁 Run Directory: {run_dir}")

    # 2. Prepare Data
    df_train = pd.read_csv(args.train_csv)
    df_val   = pd.read_csv(args.val_csv)
    df_test  = pd.read_csv(args.test_csv)

    print(f"📊 Loaded Train V5 : {len(df_train):,} samples (Real: {(df_train['label']==0).sum():,}, Fake: {(df_train['label']==1).sum():,})")
    print(f"📊 Loaded Val V5   : {len(df_val):,} samples (Real: {(df_val['label']==0).sum():,}, Fake: {(df_val['label']==1).sum():,})")
    print(f"📊 Loaded Test V5  : {len(df_test):,} samples (Real: {(df_test['label']==0).sum():,}, Fake: {(df_test['label']==1).sum():,})")

    train_ds = DeepfakeDataset(df_train, transform=get_train_transforms(args.img_size))
    val_ds   = DeepfakeDataset(df_val, transform=get_eval_transforms(args.img_size))
    test_ds  = DeepfakeDataset(df_test, transform=get_eval_transforms(args.img_size))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size * 2, shuffle=False, num_workers=args.workers, pin_memory=True)
    test_loader  = DataLoader(test_ds, batch_size=args.batch_size * 2, shuffle=False, num_workers=args.workers, pin_memory=True)

    # 3. Model Build & Pre-trained Initialization
    weights_path = str(PROJECT_ROOT / "models" / "dinov3_small" / "model.safetensors")
    print(f"🧠 Building DINOv3 ViT from: {weights_path}")
    model = build_dinov3_classifier(weights_path=weights_path, img_size=args.img_size, device=str(device))

    # Loss & Optimizer
    criterion = LabelSmoothingCrossEntropy(smoothing=0.05)
    param_groups = get_parameter_groups(model, base_lr=args.base_lr, head_lr=args.head_lr, weight_decay=args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    ema = ModelEMA(model, decay=0.999) if args.use_ema else None

    # History Tracking
    history = {
        "epoch": [],
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "val_auc": [],
        "val_f1": []
    }

    best_val_auc = 0.0
    best_epoch = 0

    print(f"\n⚡ Starting Training ({args.epochs} Epochs, Batch Size: {args.batch_size}, Base LR: {args.base_lr}, Head LR: {args.head_lr})...")
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_epoch(model, train_loader, criterion, optimizer, device, ema)
        
        eval_model = ema.module if ema else model
        val_res = evaluate(eval_model, val_loader, criterion, device)
        scheduler.step()

        elapsed = time.time() - t0
        print(f"[Epoch {epoch:02d}/{args.epochs:02d}] "
              f"Train Loss: {tr_loss:.4f} | Acc: {tr_acc*100:.2f}% || "
              f"Val Loss: {val_res['loss']:.4f} | Acc: {val_res['acc']*100:.2f}% | AUC: {val_res['auc']*100:.2f}% | F1: {val_res['f1']*100:.2f}% ({elapsed:.1f}s)")

        history["epoch"].append(epoch)
        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(val_res["loss"])
        history["val_acc"].append(val_res["acc"])
        history["val_auc"].append(val_res["auc"])
        history["val_f1"].append(val_res["f1"])

        # Save Last Checkpoint (Conforming to LOGGING_CHECKPOINT_RULES.md)
        last_ckpt_payload = {
            "model_state_dict": (ema.module if ema else model).state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "epoch": epoch,
            "best_val_auc": best_val_auc,
            "history": history,
            "config": vars(args),
            "timestamp": datetime.now().isoformat()
        }
        torch.save(last_ckpt_payload, ckpt_dir / "v5_combined_last.pt")

        # Save Best Checkpoint
        if val_res["auc"] > best_val_auc:
            best_val_auc = val_res["auc"]
            best_epoch = epoch
            torch.save(last_ckpt_payload, ckpt_dir / "v5_combined_best.pt")
            torch.save(last_ckpt_payload, global_ckpt_dir / "best_model.pt")
            print(f"   🏆 New Best Model Saved! (Val AUC: {best_val_auc*100:.2f}% at Epoch {epoch})")

    total_train_time = time.time() - start_time
    print(f"\n🎉 Training Complete in {total_train_time/60:.2f} mins! Best Val AUC: {best_val_auc*100:.2f}% (Epoch {best_epoch})")

    # 4. Comprehensive Evaluation on Fixed Zero-Leakage Test Suite
    print("\n" + "=" * 80)
    print("🧪 EVALUATING ON FIXED ZERO-LEAKAGE BENCHMARK TEST SUITE (2,354 SAMPLES)")
    print("=" * 80)

    # Load best weights
    best_ckpt = torch.load(global_ckpt_dir / "best_model.pt", map_location=device, weights_only=False)
    eval_model = model
    eval_model.load_state_dict(best_ckpt["model_state_dict"])
    
    test_res = evaluate(eval_model, test_loader, nn.CrossEntropyLoss(), device)
    
    print(f"\n📊 FINAL BENCHMARK TEST METRICS:")
    print(f"   • Test Accuracy  : {test_res['acc']*100:.2f}%")
    print(f"   • Test ROC-AUC   : {test_res['auc']*100:.2f}%")
    print(f"   • Test Precision : {test_res['prec']*100:.2f}%")
    print(f"   • Test Recall    : {test_res['rec']*100:.2f}%")
    print(f"   • Test F1-Score  : {test_res['f1']*100:.2f}%")

    # Per-Method Detailed Breakdown
    df_test["pred"] = test_res["preds"]
    df_test["prob"] = test_res["probs"]

    method_perf = []
    for m in df_test["method"].unique():
        sub = df_test[df_test["method"] == m]
        m_acc = (sub["pred"] == sub["label"]).mean()
        m_prob = sub["prob"].mean()
        method_perf.append({
            "Method": m,
            "Label": "REAL" if sub["label"].iloc[0] == 0 else "FAKE",
            "Samples": len(sub),
            "Accuracy": round(m_acc * 100, 2),
            "Mean Fake Probability": round(m_prob, 4)
        })

    df_method_perf = pd.DataFrame(method_perf).sort_values(by=["Label", "Accuracy"], ascending=[True, False])
    print("\n🔍 PER-METHOD DETECTION ACCURACY:")
    print(df_method_perf.to_string(index=False))

    # Save results
    results_dir = PROJECT_ROOT / "experiments" / "results"
    report_path = results_dir / "v5_combined_training_report.json"
    
    final_report = {
        "timestamp": datetime.now().isoformat(),
        "run_name": run_name,
        "epochs_trained": args.epochs,
        "best_epoch": best_epoch,
        "best_val_auc": float(best_val_auc),
        "test_metrics": {
            "accuracy": float(test_res["acc"]),
            "roc_auc": float(test_res["auc"]),
            "precision": float(test_res["prec"]),
            "recall": float(test_res["rec"]),
            "f1_score": float(test_res["f1"]),
            "confusion_matrix": test_res["cm"]
        },
        "history": history,
        "method_breakdown": df_method_perf.to_dict(orient="records")
    }

    with open(report_path, "w") as f:
        json.dump(final_report, f, indent=2)

    df_method_perf.to_csv(results_dir / "v5_combined_per_method_accuracy.csv", index=False)
    print(f"\n💾 Saved full report to: {report_path}")
    print(f"💾 Saved per-method metrics to: {results_dir / 'v5_combined_per_method_accuracy.csv'}")

    return final_report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DINOv3 ViT on V5 Combined Dataset.")
    parser.add_argument("--train-csv", type=str, default="/workspace/hoangtuan/deepfake-ViT/data/splits/train_v5_combined_universal_kaggle_boost.csv")
    parser.add_argument("--val-csv", type=str, default="/workspace/hoangtuan/deepfake-ViT/data/splits/val_v5_combined_universal_kaggle_boost.csv")
    parser.add_argument("--test-csv", type=str, default="/workspace/data/zero_leakage_benchmark_fixed/test_balanced_fixed_zero_leakage.csv")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--base-lr", type=float, default=2e-5)
    parser.add_argument("--head-lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--img-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use-ema", action="store_true", default=True)
    args = parser.parse_args()

    run_training(args)
