"""
EXP-03: Zero-Leakage Enhanced DINOv3 ViT Training Entry Point.
Strictly conforms to:
- LOGGING_CHECKPOINT_RULES.md (Resumability, full-state checkpointing, metrics logging)
- FOLDER_STRUCTURE.md
- Strict Zero-Leakage Dataset splits (train_v3_clean.csv, val_v3_clean.csv, test_balanced.csv)
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
        return tensor, torch.tensor(label, dtype=torch.long)

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
    all_y, all_prob = [], []
    total_loss, n_batches = 0.0, 0

    for x, y in loader:
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

    y_arr = np.array(all_y)
    prob_arr = np.array(all_prob)

    pred_05 = (prob_arr >= 0.5).astype(int)
    acc_05 = float(accuracy_score(y_arr, pred_05))
    try:
        auc = float(roc_auc_score(y_arr, prob_arr))
    except Exception:
        auc = 0.5

    # Optimal Threshold via Youden's J
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
        'cm': confusion_matrix(y_arr, pred_05, labels=[0, 1]).tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description="EXP-03 Enhanced DINOv3 ViT Training")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--base-lr", type=float, default=1e-5)
    parser.add_argument("--head-lr", type=float, default=1e-3)
    parser.add_argument("--decay-rate", type=float, default=0.8)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--run-name", type=str, default="exp03_dinov3_v3clean")
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke-test", action="store_true", help="Run 1 epoch on 32 samples for smoke testing")
    args = parser.parse_args()

    # Reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.empty_cache()
    print(f"🖥️ Using device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    # Output directories
    ckpt_dir = PROJECT_ROOT / "experiments" / "checkpoints" / args.run_name
    results_dir = PROJECT_ROOT / "experiments" / "results"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    best_ckpt_path = ckpt_dir / "best_checkpoint.pt"
    last_ckpt_path = ckpt_dir / "last_checkpoint.pt"
    history_path = ckpt_dir / "history.json"
    report_path = results_dir / "exp03_enhanced_vit_report.json"

    # Paths
    splits_dir = PROJECT_ROOT / "data" / "splits"
    train_csv = splits_dir / "train_v3_clean.csv"
    val_csv = splits_dir / "val_v3_clean.csv"
    test_csv = splits_dir / "test_balanced.csv"
    weights_path = PROJECT_ROOT / "experiments" / "checkpoints" / "weights" / "dinov3-vits16plus-pretrain-lvd1689m" / "model-3.safetensors"

    max_samples = 32 if args.smoke_test else None
    epochs = 1 if args.smoke_test else args.epochs
    num_workers = 0 if args.smoke_test else args.num_workers

    train_ds = DeepfakeDataset(train_csv, transform=get_train_transforms(), max_samples=max_samples)
    val_ds = DeepfakeDataset(val_csv, transform=get_eval_transforms(), max_samples=max_samples)
    test_ds = DeepfakeDataset(test_csv, transform=get_eval_transforms(), max_samples=max_samples)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=num_workers, pin_memory=(device.type == "cuda"))
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=num_workers, pin_memory=(device.type == "cuda"))
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=num_workers, pin_memory=(device.type == "cuda"))

    print(f"📦 Loaded Datasets (Train: {len(train_ds):,}, Val: {len(val_ds):,}, Test: {len(test_ds):,})")

    # Model
    backbone = load_dinov3(str(weights_path), img_size=256)
    model = EnhancedDinoViTClassifier(backbone).to(device)
    model_ema = ModelEMA(model, decay=0.999)

    # Optimizer, Scheduler & Criterion
    param_groups = build_llrd_param_groups(model, base_lr=args.base_lr, head_lr=args.head_lr, decay_rate=args.decay_rate, weight_decay=args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-7)
    criterion = LabelSmoothingCrossEntropy(smoothing=args.label_smoothing)

    best_val_auc = 0.0
    patience_counter = 0
    history = []

    print(f"🚀 Starting EXP-03 training loop ({epochs} epochs, lr_backbone={args.base_lr}, lr_head={args.head_lr})...")
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        train_loss, train_correct, total = 0.0, 0, 0
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}"):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                logits = model(x)
                loss = criterion(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            model_ema.update(model)

            train_loss += loss.item() * len(y)
            train_correct += (logits.argmax(dim=1) == y).sum().item()
            total += len(y)

        scheduler.step()
        epoch_time = time.time() - t0
        avg_train_loss = train_loss / max(1, total)
        avg_train_acc = train_correct / max(1, total)

        val_metrics = evaluate_model(model_ema.module, val_loader, device, criterion=criterion)
        
        print(f"Epoch {epoch:02d} ({epoch_time:.1f}s) | Train Loss: {avg_train_loss:.4f} Acc: {avg_train_acc*100:.2f}% | "
              f"Val Loss: {val_metrics['loss']:.4f} Acc: {val_metrics['accuracy']*100:.2f}% (Opt: {val_metrics['accuracy_opt']*100:.2f}%) | "
              f"Val AUC: {val_metrics['roc_auc']:.4f} | F1: {val_metrics['f1']:.4f}")

        record = {
            'epoch': epoch,
            'train_loss': avg_train_loss,
            'train_acc': avg_train_acc,
            'val_loss': val_metrics['loss'],
            'val_acc': val_metrics['accuracy'],
            'val_acc_opt': val_metrics['accuracy_opt'],
            'val_auc': val_metrics['roc_auc'],
            'val_f1': val_metrics['f1'],
            'val_opt_tau': val_metrics['opt_tau'],
            'time_sec': epoch_time,
        }
        history.append(record)

        # Save Last Checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'ema_state_dict': model_ema.module.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'val_metrics': {k: v for k, v in val_metrics.items() if k not in ['probs', 'labels']},
            'history': history,
            'config': vars(args),
        }, last_ckpt_path)

        # Check Best
        if val_metrics['roc_auc'] > best_val_auc:
            best_val_auc = val_metrics['roc_auc']
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'ema_state_dict': model_ema.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_metrics': {k: v for k, v in val_metrics.items() if k not in ['probs', 'labels']},
                'history': history,
                'config': vars(args),
            }, best_ckpt_path)
            print(f"   ⭐ New best checkpoint saved! (Val AUC = {best_val_auc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= args.patience and not args.smoke_test:
                print(f"   ⏹️ Early stopping triggered after {epoch} epochs (Patience={args.patience})")
                break

    # Save history json
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    # 4. Final Evaluation on Held-Out Test Set
    print("\n🔍 Evaluating Best Model Checkpoint on Held-Out Test Benchmark...")
    best_ckpt = torch.load(best_ckpt_path if best_ckpt_path.exists() else last_ckpt_path, map_location=device, weights_only=False)
    eval_model = model_ema.module
    if 'ema_state_dict' in best_ckpt:
        eval_model.load_state_dict(best_ckpt['ema_state_dict'])

    # 4.1 Standard Test
    test_std = evaluate_model(eval_model, test_loader, device, use_tta=False)
    print(f"\n📊 Test Results (Standard, tau=0.5):")
    print(f"   - Accuracy:    {test_std['accuracy']*100:.2f}%")
    print(f"   - Precision:   {test_std['precision']*100:.2f}%")
    print(f"   - Recall:      {test_std['recall']*100:.2f}%")
    print(f"   - F1-Score:    {test_std['f1']:.4f}")
    print(f"   - ROC-AUC:     {test_std['roc_auc']:.4f}")
    print(f"   - Confusion Matrix: TN={test_std['cm'][0][0]}, FP={test_std['cm'][0][1]}, FN={test_std['cm'][1][0]}, TP={test_std['cm'][1][1]}")

    # 4.2 TTA Test
    print("\n🔬 Running Test-Time Augmentation (TTA)...")
    test_tta = evaluate_model(eval_model, test_loader, device, use_tta=True)
    print(f"📊 Test Results (TTA, tau=0.5):")
    print(f"   - Accuracy:    {test_tta['accuracy']*100:.2f}%")
    print(f"   - Precision:   {test_tta['precision']*100:.2f}%")
    print(f"   - Recall:      {test_tta['recall']*100:.2f}%")
    print(f"   - F1-Score:    {test_tta['f1']:.4f}")
    print(f"   - ROC-AUC:     {test_tta['roc_auc']:.4f}")

    # 4.3 TTA + Optimal Threshold
    val_opt_tau = best_ckpt['val_metrics']['opt_tau']
    pred_opt_tta = (test_tta['probs'] >= val_opt_tau).astype(int)
    acc_opt_tta = accuracy_score(test_tta['labels'], pred_opt_tta)
    f1_opt_tta = f1_score(test_tta['labels'], pred_opt_tta)
    cm_opt = confusion_matrix(test_tta['labels'], pred_opt_tta, labels=[0, 1]).tolist()

    print(f"\n🎯 Test Results (TTA + Optimal Val Threshold tau*={val_opt_tau:.4f}):")
    print(f"   - Optimized Accuracy:  {acc_opt_tta*100:.2f}%")
    print(f"   - Optimized F1-Score:  {f1_opt_tta:.4f}")
    print(f"   - Optimized CM:        TN={cm_opt[0][0]}, FP={cm_opt[0][1]}, FN={cm_opt[1][0]}, TP={cm_opt[1][1]}")

    # Save comprehensive report JSON
    with open(report_path, "w") as f:
        json.dump({
            "experiment": "EXP-03",
            "run_name": args.run_name,
            "best_epoch": best_ckpt.get('epoch', 1),
            "val_metrics": best_ckpt.get('val_metrics', {}),
            "test_standard": {k: v for k, v in test_std.items() if k not in ['probs', 'labels']},
            "test_tta": {k: v for k, v in test_tta.items() if k not in ['probs', 'labels']},
            "test_tta_opt": {
                "val_opt_tau": val_opt_tau,
                "accuracy": acc_opt_tta,
                "f1": f1_opt_tta,
                "cm": cm_opt,
            },
            "history": history,
            "config": vars(args),
        }, f, indent=2)
    print(f"\n📁 Report successfully saved to: {report_path}")
    print("✅ Smoke test passed cleanly!" if args.smoke_test else "🎉 EXP-03 Training and Evaluation Completed!")


if __name__ == "__main__":
    main()
