"""EXP-02: Enhanced Deepfake ViT Training Pipeline.

Key Improvements:
1. Architecture: DINOv3 ViT-S/16 with Enhanced MLP Classifier Head (LayerNorm, Dropout, GELU)
2. Loss: Label Smoothing (eps=0.05) & Focal Loss support for hard sample mining
3. Regularization: Model Exponential Moving Average (EMA) with decay=0.999
4. Optimization: Layer-wise Learning Rate Decay (LLRD gamma=0.85), AdamW, Cosine Annealing
5. Augmentation: Domain-Agnostic with ColorJitter, GaussianBlur, Sharpness, RandomErasing, Lighting variation
6. Evaluation: Best Validation Checkpointing (ROC-AUC & Acc), Test-Time Augmentation (TTA), Optimal Threshold (Youden's J)
"""
import argparse
import copy
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from src.models.dinov3_vit import load_dinov3
from src.training.losses import LabelSmoothingCrossEntropy, FocalLoss
from src.training.ema import ModelEMA
from src.eval.tta import predict_batch_with_tta

IMG_SIZE = 256
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class EnhancedDinoViTClassifier(nn.Module):
    """DINOv3 ViT with LayerNorm, Dropout, and 2-layer GELU MLP head."""

    def __init__(
        self,
        backbone: nn.Module,
        num_classes: int = 2,
        hidden_dim: int = 384,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.backbone = backbone
        embed_dim = backbone.embed_dim

        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x)  # CLS token (B, embed_dim)
        return self.head(feat)


class FaceDeepfakeDataset(Dataset):
    def __init__(self, csv_path: str, transform=None):
        df = pd.read_csv(csv_path)
        img_col = "image_path" if "image_path" in df.columns else "path"
        lbl_col = "label" if "label" in df.columns else "true_label"
        self.samples = list(zip(df[img_col].tolist(), df[lbl_col].tolist()))
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception:
            image = Image.new('RGB', (IMG_SIZE, IMG_SIZE), (0, 0, 0))

        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.long)


def get_train_transform():
    return T.Compose([
        T.Resize((IMG_SIZE, IMG_SIZE), interpolation=T.InterpolationMode.BICUBIC),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomRotation(degrees=10),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        T.RandomApply([T.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))], p=0.25),
        T.RandomAdjustSharpness(sharpness_factor=1.5, p=0.25),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        T.RandomErasing(p=0.20, scale=(0.02, 0.20), ratio=(0.3, 3.3), value='random'),
    ])


def get_eval_transform():
    return T.Compose([
        T.Resize((IMG_SIZE, IMG_SIZE), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def build_llrd_optimizer(model, base_lr=1e-5, head_lr=1e-3, decay_rate=0.85, weight_decay=0.05):
    """Layer-wise Learning Rate Decay (LLRD) for ViT."""
    opt_groups = []
    num_layers = len(model.backbone.layer)

    # Classification Head
    opt_groups.append({
        'params': [p for p in model.head.parameters() if p.requires_grad],
        'lr': head_lr,
        'weight_decay': weight_decay,
    })

    # Backbone Norm
    opt_groups.append({
        'params': [p for p in model.backbone.norm.parameters() if p.requires_grad],
        'lr': base_lr,
        'weight_decay': weight_decay,
    })

    # Transformer Blocks (decay backwards from layer 11 to 0)
    for layer_idx in range(num_layers - 1, -1, -1):
        layer_lr = base_lr * (decay_rate ** (num_layers - 1 - layer_idx))
        opt_groups.append({
            'params': [p for p in model.backbone.layer[layer_idx].parameters() if p.requires_grad],
            'lr': layer_lr,
            'weight_decay': weight_decay,
        })

    # Patch & Token Embeddings
    embed_lr = base_lr * (decay_rate ** num_layers)
    opt_groups.append({
        'params': [p for p in model.backbone.embeddings.parameters() if p.requires_grad],
        'lr': embed_lr,
        'weight_decay': weight_decay,
    })

    optimizer = torch.optim.AdamW(opt_groups)
    return optimizer


@torch.no_grad()
def evaluate_model(model, loader, device, criterion=None, use_tta=False):
    model.eval()
    all_y, all_prob = [], []
    total_loss, n_batches = 0.0, 0

    for x, y in loader:
        x, y_dev = x.to(device), y.to(device)
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            if use_tta:
                probs = predict_batch_with_tta(model, x, use_flips=True, use_multi_lighting=True)
                loss = None
            else:
                logits = model(x)
                if criterion:
                    loss = criterion(logits, y_dev)
                    total_loss += loss.item()
                    n_batches += 1
                probs = F.softmax(logits, dim=-1)[:, 1]

        all_y.extend(y.tolist())
        all_prob.extend(probs.cpu().numpy().tolist())

    y_arr = np.array(all_y)
    prob_arr = np.array(all_prob)

    pred_05 = (prob_arr >= 0.5).astype(int)
    acc_05 = float(accuracy_score(y_arr, pred_05))
    auc = float(roc_auc_score(y_arr, prob_arr))

    # Optimal Threshold via Youden's J
    fpr, tpr, thresholds = roc_curve(y_arr, prob_arr)
    j_scores = tpr - fpr
    opt_idx = np.argmax(j_scores)
    opt_tau = float(thresholds[opt_idx])
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
    parser = argparse.ArgumentParser(description="EXP-02 ViT Fine-Tuning")
    parser.add_argument("--train-csv", default="data/splits/train_domain_balanced.csv")
    parser.add_argument("--val-csv", default="data/splits/val_domain_balanced.csv")
    parser.add_argument("--test-csv", default="data/splits/test_balanced.csv")
    parser.add_argument("--model-weights", default="experiments/checkpoints/weights/dinov3-vits16plus-pretrain-lvd1689m/model-3.safetensors")
    parser.add_argument("--output-dir", default="experiments/checkpoints")
    parser.add_argument("--run-name", default="dinov3_vit_exp02")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr-backbone", type=float, default=1e-5)
    parser.add_argument("--lr-head", type=float, default=1e-3)
    parser.add_argument("--decay-rate", type=float, default=0.85)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--focal-gamma", type=float, default=1.5)
    parser.add_argument("--use-focal", action="store_true", help="Use Focal Loss instead of Label Smoothing CE")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Seeding
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 EXP-02 Training Pipeline on {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    # Data
    train_ds = FaceDeepfakeDataset(args.train_csv, transform=get_train_transform())
    val_ds = FaceDeepfakeDataset(args.val_csv, transform=get_eval_transform())
    test_ds = FaceDeepfakeDataset(args.test_csv, transform=get_eval_transform())

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    print(f"📊 Datasets: Train={len(train_ds):,} | Val={len(val_ds):,} | Test={len(test_ds):,}")

    # Model
    backbone = load_dinov3(args.model_weights, img_size=IMG_SIZE)
    model = EnhancedDinoViTClassifier(backbone, num_classes=2, hidden_dim=384, dropout=0.2).to(device)

    # EMA
    ema = ModelEMA(model, decay=0.999)

    # Optimizer & Scheduler
    optimizer = build_llrd_optimizer(
        model,
        base_lr=args.lr_backbone,
        head_lr=args.lr_head,
        decay_rate=args.decay_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-7)

    # Loss
    if args.use_focal:
        criterion = FocalLoss(gamma=args.focal_gamma, label_smoothing=args.label_smoothing)
        print(f"✅ Loss: FocalLoss (gamma={args.focal_gamma}, label_smoothing={args.label_smoothing})")
    else:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.label_smoothing)
        print(f"✅ Loss: LabelSmoothingCrossEntropy (smoothing={args.label_smoothing})")

    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))

    os.makedirs(args.output_dir, exist_ok=True)
    best_ckpt_path = os.path.join(args.output_dir, f"{args.run_name}_best.pt")
    last_ckpt_path = os.path.join(args.output_dir, f"{args.run_name}_last.pt")

    best_val_auc = 0.0
    history = []

    print("\n🏁 Starting Training Loop...")
    for epoch in range(args.epochs):
        t0 = time.time()
        model.train()
        total_loss, n_batches = 0.0, 0
        correct_train, total_train = 0, 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1:02d}/{args.epochs:02d}", ncols=100)
        for x, y in pbar:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                logits = model(x)
                loss = criterion(logits, y)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            ema.update(model)

            total_loss += loss.item()
            n_batches += 1
            preds = logits.argmax(dim=-1)
            correct_train += (preds == y).sum().item()
            total_train += y.size(0)

            pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{correct_train/total_train*100:.1f}%")

        scheduler.step()
        train_loss = total_loss / max(1, n_batches)
        train_acc = correct_train / max(1, total_train)

        # Validation with EMA model
        val_res = evaluate_model(ema.module, val_loader, device, criterion=criterion)
        elapsed = time.time() - t0

        print(f"Epoch {epoch+1:02d} ({elapsed:.1f}s) | Train Loss: {train_loss:.4f} Acc: {train_acc*100:.2f}% | "
              f"Val Loss: {val_res['loss']:.4f} Acc: {val_res['accuracy']*100:.2f}% (Opt: {val_res['accuracy_opt']*100:.2f}%) | "
              f"Val AUC: {val_res['roc_auc']:.4f} | F1: {val_res['f1']:.4f}")

        record = {
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_res['loss'],
            'val_acc': val_res['accuracy'],
            'val_acc_opt': val_res['accuracy_opt'],
            'val_auc': val_res['roc_auc'],
            'val_f1': val_res['f1'],
            'val_opt_tau': val_res['opt_tau'],
            'time_sec': elapsed,
        }
        history.append(record)

        # Save Best
        if val_res['roc_auc'] > best_val_auc:
            best_val_auc = val_res['roc_auc']
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': ema.module.state_dict(),
                'val_metrics': val_res,
                'history': history,
                'config': vars(args),
            }, best_ckpt_path)
            print(f"   ⭐ New best checkpoint saved! (Val AUC = {best_val_auc:.4f})")

        # Save Last
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': ema.module.state_dict(),
            'val_metrics': val_res,
            'history': history,
        }, last_ckpt_path)

    # Final Evaluation on Held-Out Test Set
    print("\n🔍 Evaluating Best Model Checkpoint on Held-Out Test Set...")
    best_ckpt = torch.load(best_ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(best_ckpt['model_state_dict'])

    # 1. Standard Test Evaluation
    test_std = evaluate_model(model, test_loader, device, use_tta=False)
    print(f"\n📊 Test Results (Standard, tau=0.5):")
    print(f"   - Accuracy:    {test_std['accuracy']*100:.2f}%")
    print(f"   - Precision:   {test_std['precision']*100:.2f}%")
    print(f"   - Recall:      {test_std['recall']*100:.2f}%")
    print(f"   - F1-Score:    {test_std['f1']:.4f}")
    print(f"   - ROC-AUC:     {test_std['roc_auc']:.4f}")
    print(f"   - Confusion Matrix: TN={test_std['cm'][0][0]}, FP={test_std['cm'][0][1]}, FN={test_std['cm'][1][0]}, TP={test_std['cm'][1][1]}")

    # 2. TTA Test Evaluation
    print("\n🔬 Running Test-Time Augmentation (TTA)...")
    test_tta = evaluate_model(model, test_loader, device, use_tta=True)
    print(f"📊 Test Results (TTA, tau=0.5):")
    print(f"   - Accuracy:    {test_tta['accuracy']*100:.2f}%")
    print(f"   - Precision:   {test_tta['precision']*100:.2f}%")
    print(f"   - Recall:      {test_tta['recall']*100:.2f}%")
    print(f"   - F1-Score:    {test_tta['f1']:.4f}")
    print(f"   - ROC-AUC:     {test_tta['roc_auc']:.4f}")

    # 3. TTA + Optimal Threshold from Val
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
    report_path = "experiments/results/exp02_enhanced_vit_report.json"
    os.makedirs("experiments/results", exist_ok=True)
    with open(report_path, "w") as f:
        json.dump({
            "experiment": "EXP-02",
            "run_name": args.run_name,
            "best_epoch": best_ckpt['epoch'],
            "val_metrics": best_ckpt['val_metrics'],
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


if __name__ == "__main__":
    main()
