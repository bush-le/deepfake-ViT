"""
Finetune hoangtuan's v5 combined model (DINOv3 ViT-Small) on the weak-method
dataset (train_v5_weakfix.csv) to fix collapsed deepfake methods.

Differences from train_v5_combined.py:
  - Initializes from experiments/checkpoints/exp05_v5_combined/best_model.pt
  - Method-balanced sampler: 50% real, 50% fake; fake split uniformly across
    methods so every method (esp. weak ones) contributes equally per epoch.
  - Saves to experiments/checkpoints/exp05_v5_weakfix/ (does NOT touch v5 ckpt)
  - Evaluates on the same zero-leakage test set; writes per-method report.

Usage:
  .venv/bin/python scripts/finetune_v5_weakfix.py [--epochs 2] [--tag v1]
"""
import os, sys, json, argparse, time
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms as T

from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix)

PROJECT_ROOT = Path(os.getenv("REPO_ROOT", Path(__file__).resolve().parents[1]))
DATA_ROOT = Path(os.getenv("DF40_ROOT", PROJECT_ROOT / "data"))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.dinov3_vit import build_dinov3_classifier
from src.training.losses import LabelSmoothingCrossEntropy
from src.training.ema import ModelEMA


def get_train_transforms(img_size=256):
    return T.Compose([
        T.Resize((img_size, img_size), interpolation=T.InterpolationMode.BICUBIC),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomApply([T.ColorJitter(0.2, 0.2, 0.2, 0.05)], p=0.5),
        T.RandomApply([T.GaussianBlur((3, 5), (0.1, 2.0))], p=0.3),
        T.RandomApply([T.RandomAdjustSharpness(2.0)], p=0.3),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def get_eval_transforms(img_size=256):
    return T.Compose([
        T.Resize((img_size, img_size), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


class DeepfakeDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.paths = self.df["path"].values
        self.labels = self.df["label"].values.astype(np.int64)
        self.methods = self.df["method"].values if "method" in self.df.columns else np.array(["unknown"] * len(self.df))
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        try:
            img = Image.open(self.paths[idx]).convert("RGB")
        except Exception:
            img = Image.new("RGB", (256, 256), (0, 0, 0))
        if self.transform:
            tensor = self.transform(img)
        else:
            tensor = T.ToTensor()(img)
        return tensor, self.labels[idx], idx


def method_balanced_weights(df):
    """Per-sample weights: P(real)=0.5; P(fake method m)=0.5/num_methods."""
    labels = df["label"].values
    methods = df["method"].values
    num_real = int((labels == 0).sum())
    fake_methods = sorted(set(methods[labels == 1]))
    num_methods = len(fake_methods)
    w = np.ones(len(df), dtype=np.float64)
    # real
    w[labels == 0] = 1.0 / num_real
    # fake: equal total weight per method
    for m in fake_methods:
        idx = (labels == 1) & (methods == m)
        cnt = int(idx.sum())
        w[idx] = 1.0 / (num_methods * cnt)
    w /= w.sum()  # normalize
    return w


def get_parameter_groups(model, base_lr=2e-5, head_lr=5e-4, weight_decay=0.05):
    backbone, head = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        (head if "head" in name else backbone).append(param)
    return [
        {"params": backbone, "lr": base_lr, "weight_decay": weight_decay},
        {"params": head, "lr": head_lr, "weight_decay": weight_decay * 0.1},
    ]


def train_epoch(model, loader, criterion, optimizer, device, ema=None):
    model.train()
    tl, correct, total = 0.0, 0, 0
    for images, targets, _ in tqdm(loader, desc="Train", leave=False):
        images, targets = images.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        optimizer.zero_grad()
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=True):
            outputs = model(images)
            loss = criterion(outputs, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if ema:
            ema.update(model)
        tl += loss.item() * targets.size(0)
        _, preds = torch.max(outputs.float(), 1)
        correct += (preds == targets).sum().item()
        total += targets.size(0)
    return tl / max(1, total), correct / max(1, total)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    preds, probs, targets = [], [], []
    for images, tgt, _ in tqdm(loader, desc="Eval", leave=False):
        images = images.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=True):
            outputs = model(images)
        out32 = outputs.float()
        probs.extend(F.softmax(out32, 1)[:, 1].cpu().numpy())
        preds.extend(torch.max(out32, 1)[1].cpu().numpy())
        targets.extend(tgt.cpu().numpy())
    preds, probs, targets = map(np.array, (preds, probs, targets))
    probs = np.nan_to_num(probs, nan=0.5)
    return {
        "acc": float(accuracy_score(targets, preds)),
        "prec": float(precision_score(targets, preds, zero_division=0)),
        "rec": float(recall_score(targets, preds, zero_division=0)),
        "f1": float(f1_score(targets, preds, zero_division=0)),
        "auc": float(roc_auc_score(targets, probs)) if len(np.unique(targets)) > 1 else 0.5,
        "cm": confusion_matrix(targets, preds).tolist(),
        "preds": preds, "probs": probs, "targets": targets,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-csv", type=str,
                    default=str(PROJECT_ROOT / "data/splits/train_v5_weakfix.csv"))
    ap.add_argument("--val-csv", type=str,
                    default=str(PROJECT_ROOT / "data/splits/val_v5_combined_universal_kaggle_boost.csv"))
    ap.add_argument("--test-csv", type=str,
                    default=os.getenv("TEST_CSV", str(DATA_ROOT / "zero_leakage_benchmark_fixed/test_balanced_fixed_zero_leakage.csv")))
    ap.add_argument("--init-ckpt", type=str,
                    default=str(PROJECT_ROOT / "experiments/checkpoints/exp05_v5_combined/best_model.pt"))
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--base-lr", type=float, default=2e-5)
    ap.add_argument("--head-lr", type=float, default=5e-4)
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--img-size", type=int, default=256)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tag", type=str, default="v1")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    print(f"Device: {device}")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{ts}_v5_weakfix_{args.tag}"
    run_dir = PROJECT_ROOT / "experiments" / "runs" / run_name
    for d in ["checkpoints", "logs", "metrics"]:
        (run_dir / d).mkdir(parents=True, exist_ok=True)
    ckpt_dir = PROJECT_ROOT / "experiments" / "checkpoints" / "exp05_v5_weakfix"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Data
    df_train = pd.read_csv(args.train_csv)
    df_val = pd.read_csv(args.val_csv)
    df_test = pd.read_csv(args.test_csv)
    print(f"Train: {len(df_train):,} ({int((df_train.label==0).sum()):,}R/{int((df_train.label==1).sum()):,}F)")
    print(f"Val: {len(df_val):,} | Test: {len(df_test):,}")

    train_ds = DeepfakeDataset(df_train, get_train_transforms(args.img_size))
    val_ds = DeepfakeDataset(df_val, get_eval_transforms(args.img_size))
    test_ds = DeepfakeDataset(df_test, get_eval_transforms(args.img_size))

    w = method_balanced_weights(df_train)
    num_real = int((df_train.label == 0).sum())
    sampler = WeightedRandomSampler(w, num_samples=2 * num_real, replacement=True)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler,
                              num_workers=args.workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size * 2, shuffle=False,
                            num_workers=args.workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size * 2, shuffle=False,
                             num_workers=args.workers, pin_memory=True)

    # Model init from v5 checkpoint
    model = build_dinov3_classifier(
        weights_path=str(PROJECT_ROOT / "models/dinov3_small/model.safetensors"),
        img_size=args.img_size, device=str(device))
    ck = torch.load(args.init_ckpt, map_location=device, weights_only=False)
    sd = ck["model_state_dict"]
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"Init from {args.init_ckpt}: missing={len(missing)} unexpected={len(unexpected)}")

    criterion = LabelSmoothingCrossEntropy(smoothing=0.05)
    param_groups = get_parameter_groups(model, args.base_lr, args.head_lr, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    ema = ModelEMA(model, decay=0.999)

    best_val_auc = 0.0
    best_epoch = 0
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_epoch(model, train_loader, criterion, optimizer, device, ema)
        val_res = evaluate(ema.module, val_loader, device)
        scheduler.step()
        print(f"[E{epoch}/{args.epochs}] loss={tr_loss:.4f} acc={tr_acc*100:.2f}% "
              f"| val_acc={val_res['acc']*100:.2f}% auc={val_res['auc']*100:.2f}% ({time.time()-t0:.0f}s)")
        if val_res["auc"] > best_val_auc:
            best_val_auc, best_epoch = val_res["auc"], epoch
            torch.save({"model_state_dict": ema.module.state_dict(), "epoch": epoch,
                        "best_val_auc": best_val_auc, "config": vars(args),
                        "timestamp": datetime.now().isoformat()},
                       ckpt_dir / "best_model.pt")
    print(f"Best val AUC {best_val_auc*100:.2f}% at epoch {best_epoch}")

    # Final eval on test
    best = torch.load(ckpt_dir / "best_model.pt", map_location=device, weights_only=False)
    model.load_state_dict(best["model_state_dict"])
    test_res = evaluate(model, test_loader, device)
    print(f"\nTEST: acc={test_res['acc']*100:.2f}% auc={test_res['auc']*100:.2f}% "
          f"prec={test_res['prec']*100:.2f}% rec={test_res['rec']*100:.2f}% cm={test_res['cm']}")

    df_test["pred"] = test_res["preds"]
    df_test["prob"] = test_res["probs"]
    method_perf = []
    for m in df_test["method"].unique():
        sub = df_test[df_test["method"] == m]
        method_perf.append({"Method": m,
                            "Label": "REAL" if sub["label"].iloc[0] == 0 else "FAKE",
                            "Samples": len(sub),
                            "Accuracy": round(float((sub["pred"] == sub["label"]).mean()) * 100, 2),
                            "Mean Fake Probability": round(float(sub["prob"].mean()), 4)})
    df_method = pd.DataFrame(method_perf).sort_values(["Label", "Accuracy"], ascending=[True, False])
    print(df_method.to_string(index=False))

    # Compare vs baseline per-method accuracy
    baseline = pd.read_csv(PROJECT_ROOT / "experiments/results/v5_combined_per_method_accuracy.csv")
    merged = df_method.merge(baseline[["Method", "Accuracy"]], on="Method", suffixes=("_weakfix", "_baseline"))
    merged["delta"] = merged["Accuracy_weakfix"] - merged["Accuracy_baseline"]
    merged = merged.sort_values("Accuracy_baseline")
    print("\n=== DELTA vs BASELINE (worst baseline methods first) ===")
    print(merged.to_string(index=False))

    report = {
        "timestamp": datetime.now().isoformat(),
        "run_name": run_name,
        "config": vars(args),
        "best_epoch": best_epoch,
        "best_val_auc": float(best_val_auc),
        "test_metrics": {k: test_res[k] for k in ["acc", "auc", "precision", "recall", "f1", "cm"]},
        "method_breakdown": df_method.to_dict("records"),
    }
    rpath = PROJECT_ROOT / "experiments/results/v5_weakfix_training_report.json"
    rpath.write_text(json.dumps(report, indent=2))
    df_method.to_csv(PROJECT_ROOT / "experiments/results/v5_weakfix_per_method_accuracy.csv", index=False)
    merged.to_csv(PROJECT_ROOT / "experiments/results/v5_weakfix_vs_baseline.csv", index=False)
    print(f"\nSaved report -> {rpath}")


if __name__ == "__main__":
    main()
