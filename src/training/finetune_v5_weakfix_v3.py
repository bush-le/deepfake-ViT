"""
v3: faceswap-focused finetune on top of v2 (exp05_v5_weakfix).

Goal: faceswap is the last remaining weak method (62.96% in v2, same as baseline).
Diagnosis: the method-balanced sampler gave faceswap only ~1.4% of draws/epoch, so
the model barely learned it (mean fake-prob on DF40 faceswap train frames 0.47).

Changes vs v2:
  - Train csv: train_v5_weakfix_v3.csv (same as v2 + 8K extra identity-disjoint
    DF40 faceswap frames -> faceswap group ~12.5K rows).
  - Init: from exp05_v5_weakfix/best_model.pt (v2) instead of the raw v5 ckpt.
  - Sampler: faceswap-focused. P(real)=0.35, P(faceswap group)=0.35, P(other fake
    methods)=0.30 split uniformly. -> faceswap gets ~10x the attention of v2.
  - 3 epochs, batch 64, base-lr 1.5e-5 / head-lr 4e-4 (gentler to avoid forgetting
    the 40 methods fixed in v2).
  - Saves to experiments/checkpoints/exp05_v5_weakfix_v3/ (v2 ckpt untouched).
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

PROJECT_ROOT = Path(__file__).resolve().parents[2]
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


def faceswap_focused_weights(df, p_real=0.35, p_faceswap=0.35):
    """Weights: P(real)=p_real; P(faceswap group)=p_faceswap; rest split uniformly
    across all other fake methods. faceswap group = every row with method=='faceswap'."""
    labels = df["label"].values
    methods = df["method"].values
    num_real = int((labels == 0).sum())
    p_other = 1.0 - p_real - p_faceswap
    w = np.zeros(len(df), dtype=np.float64)

    w[labels == 0] = p_real / num_real

    is_fs = (labels == 1) & (methods == "faceswap")
    n_fs = int(is_fs.sum())
    w[is_fs] = p_faceswap / n_fs

    other = (labels == 1) & (~is_fs)
    other_methods = sorted(set(methods[other]))
    per_m = p_other / max(1, len(other_methods))
    for m in other_methods:
        idx = (labels == 1) & (methods == m) & (~is_fs)
        w[idx] = per_m / int(idx.sum())

    w /= w.sum()
    return w


def get_parameter_groups(model, base_lr=1.5e-5, head_lr=4e-4, weight_decay=0.05):
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
                    default=str(PROJECT_ROOT / "data/splits/train_v5_weakfix_v3.csv"))
    ap.add_argument("--val-csv", type=str,
                    default=str(PROJECT_ROOT / "data/splits/val_v5_combined_universal_kaggle_boost.csv"))
    ap.add_argument("--test-csv", type=str,
                    default=str(DATA_ROOT / "zero_leakage_benchmark_fixed/test_balanced_fixed_zero_leakage.csv"))
    ap.add_argument("--init-ckpt", type=str,
                    default=str(PROJECT_ROOT / "experiments/checkpoints/exp05_v5_weakfix/best_model.pt"))
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--base-lr", type=float, default=1.5e-5)
    ap.add_argument("--head-lr", type=float, default=4e-4)
    ap.add_argument("--p-faceswap", type=float, default=0.35)
    ap.add_argument("--p-real", type=float, default=0.35)
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--img-size", type=int, default=256)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tag", type=str, default="v1")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    print(f"Device: {device} | faceswap focus: P(fs)={args.p_faceswap} P(real)={args.p_real}")

    ckpt_dir = PROJECT_ROOT / "experiments" / "checkpoints" / "exp05_v5_weakfix_v3"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    df_train = pd.read_csv(args.train_csv)
    df_val = pd.read_csv(args.val_csv)
    df_test = pd.read_csv(args.test_csv)
    n_fs = int((df_train["method"] == "faceswap").sum())
    print(f"Train: {len(df_train):,} ({int((df_train.label==0).sum()):,}R/{int((df_train.label==1).sum()):,}F) | faceswap rows: {n_fs:,}")
    print(f"Val: {len(df_val):,} | Test: {len(df_test):,}")

    train_ds = DeepfakeDataset(df_train, get_train_transforms(args.img_size))
    val_ds = DeepfakeDataset(df_val, get_eval_transforms(args.img_size))
    test_ds = DeepfakeDataset(df_test, get_eval_transforms(args.img_size))

    w = faceswap_focused_weights(df_train, p_real=args.p_real, p_faceswap=args.p_faceswap)
    num_real = int((df_train.label == 0).sum())
    sampler = WeightedRandomSampler(w, num_samples=2 * num_real, replacement=True)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler,
                              num_workers=args.workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size * 2, shuffle=False,
                            num_workers=args.workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size * 2, shuffle=False,
                             num_workers=args.workers, pin_memory=True)

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

    baseline = pd.read_csv(PROJECT_ROOT / "experiments/results/v5_combined_per_method_accuracy.csv")
    merged = df_method.merge(baseline[["Method", "Accuracy"]], on="Method", suffixes=("_weakfix_v3", "_baseline"))
    merged["delta"] = merged["Accuracy_weakfix_v3"] - merged["Accuracy_baseline"]
    merged = merged.sort_values("Accuracy_baseline")
    print("\n=== DELTA vs BASELINE (worst baseline methods first) ===")
    print(merged.to_string(index=False))

    report = {
        "timestamp": datetime.now().isoformat(),
        "config": vars(args),
        "best_epoch": best_epoch,
        "best_val_auc": float(best_val_auc),
        "test_metrics": {k: test_res[k] for k in ["acc", "auc", "prec", "rec", "f1", "cm"]},
        "method_breakdown": df_method.to_dict("records"),
    }
    rpath = PROJECT_ROOT / "experiments/results/v5_weakfix_v3_training_report.json"
    rpath.write_text(json.dumps(report, indent=2))
    df_method.to_csv(PROJECT_ROOT / "experiments/results/v5_weakfix_v3_per_method_accuracy.csv", index=False)
    merged.to_csv(PROJECT_ROOT / "experiments/results/v5_weakfix_v3_vs_baseline.csv", index=False)
    print(f"\nSaved report -> {rpath}")


if __name__ == "__main__":
    main()
