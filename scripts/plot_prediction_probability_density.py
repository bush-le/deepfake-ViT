#!/usr/bin/env python3
"""
Prediction Probability Density Visualization Script
Models benchmarked:
  - Vision Transformer (ViT): Meta DINOv3 ViT-Small/16 Plus ('plus_v3_s1_best.pt', SwiGLU Gated MLP)
  - Convolutional Neural Network (CNN): Meta DINOv3 ConvNeXt-Tiny ('convnext_weakfix_v3.pt')

This script generates publication-quality Prediction Probability Density plots
(both Kernel Density Estimation KDE and Normalized Histograms) for both models.
"""

import os
import sys
import argparse
import importlib.util
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import gaussian_kde
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    average_precision_score,
)

# Root directory of deepfake-ViT repository
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# Color palette
COLOR_REAL = "#27ae60"       # Vibrant Green for Authentic / Real (Label 0)
COLOR_FAKE_CNN = "#e74c3c"   # Red for Fake on CNN (Label 1)
COLOR_FAKE_VIT = "#8e44ad"   # Purple for Fake on ViT (Label 1)
COLOR_THRESHOLD = "#2c3e50"  # Dark Slate for Decision Threshold (tau = 0.5)


def load_predictions(npz_path: Path):
    """Load cached predictions from npz archive."""
    if not npz_path.exists():
        return None
    data = np.load(npz_path)
    return {
        "preds": data["preds"],
        "probs": data["probs"],
        "labels": data["labels"]
    }


def compute_model_statistics(labels: np.ndarray, probs: np.ndarray, preds: np.ndarray = None):
    """Compute detailed diagnostic metrics for a model's prediction probabilities."""
    if preds is None:
        preds = (probs >= 0.5).astype(int)

    acc = accuracy_score(labels, preds) * 100
    auc = roc_auc_score(labels, probs) * 100
    ap = average_precision_score(labels, probs) * 100
    f1 = f1_score(labels, preds, pos_label=1, zero_division=0) * 100
    prec = precision_score(labels, preds, pos_label=1, zero_division=0) * 100
    rec = recall_score(labels, preds, pos_label=1, zero_division=0) * 100
    
    cm = confusion_matrix(labels, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    spec = (tn / (tn + fp)) * 100 if (tn + fp) > 0 else 0

    real_probs = probs[labels == 0]
    fake_probs = probs[labels == 1]

    # Separation Margin
    margin = fake_probs.mean() - real_probs.mean()
    
    # Ambiguity rate: percentage of samples in [0.3, 0.7]
    ambiguity = np.mean((probs >= 0.3) & (probs <= 0.7)) * 100

    return {
        "acc": acc,
        "auc": auc,
        "ap": ap,
        "f1": f1,
        "prec": prec,
        "rec": rec,
        "spec": spec,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "real_mean": real_probs.mean(),
        "real_std": real_probs.std(),
        "fake_mean": fake_probs.mean(),
        "fake_std": fake_probs.std(),
        "margin": margin,
        "ambiguity": ambiguity
    }


def run_live_inference(model, rows, device="cpu", batch_size=32, num_workers=2):
    """Run live inference on test images using PyTorch."""
    import torch
    from torch.utils.data import Dataset, DataLoader
    from torchvision import transforms
    from PIL import Image

    eval_tf = transforms.Compose([
        transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    class DirectImgDS(Dataset):
        def __init__(self, data_rows, transform=None):
            self.data_rows = data_rows
            self.transform = transform

        def __len__(self):
            return len(self.data_rows)

        def __getitem__(self, idx):
            row = self.data_rows[idx]
            img_path = Path(row["path"])
            img = Image.open(img_path).convert("RGB")
            if self.transform:
                img = self.transform(img)
            label = int(row["label"])
            return img, label

    ds = DirectImgDS(rows, transform=eval_tf)
    dl = DataLoader(ds, batch_size=batch_size, num_workers=num_workers, shuffle=False)

    preds, probs, labels = [], [], []
    model.eval()
    with torch.inference_mode():
        for x, y in dl:
            x = x.to(device)
            logits = model(x)
            p = torch.softmax(logits.float(), dim=1)
            preds.append(p.argmax(1).cpu().numpy())
            probs.append(p[:, 1].cpu().numpy())
            labels.append(y.numpy())

    return {
        "preds": np.concatenate(preds),
        "probs": np.concatenate(probs),
        "labels": np.concatenate(labels)
    }


def plot_kde_probability_density(
    labels: np.ndarray,
    probs_cnn: np.ndarray,
    probs_vit: np.ndarray,
    output_path: Path,
    dpi: int = 300,
    title_suffix: str = "(TEST BALANCED 20.8K)"
):
    """Generate high-resolution continuous KDE Prediction Probability Density plot."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=False)
    plt.subplots_adjust(wspace=0.20)

    x_grid = np.linspace(-0.05, 1.05, 600)

    # 1. CNN Subplot (ConvNeXt-Tiny / weakfix_v3)
    real_cnn = probs_cnn[labels == 0]
    fake_cnn = probs_cnn[labels == 1]
    kde_cnn_real = gaussian_kde(real_cnn)
    kde_cnn_fake = gaussian_kde(fake_cnn)

    axes[0].plot(x_grid, kde_cnn_real(x_grid), color=COLOR_REAL, lw=2.8, label=f"Real Domain (Mean = {real_cnn.mean():.3f})")
    axes[0].fill_between(x_grid, kde_cnn_real(x_grid), color=COLOR_REAL, alpha=0.35)
    axes[0].plot(x_grid, kde_cnn_fake(x_grid), color=COLOR_FAKE_CNN, lw=2.8, label=f"Fake Methods (Mean = {fake_cnn.mean():.3f})")
    axes[0].fill_between(x_grid, kde_cnn_fake(x_grid), color=COLOR_FAKE_CNN, alpha=0.35)
    axes[0].axvline(0.5, color=COLOR_THRESHOLD, linestyle="--", lw=2.0, label=r"Decision Threshold ($\tau = 0.50$)")

    axes[0].set_title("CNN (ConvNeXt-Tiny / weakfix_v3) — KDE Density", fontsize=12.5, fontweight="bold", pad=12)
    axes[0].set_xlabel(r"Predicted Deepfake Probability $P(Y = 1)$", fontsize=12)
    axes[0].set_ylabel("Density", fontsize=12)
    axes[0].set_xlim(-0.05, 1.05)
    axes[0].set_ylim(bottom=0)
    axes[0].grid(True, linestyle="--", alpha=0.4)
    axes[0].legend(loc="upper center", fontsize=10.5, framealpha=0.92)

    # 2. ViT Subplot (DINOv3 ViT-Small/16 Plus / plus_v3_s1_best)
    real_vit = probs_vit[labels == 0]
    fake_vit = probs_vit[labels == 1]
    kde_vit_real = gaussian_kde(real_vit)
    kde_vit_fake = gaussian_kde(fake_vit)

    axes[1].plot(x_grid, kde_vit_real(x_grid), color=COLOR_REAL, lw=2.8, label=f"Real Domain (Mean = {real_vit.mean():.3f})")
    axes[1].fill_between(x_grid, kde_vit_real(x_grid), color=COLOR_REAL, alpha=0.35)
    axes[1].plot(x_grid, kde_vit_fake(x_grid), color=COLOR_FAKE_VIT, lw=2.8, label=f"Fake Methods (Mean = {fake_vit.mean():.3f})")
    axes[1].fill_between(x_grid, kde_vit_fake(x_grid), color=COLOR_FAKE_VIT, alpha=0.35)
    axes[1].axvline(0.5, color=COLOR_THRESHOLD, linestyle="--", lw=2.0, label=r"Decision Threshold ($\tau = 0.50$)")

    axes[1].set_title("ViT (DINOv3 ViT-S/16 Plus / s1_best) — KDE Density", fontsize=12.5, fontweight="bold", pad=12)
    axes[1].set_xlabel(r"Predicted Deepfake Probability $P(Y = 1)$", fontsize=12)
    axes[1].set_ylabel("Density", fontsize=12)
    axes[1].set_xlim(-0.05, 1.05)
    axes[1].set_ylim(bottom=0)
    axes[1].grid(True, linestyle="--", alpha=0.4)
    axes[1].legend(loc="upper center", fontsize=10.5, framealpha=0.92)

    plt.suptitle(
        f"CONFIDENCE MARGIN DENSITIES: CNN (weakfix_v3) vs. ViT (s1_best) {title_suffix}",
        fontsize=14.5,
        fontweight="bold",
        y=1.02
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f" Saved KDE probability density plot -> {output_path}")


def plot_histogram_probability_density(
    labels: np.ndarray,
    probs_cnn: np.ndarray,
    probs_vit: np.ndarray,
    output_path: Path,
    dpi: int = 300,
    title_suffix: str = "(TEST BALANCED 20.8K)"
):
    """Generate normalized histogram Prediction Probability Density plot."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=False)
    plt.subplots_adjust(wspace=0.20)

    bins = 35

    # 1. ViT Histogram Subplot
    real_vit = probs_vit[labels == 0]
    fake_vit = probs_vit[labels == 1]
    axes[0].hist(real_vit, bins=bins, range=(0, 1), density=True, alpha=0.60, color=COLOR_REAL, edgecolor="black", linewidth=0.8, label=f"Real (Label 0, n={len(real_vit):,})")
    axes[0].hist(fake_vit, bins=bins, range=(0, 1), density=True, alpha=0.60, color=COLOR_FAKE_VIT, edgecolor="black", linewidth=0.8, label=f"Fake (Label 1, n={len(fake_vit):,})")
    axes[0].axvline(0.5, color=COLOR_THRESHOLD, linestyle="--", lw=2.0, label=r"Decision Threshold ($\tau = 0.50$)")
    axes[0].set_title("ViT (DINOv3 ViT-S/16 Plus / s1_best): Probability Histogram", fontsize=12.5, fontweight="bold", pad=12)
    axes[0].set_xlabel(r"Predicted Probability $P(\mathrm{Fake})$", fontsize=12)
    axes[0].set_ylabel("Density", fontsize=12)
    axes[0].set_xlim(-0.02, 1.02)
    axes[0].grid(True, linestyle="--", alpha=0.4)
    axes[0].legend(loc="upper center", fontsize=10.5, framealpha=0.92)

    # 2. CNN Histogram Subplot
    real_cnn = probs_cnn[labels == 0]
    fake_cnn = probs_cnn[labels == 1]
    axes[1].hist(real_cnn, bins=bins, range=(0, 1), density=True, alpha=0.60, color=COLOR_REAL, edgecolor="black", linewidth=0.8, label=f"Real (Label 0, n={len(real_cnn):,})")
    axes[1].hist(fake_cnn, bins=bins, range=(0, 1), density=True, alpha=0.60, color=COLOR_FAKE_CNN, edgecolor="black", linewidth=0.8, label=f"Fake (Label 1, n={len(fake_cnn):,})")
    axes[1].axvline(0.5, color=COLOR_THRESHOLD, linestyle="--", lw=2.0, label=r"Decision Threshold ($\tau = 0.50$)")
    axes[1].set_title("CNN (ConvNeXt-Tiny / weakfix_v3): Probability Histogram", fontsize=12.5, fontweight="bold", pad=12)
    axes[1].set_xlabel(r"Predicted Probability $P(\mathrm{Fake})$", fontsize=12)
    axes[1].set_ylabel("Density", fontsize=12)
    axes[1].set_xlim(-0.02, 1.02)
    axes[1].grid(True, linestyle="--", alpha=0.4)
    axes[1].legend(loc="upper center", fontsize=10.5, framealpha=0.92)

    plt.suptitle(
        f"POST-TRAINING PROBABILITY DENSITY SEPARATION: ViT (s1_best) vs. CNN (weakfix_v3) {title_suffix}",
        fontsize=14.5,
        fontweight="bold",
        y=1.02
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f" Saved Histogram density plot -> {output_path}")


def plot_combined_panel(
    labels: np.ndarray,
    probs_cnn: np.ndarray,
    probs_vit: np.ndarray,
    stats_cnn: dict,
    stats_vit: dict,
    output_path: Path,
    dpi: int = 300,
    title_suffix: str = "(TEST BALANCED 20.8K)"
):
    """Generate 2x2 combined diagnostic panel with KDE, Histograms, and quantitative summary box."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 11), sharey=False)
    plt.subplots_adjust(wspace=0.18, hspace=0.28)

    x_grid = np.linspace(-0.05, 1.05, 600)
    bins = 35

    # Row 0, Col 0: CNN KDE
    real_cnn = probs_cnn[labels == 0]
    fake_cnn = probs_cnn[labels == 1]
    kde_cnn_real = gaussian_kde(real_cnn)
    kde_cnn_fake = gaussian_kde(fake_cnn)
    axes[0, 0].plot(x_grid, kde_cnn_real(x_grid), color=COLOR_REAL, lw=2.5, label="Real (Label 0)")
    axes[0, 0].fill_between(x_grid, kde_cnn_real(x_grid), color=COLOR_REAL, alpha=0.35)
    axes[0, 0].plot(x_grid, kde_cnn_fake(x_grid), color=COLOR_FAKE_CNN, lw=2.5, label="Fake (Label 1)")
    axes[0, 0].fill_between(x_grid, kde_cnn_fake(x_grid), color=COLOR_FAKE_CNN, alpha=0.35)
    axes[0, 0].axvline(0.5, color=COLOR_THRESHOLD, linestyle="--", lw=1.8, label=r"Threshold $\tau=0.5$")
    axes[0, 0].set_title(r"$\mathbf{(A)}$ CNN (ConvNeXt-Tiny / weakfix_v3) — Continuous KDE", fontsize=12, fontweight="bold")
    axes[0, 0].set_xlabel(r"Predicted Probability $P(Y = 1)$", fontsize=11)
    axes[0, 0].set_ylabel("Density", fontsize=11)
    axes[0, 0].grid(True, linestyle="--", alpha=0.35)
    axes[0, 0].legend(fontsize=9.5, loc="upper center")

    # Annotation box for CNN
    cnn_text = (
        f"CNN Metrics:\n"
        f"• Accuracy: {stats_cnn['acc']:.2f}%\n"
        f"• ROC-AUC: {stats_cnn['auc']:.2f}%\n"
        f"• Real Mean: {stats_cnn['real_mean']:.3f}\n"
        f"• Fake Mean: {stats_cnn['fake_mean']:.3f}\n"
        f"• Ambiguity [0.3-0.7]: {stats_cnn['ambiguity']:.2f}%"
    )
    axes[0, 0].text(
        0.50, 0.45, cnn_text, transform=axes[0, 0].transAxes,
        fontsize=9.5, verticalalignment="center", horizontalalignment="center",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#ffffff", edgecolor="#bdc3c7", alpha=0.92)
    )

    # Row 0, Col 1: ViT KDE
    real_vit = probs_vit[labels == 0]
    fake_vit = probs_vit[labels == 1]
    kde_vit_real = gaussian_kde(real_vit)
    kde_vit_fake = gaussian_kde(fake_vit)
    axes[0, 1].plot(x_grid, kde_vit_real(x_grid), color=COLOR_REAL, lw=2.5, label="Real (Label 0)")
    axes[0, 1].fill_between(x_grid, kde_vit_real(x_grid), color=COLOR_REAL, alpha=0.35)
    axes[0, 1].plot(x_grid, kde_vit_fake(x_grid), color=COLOR_FAKE_VIT, lw=2.5, label="Fake (Label 1)")
    axes[0, 1].fill_between(x_grid, kde_vit_fake(x_grid), color=COLOR_FAKE_VIT, alpha=0.35)
    axes[0, 1].axvline(0.5, color=COLOR_THRESHOLD, linestyle="--", lw=1.8, label=r"Threshold $\tau=0.5$")
    axes[0, 1].set_title(r"$\mathbf{(B)}$ ViT (DINOv3 ViT-S/16 Plus / s1_best) — Continuous KDE", fontsize=12, fontweight="bold")
    axes[0, 1].set_xlabel(r"Predicted Probability $P(Y = 1)$", fontsize=11)
    axes[0, 1].set_ylabel("Density", fontsize=11)
    axes[0, 1].grid(True, linestyle="--", alpha=0.35)
    axes[0, 1].legend(fontsize=9.5, loc="upper center")

    # Annotation box for ViT
    vit_text = (
        f"ViT Metrics:\n"
        f"• Accuracy: {stats_vit['acc']:.2f}%\n"
        f"• ROC-AUC: {stats_vit['auc']:.2f}%\n"
        f"• Real Mean: {stats_vit['real_mean']:.3f}\n"
        f"• Fake Mean: {stats_vit['fake_mean']:.3f}\n"
        f"• Ambiguity [0.3-0.7]: {stats_vit['ambiguity']:.2f}%"
    )
    axes[0, 1].text(
        0.50, 0.45, vit_text, transform=axes[0, 1].transAxes,
        fontsize=9.5, verticalalignment="center", horizontalalignment="center",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#ffffff", edgecolor="#bdc3c7", alpha=0.92)
    )

    # Row 1, Col 0: CNN Histogram
    axes[1, 0].hist(real_cnn, bins=bins, range=(0, 1), density=True, alpha=0.60, color=COLOR_REAL, edgecolor="black", linewidth=0.7, label=f"Real (n={len(real_cnn):,})")
    axes[1, 0].hist(fake_cnn, bins=bins, range=(0, 1), density=True, alpha=0.60, color=COLOR_FAKE_CNN, edgecolor="black", linewidth=0.7, label=f"Fake (n={len(fake_cnn):,})")
    axes[1, 0].axvline(0.5, color=COLOR_THRESHOLD, linestyle="--", lw=1.8, label=r"Threshold $\tau=0.5$")
    axes[1, 0].set_title(r"$\mathbf{(C)}$ CNN Histogram Density Distribution", fontsize=12, fontweight="bold")
    axes[1, 0].set_xlabel(r"Predicted Probability $P(\mathrm{Fake})$", fontsize=11)
    axes[1, 0].set_ylabel("Density", fontsize=11)
    axes[1, 0].grid(True, linestyle="--", alpha=0.35)
    axes[1, 0].legend(fontsize=9.5, loc="upper center")

    # Row 1, Col 1: ViT Histogram
    axes[1, 1].hist(real_vit, bins=bins, range=(0, 1), density=True, alpha=0.60, color=COLOR_REAL, edgecolor="black", linewidth=0.7, label=f"Real (n={len(real_vit):,})")
    axes[1, 1].hist(fake_vit, bins=bins, range=(0, 1), density=True, alpha=0.60, color=COLOR_FAKE_VIT, edgecolor="black", linewidth=0.7, label=f"Fake (n={len(fake_vit):,})")
    axes[1, 1].axvline(0.5, color=COLOR_THRESHOLD, linestyle="--", lw=1.8, label=r"Threshold $\tau=0.5$")
    axes[1, 1].set_title(r"$\mathbf{(D)}$ ViT Histogram Density Distribution", fontsize=12, fontweight="bold")
    axes[1, 1].set_xlabel(r"Predicted Probability $P(\mathrm{Fake})$", fontsize=11)
    axes[1, 1].set_ylabel("Density", fontsize=11)
    axes[1, 1].grid(True, linestyle="--", alpha=0.35)
    axes[1, 1].legend(fontsize=9.5, loc="upper center")

    plt.suptitle(
        f"COMPREHENSIVE PROBABILITY DENSITY BENCHMARK: CNN (weakfix_v3) vs. ViT (s1_best) {title_suffix}",
        fontsize=15,
        fontweight="bold",
        y=0.995
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f" Saved Combined 2x2 density panel -> {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate Prediction Probability Density plots for ViT (plus_v3_s1_best) and CNN (convnext_weakfix_v3)."
    )
    parser.add_argument(
        "--vit-ckpt",
        type=str,
        default=str(REPO_ROOT / "experiments/checkpoints/plus_v3_s1_best.pt"),
        help="Path to ViT model checkpoint (.pt)"
    )
    parser.add_argument(
        "--cnn-ckpt",
        type=str,
        default=str(REPO_ROOT / "experiments/checkpoints/convnext_weakfix_v3.pt"),
        help="Path to CNN model checkpoint (.pt)"
    )
    parser.add_argument(
        "--vit-npz",
        type=str,
        default=str(REPO_ROOT / "experiments/results/courseWorkCheck/vit_test_bal.npz"),
        help="Path to cached ViT predictions npz file"
    )
    parser.add_argument(
        "--cnn-npz",
        type=str,
        default=str(REPO_ROOT / "experiments/results/courseWorkCheck/cnn_test_bal.npz"),
        help="Path to cached CNN predictions npz file"
    )
    parser.add_argument(
        "--test-csv",
        type=str,
        default=str(REPO_ROOT / "data/splits/test_coursework_44methods_balanced_zero_leakage.csv"),
        help="Path to test CSV split if re-running inference"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(REPO_ROOT / "experiments/plots"),
        help="Directory to save generated density figures"
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="prediction_probability_density_s1_best_vs_weakfix_v3",
        help="Filename prefix for generated plots"
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Figure DPI resolution (default: 300)"
    )
    parser.add_argument(
        "--force-inference",
        action="store_true",
        help="Force direct PyTorch model inference instead of using cached npz files"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu",
        help="Device for inference (cuda / mps / cpu)"
    )

    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    vit_npz_path = Path(args.vit_npz)
    cnn_npz_path = Path(args.cnn_npz)

    vit_data = None
    cnn_data = None

    if not args.force_inference:
        vit_data = load_predictions(vit_npz_path)
        cnn_data = load_predictions(cnn_npz_path)

    if vit_data is None or cnn_data is None or args.force_inference:
        print(" Loading models and performing live inference...")
        import torch
        import pandas as pd

        test_csv_path = Path(args.test_csv)
        if not test_csv_path.exists():
            raise FileNotFoundError(f"Test CSV not found at {test_csv_path}")

        df = pd.read_csv(test_csv_path)
        rows = df.to_dict("records")
        print(f" Loaded test dataset: {len(rows):,} samples from {test_csv_path.name}")

        # Load ViT Model
        spec_vit = importlib.util.spec_from_file_location("dinov3_vit", REPO_ROOT / "src/models/dinov3_vit.py")
        mod_vit = importlib.util.module_from_spec(spec_vit)
        spec_vit.loader.exec_module(mod_vit)

        ck_vit = torch.load(args.vit_ckpt, map_location="cpu", weights_only=False)
        state_dict_vit = ck_vit.get("state_dict", ck_vit.get("model_state_dict", ck_vit))
        is_gated = any("gate_proj" in k for k in state_dict_vit.keys())
        backbone_vit = mod_vit.DinoViT(img_size=256, gated_mlp=is_gated)
        model_vit = mod_vit.DinoViTClassifier(backbone=backbone_vit, num_classes=2)
        model_vit.load_state_dict(state_dict_vit, strict=True)
        model_vit = model_vit.to(args.device)
        print(f"✅ Loaded ViT (plus_v3_s1_best) from {args.vit_ckpt}")

        # Load CNN Model
        spec_cnn = importlib.util.spec_from_file_location("dinov3_convnext", REPO_ROOT / "src/models/dinov3_convnext.py")
        mod_cnn = importlib.util.module_from_spec(spec_cnn)
        spec_cnn.loader.exec_module(mod_cnn)

        spec_cls = importlib.util.spec_from_file_location("classifier_v2", REPO_ROOT / "src/models/classifier_v2.py")
        mod_cls = importlib.util.module_from_spec(spec_cls)
        spec_cls.loader.exec_module(mod_cls)

        backbone_cnn = mod_cnn.DinoConvNext()
        model_cnn = mod_cls.DinoConvNextClassifier(backbone_cnn, num_classes=2, hidden_dim=384)
        ck_cnn = torch.load(args.cnn_ckpt, map_location="cpu", weights_only=False)
        state_dict_cnn = ck_cnn.get("model_state_dict", ck_cnn.get("state_dict", ck_cnn))
        model_cnn.load_state_dict(state_dict_cnn, strict=False)
        model_cnn = model_cnn.to(args.device)
        print(f"✅ Loaded CNN (convnext_weakfix_v3) from {args.cnn_ckpt}")

        print(" Running ViT inference...")
        vit_data = run_live_inference(model_vit, rows, device=args.device)
        print(" Running CNN inference...")
        cnn_data = run_live_inference(model_cnn, rows, device=args.device)

    labels = vit_data["labels"]
    probs_vit = vit_data["probs"]
    probs_cnn = cnn_data["probs"]

    print("\n" + "=" * 70)
    print(" 📊 PREDICTION PROBABILITY DENSITY BENCHMARK SUMMARY")
    print("=" * 70)
    print(f" Total Evaluation Samples: {len(labels):,} (Real: {(labels == 0).sum():,}, Fake: {(labels == 1).sum():,})")

    stats_vit = compute_model_statistics(labels, probs_vit, vit_data.get("preds"))
    stats_cnn = compute_model_statistics(labels, probs_cnn, cnn_data.get("preds"))

    print("\n--- ViT Model (DINOv3 ViT-Small/16 Plus / plus_v3_s1_best.pt) ---")
    print(f"  • Test Accuracy:   {stats_vit['acc']:.2f}%")
    print(f"  • ROC-AUC:         {stats_vit['auc']:.2f}%")
    print(f"  • Average Prec.:   {stats_vit['ap']:.2f}%")
    print(f"  • Specificity:     {stats_vit['spec']:.2f}% (Real Domain Accuracy)")
    print(f"  • Sensitivity:     {stats_vit['rec']:.2f}% (Deepfake Catch Rate)")
    print(f"  • Real Probs:      Mean = {stats_vit['real_mean']:.4f} ± {stats_vit['real_std']:.4f}")
    print(f"  • Fake Probs:      Mean = {stats_vit['fake_mean']:.4f} ± {stats_vit['fake_std']:.4f}")
    print(f"  • Margin:          {stats_vit['margin']:.4f}")
    print(f"  • Ambiguity Zone:  {stats_vit['ambiguity']:.2f}% in [0.3, 0.7]")

    print("\n--- CNN Model (Meta DINOv3 ConvNeXt-Tiny / convnext_weakfix_v3.pt) ---")
    print(f"  • Test Accuracy:   {stats_cnn['acc']:.2f}%")
    print(f"  • ROC-AUC:         {stats_cnn['auc']:.2f}%")
    print(f"  • Average Prec.:   {stats_cnn['ap']:.2f}%")
    print(f"  • Specificity:     {stats_cnn['spec']:.2f}% (Real Domain Accuracy)")
    print(f"  • Sensitivity:     {stats_cnn['rec']:.2f}% (Deepfake Catch Rate)")
    print(f"  • Real Probs:      Mean = {stats_cnn['real_mean']:.4f} ± {stats_cnn['real_std']:.4f}")
    print(f"  • Fake Probs:      Mean = {stats_cnn['fake_mean']:.4f} ± {stats_cnn['fake_std']:.4f}")
    print(f"  • Margin:          {stats_cnn['margin']:.4f}")
    print(f"  • Ambiguity Zone:  {stats_cnn['ambiguity']:.2f}% in [0.3, 0.7]")
    print("=" * 70 + "\n")

    # Generate Plots
    kde_plot_path = out_dir / f"{args.output_prefix}_kde.png"
    hist_plot_path = out_dir / f"{args.output_prefix}_hist.png"
    combined_plot_path = out_dir / f"{args.output_prefix}_combined.png"

    plot_kde_probability_density(labels, probs_cnn, probs_vit, kde_plot_path, dpi=args.dpi)
    plot_histogram_probability_density(labels, probs_cnn, probs_vit, hist_plot_path, dpi=args.dpi)
    plot_combined_panel(labels, probs_cnn, probs_vit, stats_cnn, stats_vit, combined_plot_path, dpi=args.dpi)

    print("\n🎯 All Prediction Probability Density plots generated successfully!")


if __name__ == "__main__":
    main()
