"""Shared training utilities and dataset handlers across training pipelines."""

import os
import torch
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

IMG_SIZE = 256
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_standard_train_transforms(img_size: int = IMG_SIZE) -> T.Compose:
    """Build standard domain-agnostic training data augmentation transform."""
    return T.Compose([
        T.Resize((img_size, img_size), interpolation=T.InterpolationMode.BICUBIC),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomApply([T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05)], p=0.5),
        T.RandomApply([T.GaussianBlur(kernel_size=(3, 5), sigma=(0.1, 2.0))], p=0.3),
        T.RandomApply([T.RandomAdjustSharpness(sharpness_factor=2.0)], p=0.3),
        T.RandomApply([T.RandomAutocontrast()], p=0.2),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_standard_eval_transforms(img_size: int = IMG_SIZE) -> T.Compose:
    """Build standard evaluation data transform (Resize + Normalize)."""
    return T.Compose([
        T.Resize((img_size, img_size), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


class StandardDeepfakeDataset(Dataset):
    """Standardized PyTorch dataset for deepfake detection reading from CSV metadata."""

    def __init__(self, csv_path: str, transform=None, max_samples: int = None, seed: int = 42):
        df = pd.read_csv(csv_path)
        if max_samples and len(df) > max_samples:
            df = df.sample(n=max_samples, random_state=seed).reset_index(drop=True)

        img_col = "image_path" if "image_path" in df.columns else ("path" if "path" in df.columns else df.columns[0])
        lbl_col = "label" if "label" in df.columns else ("true_label" if "true_label" in df.columns else df.columns[1])

        self.paths = df[img_col].values
        self.labels = df[lbl_col].values
        self.methods = df["method"].values if "method" in df.columns else np.array(["unknown"] * len(df))
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img_path = self.paths[idx]
        label = self.labels[idx]
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception:
            image = Image.new("RGB", (IMG_SIZE, IMG_SIZE), (0, 0, 0))

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)


def compute_classification_metrics(y_true, y_probs, threshold: float = 0.5) -> dict:
    """Compute binary classification metrics including Accuracy, AUC, F1, Precision, and Recall."""
    y_preds = (y_probs >= threshold).astype(int)

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_preds)),
        "f1": float(f1_score(y_true, y_preds, zero_division=0)),
        "precision": float(precision_score(y_true, y_preds, zero_division=0)),
        "recall": float(recall_score(y_true, y_preds, zero_division=0)),
    }

    try:
        metrics["auc"] = float(roc_auc_score(y_true, y_probs))
    except Exception:
        metrics["auc"] = 0.5

    return metrics
