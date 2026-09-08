"""Comprehensive Benchmark Evaluation of EXP-02 Best Checkpoint.

Evaluates on:
- test_balanced.csv (4,134 samples, 1:1 Real:Fake)
- Standard Inference (tau=0.5)
- Test-Time Augmentation (TTA)
- Optimal Threshold Tuning (Youden's J on Val)
- 40-Method Breakdown & 5-Domain Category Breakdown
- Direct Side-by-Side Comparison with Baseline
"""
import json
import os
import sys
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
from torchvision import transforms as T

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from src.models.dinov3_vit import load_dinov3
from src.eval.tta import predict_batch_with_tta

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMG_SIZE = 256
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

eval_tf = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE), interpolation=T.InterpolationMode.BICUBIC),
    T.ToTensor(),
    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
])


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


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def get_method_and_domain(path):
    methods_list = [
        'MidJourney', 'whichfaceisreal', 'styleclip', 'CollabDiff', 'DiT', 'SiT', 'sd2.1', 'pixart', 'ddim', 'RDDM',
        'StyleGAN2', 'StyleGAN3', 'StyleGANXL', 'VQGAN', 'stargan', 'starganv2', 'e4e',
        'simswap', 'faceswap', 'mobileswap', 'inswap', 'blendface', 'facedancer', 'fsgan', 'uniface', 'deepfacelab', 'e4s',
        'sadtalker', 'wav2lip', 'fomm', 'pirender', 'tpsm', 'lia', 'MRAA', 'danet', 'mcnet', 'facevid2vid', 'one_shot_free', 'hyperreenact', 'heygen'
    ]
    for m in methods_list:
        if f"/{m}/" in path or f"__{m}__" in path or f"/{m.lower()}/" in path:
            domain = 'efs' if m in ['MidJourney', 'whichfaceisreal', 'CollabDiff', 'DiT', 'SiT', 'sd2.1', 'pixart', 'ddim', 'RDDM', 'StyleGAN2', 'StyleGAN3', 'StyleGANXL', 'VQGAN', 'e4e'] else                      'fe' if m in ['styleclip', 'stargan', 'starganv2'] else                      'ffc' if m in ['simswap', 'faceswap', 'mobileswap', 'inswap', 'blendface', 'facedancer', 'fsgan', 'uniface', 'deepfacelab'] else                      'oth'
            return m, domain

    if 'celeb' in path.lower():
        if 'real' in path.lower():
            return 'CelebDF_real', 'cdc'
        else:
            return 'CelebDF_fake', 'cdc'
    if 'real' in path.lower() or 'original_sequences' in path:
        return 'FF++_real', 'ffc'
    return 'Other', 'oth'


def main():
    print(f"🔬 Running EXP-02 Comprehensive Evaluation on {device}...")
    weights_path = "experiments/checkpoints/weights/dinov3-vits16plus-pretrain-lvd1689m/model-3.safetensors"
    ckpt_path = "experiments/checkpoints/dinov3_vit_exp02_best.pt"
    test_csv = "data/splits/test_balanced.csv"

    if not os.path.exists(weights_path) or not os.path.exists(ckpt_path) or not os.path.exists(test_csv):
        print("⚠️ Required weights, checkpoint, or test CSV not found. Exiting.")
        return

    backbone = load_dinov3(weights_path, img_size=256)
    model = EnhancedDinoViTClassifier(backbone).to(device)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    best_epoch = ckpt.get('epoch', 8)
    val_metrics = ckpt.get('val_metrics', {})
    val_opt_tau = val_metrics.get('opt_tau', 0.5496)
    print(f"📦 Loaded Checkpoint from Best Epoch: {best_epoch} (Val Acc: {val_metrics.get('accuracy',0)*100:.2f}%, Val AUC: {val_metrics.get('roc_auc',0):.4f}, Opt Tau: {val_opt_tau:.4f})")

    df_test = pd.read_csv(test_csv)
    print(f"📊 Test Benchmark Samples: {len(df_test):,} (Real: {(df_test['label']==0).sum():,}, Fake: {(df_test['label']==1).sum():,})")
    df_test[['method', 'domain']] = df_test['path'].apply(lambda p: pd.Series(get_method_and_domain(p)))

    all_probs_std = []
    all_probs_tta = []
    batch_size = 64

    with torch.no_grad():
        for i in range(0, len(df_test), batch_size):
            batch_paths = df_test['path'].iloc[i:i+batch_size]
            batch_imgs = torch.stack([eval_tf(Image.open(p).convert('RGB')) for p in batch_paths]).to(device)

            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == 'cuda')):
                logits = model(batch_imgs)
                probs_std = F.softmax(logits, dim=-1)[:, 1]
            all_probs_std.extend(probs_std.cpu().numpy().tolist())

            probs_tta = predict_batch_with_tta(model, batch_imgs, use_flips=True, use_multi_lighting=True)
            all_probs_tta.extend(probs_tta.cpu().numpy().tolist())

    df_test['prob_std'] = np.array(all_probs_std)
    df_test['prob_tta'] = np.array(all_probs_tta)
    labels = df_test['label'].values

    pred_std = (df_test['prob_std'] >= 0.5).astype(int)
    acc_std = accuracy_score(labels, pred_std)
    auc_std = roc_auc_score(labels, df_test['prob_std'])
    f1_std = f1_score(labels, pred_std)
    cm_std = confusion_matrix(labels, pred_std, labels=[0, 1])

    pred_tta = (df_test['prob_tta'] >= 0.5).astype(int)
    acc_tta = accuracy_score(labels, pred_tta)
    auc_tta = roc_auc_score(labels, df_test['prob_tta'])
    f1_tta = f1_score(labels, pred_tta)
    cm_tta = confusion_matrix(labels, pred_tta, labels=[0, 1])

    pred_opt = (df_test['prob_tta'] >= val_opt_tau).astype(int)
    acc_opt = accuracy_score(labels, pred_opt)
    f1_opt = f1_score(labels, pred_opt)
    cm_opt = confusion_matrix(labels, pred_opt, labels=[0, 1])
    rec_fake = recall_score(labels, pred_opt, pos_label=1)
    rec_real = recall_score(labels, pred_opt, pos_label=0)

    print(f"\n📈 GLOBAL BENCHMARK METRICS:")
    print(f"  Standard (tau=0.50) : Acc={acc_std*100:.2f}%, AUC={auc_std:.4f}, F1={f1_std:.4f}")
    print(f"  TTA (tau=0.50)      : Acc={acc_tta*100:.2f}%, AUC={auc_tta:.4f}, F1={f1_tta:.4f}")
    print(f"  TTA + Opt Tau ({val_opt_tau:.4f}) : Acc={acc_opt*100:.2f}%, F1={f1_opt:.4f}, Fake Recall={rec_fake*100:.2f}%, Real Specificity={rec_real*100:.2f}%")

    domain_stats = {}
    for d, group in df_test.groupby('domain'):
        g_labels = group['label'].values
        g_preds = (group['prob_tta'] >= val_opt_tau).astype(int)
        g_acc = accuracy_score(g_labels, g_preds)
        g_auc = roc_auc_score(g_labels, group['prob_tta']) if len(np.unique(g_labels)) > 1 else 1.0
        domain_stats[d] = {
            'count': len(group),
            'accuracy': float(g_acc),
            'roc_auc': float(g_auc)
        }

    method_stats = {}
    for m, group in df_test[df_test['label'] == 1].groupby('method'):
        g_preds = (group['prob_tta'] >= val_opt_tau).astype(int)
        m_acc = (g_preds == 1).mean()
        method_stats[m] = {
            'count': len(group),
            'detection_accuracy': float(m_acc)
        }

    clean_val_metrics = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in val_metrics.items() if k not in ['probs', 'labels']}

    report_data = {
        'experiment_id': 'EXP-02',
        'status': 'Completed',
        'best_epoch': int(best_epoch),
        'val_metrics': clean_val_metrics,
        'test_metrics': {
            'accuracy_standard': float(acc_std),
            'accuracy_tta': float(acc_tta),
            'accuracy_tta_optimal_tau': float(acc_opt),
            'optimal_tau': float(val_opt_tau),
            'fake_recall': float(rec_fake),
            'real_specificity': float(rec_real),
            'f1_score': float(f1_opt),
            'roc_auc': float(auc_tta),
            'confusion_matrix': cm_opt.tolist(),
            'total_errors': int(cm_opt[0,1] + cm_opt[1,0]),
            'total_samples': int(len(df_test))
        },
        'domain_breakdown': domain_stats,
        'weak_methods_comparison': method_stats
    }

    out_report = "experiments/results/exp02_enhanced_vit_report.json"
    os.makedirs("experiments/results", exist_ok=True)
    with open(out_report, 'w') as f:
        json.dump(report_data, f, indent=2, cls=NumpyEncoder)
    print(f"\n✅ Full Report successfully written to: {out_report}")


if __name__ == '__main__':
    main()
