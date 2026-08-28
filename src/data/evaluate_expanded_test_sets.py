"""
Script to compute predictions on both new test sets and old test set,
saving all cached prediction NPZ files and per-method accuracy CSV files.
"""

import os, sys, csv, time, json, importlib.util
from pathlib import Path
from collections import Counter, defaultdict

import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
ROOT = Path(__file__).resolve().parent.parent.parent
HT = ROOT if ROOT.exists() else Path('/workspace/hoangtuan/deepfake-ViT')
DATA_DIR = Path('/workspace/data/zero_leakage_benchmark_fixed')

V3_CKPT = HT / 'experiments/checkpoints/best_model_v3.pt' if (HT / 'experiments/checkpoints/best_model_v3.pt').exists() else HT / 'experiments/checkpoints/exp05_v5_weakfix_v3/best_model.pt'
PRETRAINED = HT / 'models/dinov3_small/model.safetensors'
OUT = HT / 'experiments/results/courseWorkCheck'
OUT.mkdir(parents=True, exist_ok=True)

# Model setup
spec = importlib.util.spec_from_file_location('ht_dinov3', HT / 'src/models/dinov3_vit.py')
ht = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ht)

model = ht.build_dinov3_classifier(weights_path=str(PRETRAINED), num_classes=2, img_size=256, device=DEVICE)
ck = torch.load(str(V3_CKPT), map_location='cpu', weights_only=False)
model.load_state_dict(ck['model_state_dict'], strict=False)
model.to(DEVICE)
model.eval()

IMG_SIZE = 256
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
tf = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE), interpolation=T.InterpolationMode.BICUBIC),
    T.ToTensor(), T.Normalize(MEAN, STD),
])

class ImgDS(Dataset):
    def __init__(self, rows, tf):
        self.rows, self.tf = rows, tf
    def __len__(self):
        return len(self.rows)
    def __getitem__(self, i):
        im = Image.open(self.rows[i]['path']).convert('RGB')
        return self.tf(im), int(self.rows[i]['label'])

def evaluate_and_cache(csv_path, npz_out_path, name):
    print(f"\n{'='*70}")
    print(f"📊 EVALUATING: {name}")
    print(f"   CSV: {csv_path}")
    print(f"{'='*70}")
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
        
    dl = DataLoader(ImgDS(rows, tf), batch_size=128, num_workers=8, pin_memory=True)
    preds, probs = [], []
    t0 = time.time()
    with torch.no_grad():
        for x, _ in dl:
            x = x.to(DEVICE, non_blocking=True)
            with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=(DEVICE == 'cuda')):
                logits = model(x)
            p = torch.softmax(logits.float(), dim=1)
            preds.append(p.argmax(1).cpu().numpy())
            probs.append(p[:, 1].cpu().numpy())
            
    preds = np.concatenate(preds)
    probs = np.concatenate(probs)
    labels = np.array([int(r['label']) for r in rows])
    
    np.savez_compressed(npz_out_path, preds=preds, probs=probs, labels=labels)
    print(f"Saved predictions cache -> {npz_out_path} ({time.time()-t0:.1f}s)")
    
    acc = accuracy_score(labels, preds)
    auc = roc_auc_score(labels, probs)
    prec = precision_score(labels, preds, zero_division=0)
    rec = recall_score(labels, preds, zero_division=0)
    f1 = f1_score(labels, preds, zero_division=0)
    cm = confusion_matrix(labels, preds, labels=[0, 1])
    
    print(f"Results for {name}:")
    print(f"  Accuracy : {acc*100:.2f}%")
    print(f"  ROC-AUC  : {auc*100:.2f}%")
    print(f"  Precision: {prec*100:.2f}%")
    print(f"  Recall   : {rec*100:.2f}%")
    print(f"  F1 Score : {f1*100:.2f}%")
    print(f"  Confusion Matrix:\n{cm}")
    
    # Per method metrics
    methods = [r.get('method', 'unknown') for r in rows]
    method_metrics = []
    for m in sorted(set(methods)):
        idx = [i for i, meth in enumerate(methods) if meth == m]
        m_labels = labels[idx]
        m_preds = preds[idx]
        m_acc = (m_labels == m_preds).mean()
        m_lbl = m_labels[0]
        method_metrics.append({
            'method': m,
            'label': int(m_lbl),
            'total': len(idx),
            'correct': int((m_labels == m_preds).sum()),
            'accuracy': float(m_acc)
        })
    
    return {
        'name': name,
        'samples': len(rows),
        'accuracy': acc,
        'roc_auc': auc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'confusion_matrix': cm.tolist(),
        'per_method': method_metrics
    }

if __name__ == '__main__':
    # 1. Balanced 25k
    bal_csv = DATA_DIR / 'test_expanded_44methods_balanced_zero_leakage.csv'
    bal_npz = OUT / 'v3_pred_expanded_balanced_25k.npz'
    res_bal = evaluate_and_cache(bal_csv, bal_npz, 'Expanded Balanced 1:1 Test Set (25,046 samples)')
    
    # 2. Full 59k
    full_csv = DATA_DIR / 'test_expanded_44methods_zero_leakage.csv'
    full_npz = OUT / 'v3_pred_expanded_full_59k.npz'
    res_full = evaluate_and_cache(full_csv, full_npz, 'Expanded Full Test Set (59,276 samples)')
    
    # Save combined report
    with open(OUT / 'expanded_eval_results_summary.json', 'w', encoding='utf-8') as f:
        json.dump({'balanced_25k': res_bal, 'full_59k': res_full}, f, indent=2, ensure_ascii=False)
        
    print("\n🎉 ALL TEST EVALUATIONS COMPLETED SUCCESSFULLY!")
