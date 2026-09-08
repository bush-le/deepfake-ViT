# TECHNICAL_GUIDE.md — Current State, Model Architectures, Workflows & Benchmarking

- **Motivation/Background**: Provide operational guidelines, command references, and reusable technical workflows for TECHNICAL_GUIDE.
- **Purpose**: Standardize engineering practices and technical procedures for TECHNICAL_GUIDE.
- **Overview Pipeline**: Operational procedure formulation -> best practices curation -> reference guide compilation.
- **Detailed Plan**: §1 Overview & Prerequisites; §2 Procedural Guide; §3 Common Commands & Examples; §4 Troubleshooting & FAQs.
- **References**: `docs/shared/`, `agents/rules/`.
- **Created**: 2026-08-22T21:44:36+07:00
- **Last Updated**: 2026-09-08T10:15:00+07:00

---

  - [`experiments/checkpoints/best_model_v3.pt`](../experiments/checkpoints/best_model_v3.pt) (DINOv3 ViT-Small/16, 21.60M params)
  - [`experiments/checkpoints/convnext_weakfix_v3.pt`](../experiments/checkpoints/convnext_weakfix_v3.pt) (DINOv3 ConvNeXt-Tiny, 28.12M params)

---

## 📑 Table of Contents

1. [Current System State](#1-current-system-state)
2. [Model Architectures & Implementations](#2-model-architectures--implementations)
3. [Dataset Architecture & Zero-Leakage Splits](#3-dataset-architecture--zero-leakage-splits)
4. [Inference & Evaluation Workflows](#4-inference--evaluation-workflows)
5. [Training & Fine-Tuning Workflows](#5-training--fine-tuning-workflows)
6. [Interactive Notebooks & Execution Guides](#6-interactive-notebooks--execution-guides)
7. [Environment, Dependencies & Known Quirks](#7-environment-dependencies--known-quirks)
8. [Run Verification Checklist](#8-run-verification-checklist)

---

## 1. Current System State

- **Task**: Binary face deepfake classification (**Real = 0**, **Fake = 1**) at resolution $256 \times 256$.
- **Primary Model**: Meta DINOv3 ViT-Small/16 + Linear Head (`Linear(384, 2)`).
- **Baseline Model**: Meta DINOv3 ConvNeXt-Tiny + 2-layer GELU MLP Head.
- **Ensemble Model**: Joint inference `EnsembleClassifier` fusing ViT & CNN probabilities ($0.65 \cdot P_{\text{ViT}} + 0.35 \cdot P_{\text{CNN}}$).
- **Active Dataset**: 129,884 train samples (51 balanced subsets across 44 fake methods + 7 real sources); 6,000 validation samples; 21,446 balanced test samples; 50,084 full suite test samples.
- **Zero-Leakage Guarantee**: 100% strict identity, video, and MD5 byte-level disjointness between Train and Test splits.

---

## 2. Model Architectures & Implementations

### 2.1 Meta DINOv3 ViT-Small/16 (`src/models/dinov3_vit.py`)
- **Backbone**: ViT-Small with patch size 16 (`patch_size=16`, `img_size=256`, `embed_dim=384`, `depth=12`, `num_heads=6`, 4 register tokens, SwiGLU feed-forward).
- **Classification Head**: `Linear(384, 2)` directly operating on the extracted `[CLS]` token embedding.
- **Total Parameters**: **21,602,306** (~21.60M).
- **Loading Pattern**:
```python
import importlib.util, torch
from pathlib import Path

ROOT = Path(".")
spec = importlib.util.spec_from_file_location("ht_dinov3", ROOT / "src/models/dinov3_vit.py")
ht = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ht)

model_vit = ht.build_dinov3_classifier(weights_path=None, num_classes=2, img_size=256, device="cuda")
ck_vit = torch.load(ROOT / "experiments/checkpoints/best_model_v3.pt", map_location="cpu", weights_only=False)
model_vit.load_state_dict(ck_vit["model_state_dict"], strict=False)
model_vit.to("cuda").eval()
```

### 2.2 Meta DINOv3 ConvNeXt-Tiny (`src/models/dinov3_convnext.py` & `src/models/classifier_v2.py`)
- **Backbone**: ConvNeXt-Tiny with 4 stages (`depths=[3, 3, 9, 3]`, `dims=[96, 192, 384, 768]`, $7\times 7$ depthwise convolutions, LayerScale).
- **Classification Head**: `DinoConvNextClassifier` with 2-layer GELU MLP:
  $$\text{Input (768)} \rightarrow \text{LayerNorm} \rightarrow \text{Dropout(0.2)} \rightarrow \text{Linear(768, 384)} \rightarrow \text{GELU} \rightarrow \text{Dropout(0.1)} \rightarrow \text{Linear(384, 2)}$$
- **Total Parameters**: **28,117,730** (~28.12M).
- **Loading Pattern**:
```python
spec_cnn = importlib.util.spec_from_file_location("ht_convnext", ROOT / "src/models/dinov3_convnext.py")
cnn_mod = importlib.util.module_from_spec(spec_cnn)
spec_cnn.loader.exec_module(cnn_mod)

spec_cls = importlib.util.spec_from_file_location("classifier_v2", ROOT / "src/models/classifier_v2.py")
cls_mod = importlib.util.module_from_spec(spec_cls)
spec_cls.loader.exec_module(cls_mod)

backbone_cnn = cnn_mod.DinoConvNext()
model_cnn = cls_mod.DinoConvNextClassifier(backbone_cnn, num_classes=2, hidden_dim=384)
ck_cnn = torch.load(ROOT / "experiments/checkpoints/convnext_weakfix_v3.pt", map_location="cpu", weights_only=False)
model_cnn.load_state_dict(ck_cnn["model_state_dict"], strict=False)
model_cnn.to("cuda").eval()
```

---

## 3. Dataset Architecture & Zero-Leakage Splits

### 3.1 Active CSV Splits
- **Train (v3 Clean)**: [`data/splits/train_v5_weakfix_v3.csv`](../data/splits/train_v5_weakfix_v3.csv) (**129,884 samples** — 31,006 Real : 98,878 Fake across 51 subsets).
- **Validation**: [`data/splits/val_v5_combined_universal_kaggle_boost.csv`](../data/splits/val_v5_combined_universal_kaggle_boost.csv) (**6,000 samples** — 3,000 Real : 3,000 Fake).
- **Test CourseWork Balanced**: [`test_coursework_44methods_balanced_zero_leakage.csv`](../data/splits/test_coursework_44methods_balanced_zero_leakage.csv) (**21,446 samples** — 10,723 Real : 10,723 Fake, ~300 per method).
- **Test CourseWork Full Suite**: [`test_coursework_44methods_full_zero_leakage.csv`](../data/splits/test_coursework_44methods_full_zero_leakage.csv) (**50,084 samples** — 25,042 Real : 25,042 Fake).

---

## 4. Inference & Evaluation Workflows

```python
def run_evaluation(model, rows, tf, desc="Inference"):
    ds = ImgDS(rows, tf)
    dl = DataLoader(ds, batch_size=64, num_workers=0, pin_memory=True, shuffle=False)
    preds, probs = [], []
    with torch.no_grad():
        for x, _ in tqdm(dl, desc=desc):
            x = x.to(DEVICE, non_blocking=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits = model(x)
            p = torch.softmax(logits.float(), dim=1)
            preds.append(p.argmax(1).cpu().numpy())
            probs.append(p[:, 1].cpu().numpy())
    return np.concatenate(preds), np.concatenate(probs), np.array([int(x["label"]) for x in rows])
```

---

## 5. Training & Fine-Tuning Workflows

Training is executed exclusively through standalone scripts in `src/training/`:

```bash
# Fine-tune DINOv3 ViT or ConvNeXt with LLRD and balanced sampling
.venv/bin/python src/training/finetune_compare.py \
    --model vit \
    --train_csv data/splits/train_v5_weakfix_v3.csv \
    --val_csv data/splits/val_v5_combined_universal_kaggle_boost.csv \
    --epochs 5 \
    --batch_size 64 \
    --lr 1.5e-5 \
    --amp
```

---

## 6. Interactive Notebooks & Execution Guides

1. **[`notebooks/coursework_eda.ipynb`](../../notebooks/coursework_eda.ipynb)** (46 cells (20 sections) Master EDA Notebook):
   - Comprehensive multi-split census (207,414 samples), 54-methods taxonomy, 2D FFT power spectrum, 1D Radial PSD decay curves, High-Pass noise residuals, GLCM skin texture metrics, and Kolmogorov-Smirnov (KS-Test) statistical verification.
2. **[`notebooks/coursework_deepfake.ipynb`](../../notebooks/coursework_deepfake.ipynb)** (35 Cells Master Evaluation Notebook):
   - 3-tier zero-leakage auditing (0 overlaps, 127k MD5 byte hashes), live GPU benchmarking (ViT vs ConvNeXt vs Ensemble), 5 post-training visualization charts, Youden's J threshold optimization, and visual error gallery.
3. **[`notebooks/predict_image.ipynb`](../../notebooks/predict_image.ipynb)** (9 Cells Single-Image Predictor):
   - Interactive single-image test harness for inspecting prediction probabilities on any user image.

---

## 7. Environment, Dependencies & Known Quirks

- **Python Version**: Python 3.14 / 3.11+.
- **PyTorch**: 2.13.0+cu130 with CUDA acceleration.
- **Python 3.14+ Forkserver / Pickle Quirk**: In interactive Jupyter sessions on Linux, setting `num_workers > 0` in `DataLoader` can fail with `AttributeError: module '__main__' has no attribute 'ImgDS'` due to POSIX `forkserver`. **Solution:** Always use `num_workers = 0` in interactive notebooks.
- **GPU Memory Management**: On 4 GB VRAM GPUs (e.g., RTX 3050 Laptop), use `batch_size = 64` and invoke `torch.cuda.empty_cache()` when switching between model backbones.

---

## 8. Run Verification Checklist

Before running or delivering results, verify:
- [x] Checkpoint weights exist and load with zero missing non-head keys.
- [x] `train_v5_weakfix_v3.csv` (129,884 samples) is verified and path-disjoint from test sets.
- [x] `test_coursework_44methods_balanced_zero_leakage.csv` (21,446 samples) passes 0% leakage audit.
- [x] `test_coursework_44methods_full_zero_leakage.csv` (50,084 samples) passes 0% leakage audit.
- [x] `coursework_eda.ipynb` runs from start to finish without errors.
- [x] `coursework_deepfake.ipynb` runs from start to finish without errors.
- [x] `predict_image.ipynb` runs from start to finish without errors.
