# OVERVIEW.md — Project Overview & Strategic Roadmap

- **Motivation/Background**: Deepfake generation has evolved from traditional face-swapping to high-fidelity diffusion and GAN synthesis, requiring robust cross-generator evaluation across 44 methods.
- **Purpose**: Serve as the living master roadmap, architecture guide, and checkpoint index for the deepfake-ViT project.
- **Overview Pipeline**: Multi-generator benchmarking pipeline comparing DINOv3 ViT-S/16 against ConvNeXt-Tiny under strict zero-leakage constraints.
- **Detailed Plan**: §1 Project Motivation & Scope; §2 Dataset Architecture & Zero-Leakage Protocol; §3 Model Architectures & Checkpoints; §4 Interactive Notebooks & Utilities; §5 Pipeline Phases & Roadmap; §6 Hardware & Execution Constraints; §7 Progress Pointers.
- **References**: `docs/PURPOSE.md`, `docs/phases/DATA_PREP.md`, `docs/phases/MODEL.md`, `docs/phases/EVAL.md`.
- **Created**: 2026-08-18T08:56:25+07:00
- **Last Updated**: 2026-09-08T10:15:00+07:00

---

## 📑 Table of Contents

1. [Project Motivation & Scope](#1-project-motivation--scope)
2. [Dataset Architecture & Zero-Leakage Protocol](#2-dataset-architecture--zero-leakage-protocol)
3. [Model Architectures & Checkpoints](#3-model-architectures--checkpoints)
4. [Interactive Notebooks & Utilities](#4-interactive-notebooks--utilities)
5. [Pipeline Phases & Roadmap](#5-pipeline-phases--roadmap)
6. [Hardware & Execution Constraints](#6-hardware--execution-constraints)
7. [Progress Pointers](#7-progress-pointers)

---

## 1. Project Motivation & Scope

Deepfake generation technologies have rapidly evolved from traditional autoencoder-based face-swapping (FaceSwap, DeepFaceLab) to high-fidelity diffusion models (Stable Diffusion, Midjourney, PixArt, DiT) and sophisticated GAN architectures (StyleGAN2/3, StarGAN v2). Traditional forensic classifiers often suffer from severe cross-generator performance drops due to domain overfitting and subtle data leakage.

The `deepfake-ViT` project addresses these challenges by:
1. **Leveraging Meta DINOv3 Self-Supervised Vision Transformer:** Exploiting global self-attention representations to detect subtle frequency, blending, and structural anomalies.
2. **Comparing Against Modern CNN Baseline (ConvNeXt-Tiny):** Evaluating the complementary inductive biases of local convolutions vs. global attention across 44 distinct manipulation methods.
3. **Enforcing Strict 3-Tier Zero-Leakage Auditing:** Ensuring zero identity, video, or byte-level MD5 collision overlaps between training and testing benchmarks.

---

## 2. Dataset Architecture & Zero-Leakage Protocol

The dataset comprises **44 Deepfake Fake Methods** and **7 Real Data Sources** (FFHQ, CelebA-HQ, Celeb-DF v2, FaceForensics++, VGGFace2, SFHQ, CelebV-HQ).

| Dataset Split | File Path | Total Images | Class Distribution | Description |
| :--- | :--- | :--- | :--- | :--- |
| **Train (v3 Clean)** | [`data/splits/train_v5_weakfix_v3.csv`](../data/splits/train_v5_weakfix_v3.csv) | **129,884** | 31,006 Real : 98,878 Fake | 51 balanced subsets across 44 fake methods + 7 real domains |
| **Validation** | [`data/splits/val_v5_combined_universal_kaggle_boost.csv`](../data/splits/val_v5_combined_universal_kaggle_boost.csv) | **6,000** | 3,000 Real : 3,000 Fake | Balanced validation set for hyperparameter tuning & early stopping |
| **Test CourseWork Balanced** | [`test_coursework_44methods_balanced_zero_leakage.csv`](../data/splits/test_coursework_44methods_balanced_zero_leakage.csv) | **21,446** | 10,723 Real : 10,723 Fake | Exactly balanced 1:1 test set (~300 images per method, 0% leak) |
| **Test CourseWork Full Suite** | [`test_coursework_44methods_full_zero_leakage.csv`](../data/splits/test_coursework_44methods_full_zero_leakage.csv) | **50,084** | 25,042 Real : 25,042 Fake | Full exhaustive test set (~600–1,500 images per method, 0% leak) |

### 🛡️ 3-Tier Zero-Leakage Verification:
1. **Tier 1 (Path Disjoint):** $Train \cap Test_{Bal} = \emptyset$, $Train \cap Test_{Full} = \emptyset$ (0 overlapping image paths).
2. **Tier 2 (Identity / Video Disjoint):** No source video or subject identity from train appears in test.
3. **Tier 3 (MD5 Byte-Level Deduplication):** Audited 127,185 MD5 hashes, eliminating 5,680 path overlaps and 2,261 exact byte collision pairs.

---

## 3. Model Architectures & Checkpoints

### 🌟 Model 1: Meta DINOv3 ViT-Small/16 (Vision Transformer)
- **Backbone:** Meta DINOv3 ViT-S/16 (`img_size=256`, `patch_size=16`, `embed_dim=384`, 12 Transformer blocks, 6 attention heads, 4 register tokens).
- **Classification Head:** `Linear(384, 2)`.
- **Total Parameters:** **28.69M** (DINOv3 ViT-S/16 Plus with SwiGLU Gated MLP, matching ConvNeXt-Tiny's 28.12M for strict parameter parity).
- **Active Checkpoint:** [`experiments/checkpoints/best_model_v3.pt`](../experiments/checkpoints/best_model_v3.pt) (Finetuned on 129.8k samples, Best Val AUC: `0.9940`).

### 🔷 Model 2: Meta DINOv3 ConvNeXt-Tiny (Modern CNN Baseline / ConvicT)
- **Backbone:** Meta DINOv3 ConvNeXt-Tiny (4 stages, depths=[3, 3, 9, 3], dims=[96, 192, 384, 768]).
- **Classification Head:** 2-layer GELU MLP (`768 → 384 → 2`).
- **Total Parameters:** **28.12M**.
- **Active Checkpoint:** [`experiments/checkpoints/convnext_weakfix_v3.pt`](../experiments/checkpoints/convnext_weakfix_v3.pt) (Finetuned on 129.8k samples).

---

## 4. Interactive Notebooks & Utilities

| Notebook | Cells | Description | Link |
| :--- | :---: | :--- | :--- |
| **Coursework EDA Notebook** | **32** | Comprehensive multi-split census (207.4k samples), 54-methods taxonomy, 2D FFT, 1D Radial PSD, GLCM textures & KS-test. | [`notebooks/coursework_eda.ipynb`](../notebooks/coursework_eda.ipynb) |
| **Coursework Evaluation Notebook** | **35** | 3-tier zero-leakage auditing, live GPU benchmarking (ViT vs ConvNeXt vs Ensemble), 5 post-training visualization charts, and visual error gallery. | [`notebooks/coursework_deepfake.ipynb`](../notebooks/coursework_deepfake.ipynb) |
| **Single-Image Predictor** | **9** | Interactive single-image inference tool with prediction probability visualization. | [`notebooks/predict_image.ipynb`](../notebooks/predict_image.ipynb) |

---

## 5. Hardware & Execution Constraints
- **Local Machine:** Linux workstation with NVIDIA GeForce RTX 3050 Laptop GPU (4 GB VRAM) & Python 3.14 (`.venv`).
- **Multiprocessing Rule:** Always set `num_workers = 0` in interactive `DataLoader` calls to prevent Python 3.14+ `forkserver` unpickling conflicts.
