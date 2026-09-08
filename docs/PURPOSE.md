# PURPOSE.md — Project Brief & Requirements

- **Motivation/Background**: Establish the foundational problem brief, research objectives, rubric requirements, scope boundaries, and success criteria for the deepfake-ViT coursework project.
- **Purpose**: Define the locked academic and practical requirements for binary face deepfake classification (>95% accuracy goal).
- **Overview Pipeline**: Problem formulation -> requirements definition -> model & dataset scope -> rubric criteria.
- **Detailed Plan**: §1 Academic & Practical Brief; §2 Detailed Requirements & Scope; §3 Success Criteria & Rubric Alignment; §4 Locked Objectives & Deliverables.
- **References**: `docs/OVERVIEW.md`, `docs/phases/MODEL.md`, `docs/phases/DATA_PREP.md`.
- **Created**: 2026-08-18T08:56:20+07:00
- **Last Updated**: 2026-09-08T10:15:00+07:00

---

  1. Fine-tuned **DINOv3 ViT-Small/16** face deepfake classifier exceeding >95% accuracy rubric requirement.
  2. Direct comparative benchmark against matched **DINOv3 ConvNeXt-Tiny** CNN baseline across **44 deepfake methods**.
  3. Interactive evaluation notebook ([`notebooks/coursework_deepfake.ipynb`](../notebooks/coursework_deepfake.ipynb)) with live inference, zero-leakage auditing, and per-method accuracy breakdown.
  4. Interactive single-image prediction tool ([`notebooks/predict_image.ipynb`](../notebooks/predict_image.ipynb)).

---

## 📑 Table of Contents

- [1. Academic & Practical Brief](#1-academic--practical-brief)
- [2. Detailed Requirements & Scope](#2-detailed-requirements--scope)
- [3. Success Criteria & Rubric Alignment](#3-success-criteria--rubric-alignment)
- [4. Locked Objectives & Deliverables](#4-locked-objectives--deliverables)

---

## 1. Academic & Practical Brief

The primary objective is to evaluate whether modern Self-Supervised Vision Transformers (Meta DINOv3 ViT-S/16) provide superior generalization and artifact-detection capabilities compared to Modern Convolutional Networks (ConvNeXt-Tiny) when detecting manipulated facial imagery across **44 distinct deepfake generation methods**.

### Key Investigation Dimensions:
- **Global Self-Attention vs. Local Convolutions:** How Transformer attention maps contrast with CNN convolutional receptive fields in spotting blending borders, diffusion noise, and GAN texture anomalies.
- **Cross-Method Robustness:** Measuring degradation or resilience across classic FaceSwap/Reenactment vs. modern Diffusion models (Stable Diffusion, Midjourney, PixArt, DiT).
- **Zero-Leakage Integrity:** Proving true model generalizability through strict 3-tier leakage elimination (Path, Identity, and Byte-level MD5 collision auditing).

---

## 2. Detailed Requirements & Scope

### In Scope (✅ IN):
- **Model Implementations:**
  - `DinoViTClassifier`: Meta DINOv3 ViT-Small/16 backbone (`embed_dim=384`, 12 layers, 6 heads, 4 registers) + `Linear(384, 2)` head (~21.60M params).
  - `DinoConvNextClassifier`: Meta DINOv3 ConvNeXt-Tiny backbone + 2-layer GELU MLP head (~28.12M params).
  - `EnsembleClassifier`: Joint probability fusion ($0.65 \cdot P_{ViT} + 0.35 \cdot P_{CNN}$).
  - `LoRA`: Parameter-Efficient Fine-Tuning adapter for ViT self-attention.
- **Dataset Infrastructure:**
  - Fixed Training Set: `train_v5_weakfix_v3.csv` (**129,884 samples** across 51 subsets).
  - Test Balanced (1:1): `test_coursework_44methods_balanced_zero_leakage.csv` (**21,446 samples**, 0% leak).
  - Test Full Suite: `test_coursework_44methods_full_zero_leakage.csv` (**50,084 samples**, 0% leak).
- **Evaluation & Visualizations:**
  - Confusion Matrices, ROC curves, Precision-Recall curves.
  - Per-method ranking and comparative delta ($\Delta$) charts across all 44 methods.
  - Scatter correlation analysis of ViT vs. ConvNeXt per-method accuracies.

### Out of Scope (❌ OUT):
- Multi-face bounding box localization or video temporal modeling (task is strictly binary classification on cropped face images).
- Web serving / production REST API deployment (focus is rigorous offline scientific benchmarking and evaluation).

---

## 3. Success Criteria & Rubric Alignment

| Rubric Metric | Minimum Target | Achieved Performance (DINOv3 ViT) | Status |
| :--- | :--- | :--- | :--- |
| **Test Accuracy** | $\ge 95.00\%$ | **$\ge 97.8\%$ (Zero-Leakage Benchmark)** | 🏆 **EXCEEDED** |
| **ROC-AUC** | $\ge 98.00\%$ | **$\ge 99.6\%$** | 🏆 **EXCEEDED** |
| **Zero-Leakage Audit** | 0.00% overlap | **0 path overlaps, 0 MD5 byte collisions** | ✅ **VERIFIED** |
| **Architecture Comparison** | Required | **Side-by-side DINOv3 ViT vs. ConvNeXt analysis** | ✅ **DELIVERED** |
| **Interactive Deliverable** | Required | **`coursework_deepfake.ipynb` & `predict_image.ipynb`** | ✅ **DELIVERED** |

---

## 4. Locked Objectives & Deliverables

1. **Checkpoints (Frozen Weights):**
   - ViT Checkpoint: `experiments/checkpoints/best_model_v3.pt`
   - ConvNeXt Checkpoint: `experiments/checkpoints/convnext_weakfix_v3.pt`
2. **Interactive Checkpoint Evaluator:**
   - [`notebooks/coursework_deepfake.ipynb`](../notebooks/coursework_deepfake.ipynb) allows anyone to rerun live inference on GPU with full metrics and comparisons.
3. **Single Image Tester:**
   - [`notebooks/predict_image.ipynb`](../notebooks/predict_image.ipynb) allows testing arbitrary image files directly.
