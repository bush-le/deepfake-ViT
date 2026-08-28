# EXP-03 / EXP-06: Comprehensive Deepfake Detection Optimization Plan

- **Title:** Unified Advanced Optimization Plan — Resolving Generative Blindspots & Domain Shortcuts
- **Date Created:** 2026-08-22
- **Last Updated:** 2026-08-22
- **Predecessor:** [EXP_02_ACCURACY_IMPROVEMENT_PLAN.md](EXP_02_ACCURACY_IMPROVEMENT_PLAN.md)
- **Status:** Completed & Integrated
- **Experiment ID:** EXP-03 / EXP-06

---

## 1. Post-EXP-02 Baseline Diagnostics

### 1.1 Empirical Achievements ✅
- Val Accuracy: Increased from 94.29% to **97.52%** (+3.23%).
- Val ROC-AUC: Increased to **0.9952**.
- CelebDF_fake Recall: Boosted from 78.00% to **92.62%** (+14.62%).
- FaceDancer Recall: Boosted to **96.00%**.

### 1.2 Identified Weaknesses & Vulnerabilities ❌
- **Midjourney & Modern Diffusion Blindspots**: Low baseline detection rate (~20-32%) due to lack of diverse modern diffusion models in initial training sets.
- **Unconditional High-Res GANs**: `whichfaceisreal` showed ~37-49% recall before high-frequency expansion.
- **Domain Shortcut Bias**: Models tended to associate blurry video compression with Real and high-contrast studio sharpness with Fake.

---

## 2. Five-Pillar Remediation Strategy

```
┌─────────────────────────────────────────────────────────────────────┐
│                      UNIFIED OPTIMIZATION PILLARS                   │
│                                                                     │
│  ① EXPANDED DATA (129.8K)  ② ARTIFACT-SAFE AUG  ③ WEIGHTED LOSS    │
│  51 Subsets, 44 Methods    ColorJitter, Mild Affine  Class Weights   │
│  Multi-domain Balanced     Prevent Shortcut Heuristics  W_real=1.0   │
│                                                                     │
│  ④ DUAL ARCHITECTURE BENCHMARK         ⑤ ENSEMBLE INFERENCE         │
│  ViT-S/16 (Global Self-Attention)      0.65 * ViT + 0.35 * CNN      │
│  ConvNeXt-Tiny (Local Convolutions)    Optimal Decision Threshold   │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 3. Pillar ①: Dataset Expansion & Domain Balancing
- Standardized the primary training set to **129,884 samples** across 51 balanced subsets.
- Included 11 modern diffusion models (`midjourney_v5`, `midjourney_v6`, `sd_v15`, `sd2.1`, `sdxl`, `collabdiff`, `pixart`, `sit`, `dit`, `rddm`, `ddim`).
- Balanced 7 pristine real face domains.

---

## 4. Pillar ②: Artifact-Preserving Augmentation
- Strict avoidance of aggressive Gaussian blurs or heavy low-quality JPEG compression that destroy GAN frequency checkerboard patterns.
- Utilization of mild photometric jittering and horizontal flips.

---

## 5. Pillar ③ & ④: Loss Formulation & Dual Architecture Evaluation
- Applied class-weighted Cross-Entropy Loss ($W_{\text{Real}}=1.0, W_{\text{Fake}}=0.35$) with Label Smoothing ($\epsilon=0.05$).
- Evaluated **Meta DINOv3 ViT-Small/16** against **Meta DINOv3 ConvNeXt-Tiny** across the 44-Methods Zero-Leakage test suite.
- Reached **>97.6% accuracy** and **>99.6% ROC-AUC** across held-out benchmarks.
