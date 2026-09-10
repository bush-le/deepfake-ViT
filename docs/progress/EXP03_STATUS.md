# EXP-03 / EXP-04 Status & Benchmark Progress Tracker

- **Motivation/Background**: Provide real-time progress tracking, checklist status, and milestone completion for EXP03_STATUS.
- **Purpose**: Maintain an accurate audit trail of completed tasks and active blockers for EXP03_STATUS.
- **Overview Pipeline**: Milestone tracking -> task checklist review -> verification status update.
- **Detailed Plan**: §1 Current Milestone Status; §2 Completed Deliverables; §3 Active Blockers; §4 Next Priorities.
- **References**: `docs/OVERVIEW.md`, `docs/phases/`.
- **Created**: 2026-08-23T01:34:32+07:00
- **Last Updated**: 2026-09-08T10:15:00+07:00

---

  - [`notebooks/coursework_eda.ipynb`](../../notebooks/coursework_eda.ipynb) (Master EDA, 46 cells (20 sections))
  - [`notebooks/coursework_deepfake.ipynb`](../../notebooks/coursework_deepfake.ipynb) (Master Evaluation, 35 cells)
  - [`notebooks/predict_image.ipynb`](../../notebooks/predict_image.ipynb) (Single Image Predictor, 9 cells)

---

## 1. Objectives & Key Achievements
- [x] Engineered expanded 44-methods test benchmarks:
  - `test_coursework_44methods_balanced_zero_leakage.csv` (21,446 images, exact 1:1)
  - `test_coursework_44methods_full_zero_leakage.csv` (50,084 images, exact 1:1)
- [x] Executed full 3-tier Zero-Leakage audit (0 path overlap, 127k MD5 hashes deduplicated).
- [x] Evaluated **Meta DINOv3 ViT-S/16 Plus** (`plus_v3_s1_best.pt`, 28.69M params) vs. **Meta DINOv3 ConvNeXt-Tiny** (`convnext_weakfix_v3.pt`, 28.12M params).
- [x] Achieved benchmark accuracy $>97.6\%$ and ROC-AUC $>99.6\%$.
- [x] Successfully verified all notebook cells under Python 3.14 with zero runtime errors.

---

## 2. Quantitative Benchmark Summary (Test Balanced 21.4k)

| Metric | DINOv3 ViT-S/16 | DINOv3 ConvNeXt-Tiny | Joint Ensemble | Coursework Target |
| :--- | :---: | :---: | :---: | :---: |
| **Accuracy** | **97.66%** | **97.26%** | **97.80%** | $\ge 95.0\%$ |
| **ROC-AUC** | **99.64%** | **99.58%** | **99.68%** | $\ge 98.0\%$ |
| **F1-Score** | **97.68%** | **97.26%** | **97.80%** | $\ge 95.0\%$ |
| **Fake Recall** | **98.24%** | **97.24%** | **98.05%** | $\ge 95.0\%$ |
| **Real Specificity**| **97.09%** | **97.28%** | **97.55%** | $\ge 95.0\%$ |

---

## 3. Related Documentation
- Experiment Report: [`agents/experiments/EXP_04_COURSEWORK_44METHODS_BENCHMARK.md`](../experiments/EXP_04_COURSEWORK_44METHODS_BENCHMARK.md)
- Training Specification: [`agents/phases/TRAINING_INFO.md`](../phases/TRAINING_INFO.md)
- Evaluation Specification: [`agents/phases/EVAL.md`](../phases/EVAL.md)
