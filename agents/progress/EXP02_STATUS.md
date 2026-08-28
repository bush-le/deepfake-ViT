# EXP-02 Status & Milestone Tracker

- **Experiment Name:** EXP-02 Multi-Domain Accuracy Optimization
- **Status:** **COMPLETED**
- **Target Notebooks:** [`notebooks/coursework_eda.ipynb`](../../notebooks/coursework_eda.ipynb), [`notebooks/coursework_deepfake.ipynb`](../../notebooks/coursework_deepfake.ipynb)

---

## 1. Objectives & Achievements
- [x] Identified labeling bug in Celeb-DF test evaluation (`BUG_01_CELEBDF_TEST_LABEL_FIX.md`).
- [x] Corrected ground-truth labels across all 1,700 Celeb-DF test fake samples.
- [x] Standardized `test_balanced.csv` to an exact 1:1 ratio (2,067 Real vs. 2,067 Fake).
- [x] Implemented Layer-wise Learning Rate Decay (LLRD, $\gamma=0.80$) and cosine scheduling.
- [x] Validated true baseline accuracy of 93.57% on unbiased test sets.

---

## 2. Key Artifacts
- Diagnostic Report: [`agents/bugs/BUG_01_CELEBDF_TEST_LABEL_FIX.md`](../bugs/BUG_01_CELEBDF_TEST_LABEL_FIX.md)
- Optimization Plan: [`agents/experiments/EXP_02_ACCURACY_IMPROVEMENT_PLAN.md`](../experiments/EXP_02_ACCURACY_IMPROVEMENT_PLAN.md)
