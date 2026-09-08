# EVAL.md — Evaluation & Comparative Benchmarking Specifications

- **Motivation/Background**: Establish technical specification, requirements, and deliverables for pipeline phase EVAL.
- **Purpose**: Guide the execution and quality gates of phase EVAL.
- **Overview Pipeline**: Phase scope definition -> implementation guidelines -> verification gates -> status sign-off.
- **Detailed Plan**: §1 Phase Overview & Scope; §2 Technical Specification; §3 Deliverables & Artifacts; §4 Verification Protocol.
- **References**: `docs/OVERVIEW.md`, `docs/PURPOSE.md`, `agents/rules/`.
- **Created**: 2026-08-18T11:19:39+07:00
- **Last Updated**: 2026-09-08T10:15:00+07:00

---

## 1. Background & Rubric Alignment

The evaluation protocol verifies model performance against academic rubric targets:
- **Test Accuracy Target:** $\ge 95.00\%$ on clean held-out zero-leakage test data.
- **ROC-AUC Target:** $\ge 98.00\%$.
- **Required Deliverables:** Side-by-side ViT vs. CNN comparative benchmarking, per-method accuracy breakdown, confusion matrices, post-training visualization charts, and interactive notebooks.

---

## 2. Evaluation Datasets

1. **Test CourseWork Balanced (1:1):** `test_coursework_44methods_balanced_zero_leakage.csv` (**21,446 samples** — 10,723 Real : 10,723 Fake across all 44 methods).
2. **Test CourseWork Full Suite:** `test_coursework_44methods_full_zero_leakage.csv` (**50,084 samples** — 25,042 Real : 25,042 Fake across all 44 methods).

---

## 3. Evaluated Metrics & Definitions

- **Accuracy (Overall):** $\frac{TP + TN}{TP + TN + FP + FN}$
- **ROC-AUC:** Area under the Receiver Operating Characteristic curve.
- **F1-Score (Fake):** Harmonic mean of Precision and Recall on the Fake class (Label = 1).
- **Precision:** $\frac{TP}{TP + FP}$ (Proportion of predicted fakes that are genuinely fake).
- **Recall (Detection Rate):** $\frac{TP}{TP + FN}$ (Proportion of actual fakes successfully caught).
- **Real Accuracy:** $\frac{TN}{TN + FP}$ (Proportion of pristine real images correctly classified).
- **Per-Method Accuracy:** $\frac{\text{Correct Predictions within Method } m}{\text{Total Samples of Method } m}$.

---

## 4. Evaluation Workflows

### 4.1 Standalone Script Execution
```bash
.venv/bin/python scripts/eval_v5_weakfix_v3_report.py
```

### 4.2 Master Interactive Notebook (`notebooks/coursework_deepfake.ipynb` — 35 Cells)
- **Section 0 & 1 (Setup & Zero-Leakage Audit):** Paths, GPU allocation, 3-tier zero-leakage verification (0 overlap, 127k MD5 hashes), and 44-methods census.
- **Section 2 (Model Loading):** Loads DINOv3 ViT-S/16 (`best_model_v3.pt`) and DINOv3 ConvNeXt-Tiny (`convnext_weakfix_v3.pt`).
- **Section 3 (Live Benchmarking):**
  - Evaluates ViT, ConvNeXt, and Joint Ensemble on Test Balanced (21.4k).
  - Side-by-side Confusion Matrices (clean formatting with no white border lines).
  - Evaluates on Test Full Suite (50.0k).
- **Section 4 (Post-Training Visualizations):**
  - Probability Density Distribution (KDE / Histograms) for Real vs. Fake.
  - Standard Tri-Curve Suite: ROC Curves (AUC), Precision-Recall Curves (AP), and Reliability Calibration Diagram (ECE).
  - 5-Category Breakdown (Diffusion, FaceSwap, Reenactment, GAN, Facial Editing).
  - Horizontal Bar Chart ranking all 44 Deepfake methods.
  - Inductive Bias Scatter Plot (Transformer Global vs. CNN Local Receptive Fields).
- **Section 5 (Deep Error Diagnostics & Visual Gallery):**
  - Decision threshold optimization via Youden's J Index ($J = \text{Sens} + \text{Spec} - 1$).
  - Scientific analysis of blindspots and hard cases.
  - Visual Image Gallery showing Top 10 False Negatives, Top 10 False Positives, and Top 10 Hard True Positives.

---

## 5. Summary Benchmark Results

| Model | Test Suite | Accuracy | ROC-AUC | F1-Score | Zero-Leakage |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **DINOv3 ViT-S/16** | Test Balanced (21.4k) | **97.64%** | **99.68%** | **97.63%** | ✅ 0.00% Leak |
| **DINOv3 ConvNeXt-Tiny** | Test Balanced (21.4k) | **97.12%** | **99.54%** | **97.10%** | ✅ 0.00% Leak |
| **Joint Ensemble** | Test Balanced (21.4k) | **97.88%** | **99.74%** | **97.87%** | ✅ 0.00% Leak |
| **DINOv3 ViT-S/16** | Test Full Suite (50.0k)| **97.21%** | **99.51%** | **97.20%** | ✅ 0.00% Leak |
| **DINOv3 ConvNeXt-Tiny** | Test Full Suite (50.0k)| **96.84%** | **99.40%** | **96.83%** | ✅ 0.00% Leak |

---

## 6. Links & References

- Status Tracker: [`../progress/EVAL_STATUS.md`](../progress/EVAL_STATUS.md)
- Experiment 04 Report: [`../experiments/EXP_04_COURSEWORK_44METHODS_BENCHMARK.md`](../experiments/EXP_04_COURSEWORK_44METHODS_BENCHMARK.md)
- Reporting Rules: [`../rules/RESULTS_REPORTING.md`](../../agents/rules/RESULTS_REPORTING.md)
- Notebook: [`../../notebooks/coursework_deepfake.ipynb`](../../notebooks/coursework_deepfake.ipynb)
