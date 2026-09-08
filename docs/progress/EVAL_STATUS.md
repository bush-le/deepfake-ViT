# EVAL_STATUS.md — Evaluation & Comparative Benchmarking Status

- **Title:** Evaluation & Comparative Benchmarking Status (DINOv3 ViT vs. ConvNeXt)
- **Date Created:** 2026-08-18
- **Last Updated:** 2026-08-28
- **Description:** Status of live inference benchmarking, per-method accuracy rankings, confusion matrix analysis, and post-training diagnostics.
- **Status:** **Done & Verified**
- **Phase Doc:** [`../phases/EVAL.md`](../phases/EVAL.md)

---

## Log

- **2026-08-18:** Evaluation pipeline established with secure weight loading (`weights_only=True`).
- **2026-08-22:** Published preliminary benchmark reports across 40 methods.
- **2026-08-24:** Expanded evaluation suite to **44 Deepfake methods** (21.4k balanced & 50.0k full suites).
- **2026-08-28:** Integrated live GPU inference and side-by-side comparative benchmarking (DINOv3 ViT-S/16 vs. DINOv3 ConvNeXt-Tiny) into [`notebooks/coursework_deepfake.ipynb`](../../notebooks/coursework_deepfake.ipynb).
- **2026-08-28:** Built 5 major post-training visualization charts and deep error diagnostics galleries (KDE score distribution, Tri-Curve ROC/PR/ECE suite, 5-category breakdown, 44-methods horizontal bar ranking, inductive bias scatter plot, Youden's J threshold optimization, and visual image gallery).
- **2026-08-28:** Created dedicated single-image interactive testing notebook [`notebooks/predict_image.ipynb`](../../notebooks/predict_image.ipynb).
- **2026-08-28:** Verified `coursework_deepfake.ipynb` runs from start to finish on GPU with zero errors and produces side-by-side confusion matrices, per-method accuracy charts, and inductive bias correlation plots.

---

## Benchmark Headline Performance

| Model | Test Suite | Accuracy | ROC-AUC | F1-Score | Status |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **DINOv3 ViT-S/16** | Test Balanced (21.4k) | **97.64%** | **99.68%** | **97.63%** | 🏆 Exceeded |
| **DINOv3 ConvNeXt-Tiny** | Test Balanced (21.4k) | **97.12%** | **99.54%** | **97.10%** | 🏆 Exceeded |
| **Joint Ensemble** | Test Balanced (21.4k) | **97.88%** | **99.74%** | **97.87%** | 🏆 Exceeded |
| **DINOv3 ViT-S/16** | Test Full Suite (50.0k)| **97.21%** | **99.51%** | **97.20%** | 🏆 Exceeded |
| **DINOv3 ConvNeXt-Tiny** | Test Full Suite (50.0k)| **96.84%** | **99.40%** | **96.83%** | 🏆 Exceeded |

---

## Links

- Phase Doc: [`../phases/EVAL.md`](../phases/EVAL.md)
- Experiment 04 Report: [`../experiments/EXP_04_COURSEWORK_44METHODS_BENCHMARK.md`](../experiments/EXP_04_COURSEWORK_44METHODS_BENCHMARK.md)
- Primary Evaluation Notebook: [`../../notebooks/coursework_deepfake.ipynb`](../../notebooks/coursework_deepfake.ipynb)
- Single Image Tester: [`../../notebooks/predict_image.ipynb`](../../notebooks/predict_image.ipynb)
- Evaluation Results Directory: `experiments/results/courseWorkCheck/`
