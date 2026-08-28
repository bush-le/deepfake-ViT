# Model Experiments & Research Documentation

This directory contains research experiment proposals, optimization plans, methodology audits, and empirical benchmark logs for the `deepfake-ViT` project.

---

## 🏗️ Experiment Lifecycle

```
Research Proposal & Plan (agents/experiments/EXP_*.md)
        ↓
Script-Only Execution (src/training/, scripts/)
        ↓
Checkpointing & Evaluation Artifacts (experiments/checkpoints/, experiments/results/)
        ↓
Interactive Verification (notebooks/coursework_deepfake.ipynb)
        ↓
5W1H Reporting & Status Sync (agents/progress/, agents/phases/)
```

---

## 📋 Experiments Index

| Experiment ID | Title & Strategic Objective | Target / Metric | Status | Primary Document |
| :---: | :--- | :---: | :---: | :--- |
| **`EXP-01`** | **Strategic Accuracy Optimization Plan** (LLRD, Balanced Batches, Label Smoothing, TTA, Ensemble) | Acc $>97.5\%$ | Completed | [`EXP_01_ACCURACY_OPTIMIZATION_PLAN.md`](EXP_01_ACCURACY_OPTIMIZATION_PLAN.md) |
| **`EXP-02`** | **Deep Error Analysis & Domain Generalization** (Hard method discovery, Midjourney boost) | Cross-domain Acc | Completed | [`EXP_02_ACCURACY_IMPROVEMENT_PLAN.md`](EXP_02_ACCURACY_IMPROVEMENT_PLAN.md) |
| **`EXP-03`** | **3-Tier Zero-Leakage Dataset Rebuild** (Identity disjoint, Path disjoint, MD5 Byte Collision Deduplication) | 0.00% Leak | Completed | [`EXP_03_ZERO_LEAKAGE_AND_BALANCED_TRAINING_PLAN.md`](EXP_03_ZERO_LEAKAGE_AND_BALANCED_TRAINING_PLAN.md) |
| **`EXP-04`** | **CourseWork 44-Methods Zero-Leakage Benchmark** (DINOv3 ViT vs. ConvNeXt Side-by-Side Evaluation & Diagnostics) | 44 Methods $>97\%$ | Completed | [`EXP_04_COURSEWORK_44METHODS_BENCHMARK.md`](EXP_04_COURSEWORK_44METHODS_BENCHMARK.md) |

---

## 📚 Experiment Guidelines

1. **5W1H Result Reporting:** All empirical results must explicitly state Who, What, When, Where, Why, and How ([rules/RESULTS_REPORTING.md](../rules/RESULTS_REPORTING.md)).
2. **Deterministic Reproducibility:** Fixed seeds, pinned hyperparameters, and frozen checkpoints.
3. **No Training in Notebooks:** Full-scale model training is performed strictly via scripts.
