# Documentation Index & Architectural Navigation Guide

- **Motivation/Background**: Centralized documentation index cataloging all living roadmaps, research notes, phase specifications, experiment tracking, and bug diagnoses under Archetype A.
- **Purpose**: Provide seamless navigation across all project documentation directories in deepfake-ViT.
- **Overview Pipeline**: Comprehensive documentation tree mapping the 6-stage deep learning development lifecycle.
- **Detailed Plan**: §1 Executive Overview; §2 Core Documents; §3 Pipeline Phases; §4 Experiments & Benchmarks; §5 Bug Reports; §6 Progress Tracking; §7 References & SOPs.
- **References**: `docs/PURPOSE.md`, `docs/OVERVIEW.md`, `agents/rules/FOLDER_STRUCTURE.md`.
- **Created**: 2026-09-08T08:44:12+07:00
- **Last Updated**: 2026-09-08T10:15:00+07:00

---

Welcome to the human-readable documentation portal for the **High-Generalization Facial Deepfake Forensics & Detection** project (`deepfake-ViT`), structured in accordance with **Archetype A (Single-Track Pipeline)** from `Deep_learning_template`.

---

## 📑 Core Documentation Index

| Document | Description | Key Focus & Scope |
| :--- | :--- | :--- |
| [**`PURPOSE.md`**](PURPOSE.md) | **🎯 Project Brief & Requirements** | Formal problem definition, research deliverables, rubric criteria, and success metrics (>95% accuracy). |
| [**`OVERVIEW.md`**](OVERVIEW.md) | **🗺️ Strategic Roadmap & Architecture** | Living master roadmap, 44-method taxonomy, model specs, zero-leakage protocol, and checkpoint inventory. |
| [**`PR_05_AUDIT_AND_MERGE_REPORT.md`**](PR_05_AUDIT_AND_MERGE_REPORT.md) | **🛡️ PR #5 Audit & Integration Report** | Immutable, traceable log of all 32 conflict resolutions, 5 bug findings, and verification gates for PR #5. |
| [**`CODEBASE_AUDIT_REPORT.md`**](CODEBASE_AUDIT_REPORT.md) | **📋 Codebase Audit Report** | Continuous verification of code quality, security, directory compliance, data integrity, and resolution status. |
| [**`DEEPFAKE_FORENSICS_REPORT.md`**](DEEPFAKE_FORENSICS_REPORT.md) | **🔬 Master Forensic & Experimental Research Report** | Comprehensive 5-part research report covering 207k dataset census, 16 physical forensic EDA techniques, zero-leakage firewall protocols, and 4-model per-method diagnostic charts. |
| [**`THEORY_AND_MODEL_COMPARISON.md`**](THEORY_AND_MODEL_COMPARISON.md) | **📐 Theoretical Foundations & Architectural Analysis** | Mathematical analysis of inductive biases (Global Self-Attention vs. Depthwise Convolutions), computational scaling laws, PyTorch implementations, and receptive field dynamics. |
| [**`MODELS.md`**](MODELS.md) | **🧠 Model Architecture & Checkpoint Specifications** | Complete inventory of checkpoints (`plus_v3_s1_best.pt`, `convnext_weakfix_v3.pt`), parameter counts, latency metrics (FPS), and classification head structures. |
| [**`THEORY_AND_MODEL_COMPARISON.pdf`**](THEORY_AND_MODEL_COMPARISON.pdf) | **📑 Academic Research Paper (PDF)** | Formatted academic publication PDF with theoretical derivations, math models, and full architecture comparisons. |
| [**`presentation.tex`**](presentation.tex) | **📊 Presentation Slides (LaTeX Beamer)** | Conference / coursework presentation slide deck covering theoretical foundations, model architectures, and results. |
| [**`RUNPOD.md`**](RUNPOD.md) | **🚀 GPU Infrastructure & Cloud Training Runbook** | Step-by-step operational guide for provisioning RunPod instances, mounting Hugging Face datasets via rclone, running AMP training loops, and synchronizing artifacts. |
| [**`DESCRIPTION_NOTES.md`**](DESCRIPTION_NOTES.md) | **📝 Project Synthesis & Evaluation Notes** | High-level summary of model performance, dataset splits, and benchmarking methodologies. |

---

## 📂 Documentation Subdirectories

### 1. [Pipeline Phases (`docs/phases/`)](phases/)
Technical specifications, requirements, and deliverables for each pipeline phase:
- [`phases/DATA_PREP.md`](phases/DATA_PREP.md) — Multi-domain dataset collection, face extraction, and zero-leakage deduplication across 44 fake methods.
- [`phases/DATA_PREP_SUMMARY_REPORT.md`](phases/DATA_PREP_SUMMARY_REPORT.md) — Executive summary of data harvesting, filtering, and split allocation.
- [`phases/DATA_SPLIT_SUMMARIZE.md`](phases/DATA_SPLIT_SUMMARIZE.md) — Detailed census and balance breakdown of Train, Val, and Test splits.
- [`phases/EDA_DATA_INVENTORY.md`](phases/EDA_DATA_INVENTORY.md) — Complete inventory of 207k images across all source datasets.
- [`phases/CROSS_DOMAIN_BALANCED_PROTOCOL.md`](phases/CROSS_DOMAIN_BALANCED_PROTOCOL.md) — Mathematical protocol for balanced 1:1 cross-generator sampling.
- [`phases/DOMAIN_AGNOSTIC_AUGMENTATION_SPEC.md`](phases/DOMAIN_AGNOSTIC_AUGMENTATION_SPEC.md) — Data augmentation policies (flips, color jitter, blur, sharpness).
- [`phases/MODEL.md`](phases/MODEL.md) — Backbone specifications (DINOv3 ViT-S/16, ConvNeXt-Tiny) and classification head designs.
- [`phases/TRAINING_INFO.md`](phases/TRAINING_INFO.md) — Optimization hyperparameters, cosine annealing, and loss functions.
- [`phases/EVAL.md`](phases/EVAL.md) — Evaluation harness, test-time augmentation (TTA), and optimal threshold tuning.

### 2. [Experiments & Benchmarks (`docs/experiments/`)](experiments/)
Rigorous documentation of experimental hypotheses, configurations, runs, and results:
- [`experiments/EXP_01_ACCURACY_OPTIMIZATION_PLAN.md`](experiments/EXP_01_ACCURACY_OPTIMIZATION_PLAN.md) — Baseline probe and initial learning rate sweeps.
- [`experiments/EXP_02_ACCURACY_IMPROVEMENT_PLAN.md`](experiments/EXP_02_ACCURACY_IMPROVEMENT_PLAN.md) — LayerNorm + 2-layer GELU MLP head enhancement.
- [`experiments/EXP_03_ZERO_LEAKAGE_AND_BALANCED_TRAINING_PLAN.md`](experiments/EXP_03_ZERO_LEAKAGE_AND_BALANCED_TRAINING_PLAN.md) — Strict zero-leakage 1:1 training suite V3.
- [`experiments/EXP_03_06_CONTINUED_IMPROVEMENT_PLAN.md`](experiments/EXP_03_06_CONTINUED_IMPROVEMENT_PLAN.md) — Advanced augmentation and loss weighting experiments.
- [`experiments/EXP_04_COURSEWORK_44METHODS_BENCHMARK.md`](experiments/EXP_04_COURSEWORK_44METHODS_BENCHMARK.md) — Full 44-method comparative benchmark.

### 3. [Bug Reports & Remediation (`docs/bugs/`)](bugs/)
Systematic root cause analysis, diagnoses, and regression prevention:
- [`bugs/BUG_01_CELEBDF_TEST_LABEL_FIX.md`](bugs/BUG_01_CELEBDF_TEST_LABEL_FIX.md) — Correction of inverted Celeb-DF real/fake test labels.
- [`bugs/BUG_02_IMGDS_PICKLE_FORKSERVER.md`](bugs/BUG_02_IMGDS_PICKLE_FORKSERVER.md) — Resolution of Python 3.14+ forkserver multiprocessing DataLoader deadlock.
- [`bugs/BUG_03_ZERO_DIVISION_TEST_OLD.md`](bugs/BUG_03_ZERO_DIVISION_TEST_OLD.md) — Handling of missing method evaluations in historical test runner.

### 4. [Milestone Progress Trackers (`docs/progress/`)](progress/)
Living checklists and status trackers for active phases:
- [`progress/CODEBASE_AUDIT_STATUS.md`](progress/CODEBASE_AUDIT_STATUS.md) — Audit completion status.
- [`progress/DATA_PREP_STATUS.md`](progress/DATA_PREP_STATUS.md) — Data preparation milestone tracker.
- [`progress/MODEL_STATUS.md`](progress/MODEL_STATUS.md) — Architecture and checkpoint status.
- [`progress/TRAINING_STATUS.md`](progress/TRAINING_STATUS.md) — Training convergence tracking.
- [`progress/EVAL_STATUS.md`](progress/EVAL_STATUS.md) — Evaluation pipeline status.
- [`progress/EXP02_STATUS.md`](progress/EXP02_STATUS.md) — Experiment 02 status log.
- [`progress/EXP03_STATUS.md`](progress/EXP03_STATUS.md) — Experiment 03 status log.

### 5. [Technical References & Guides (`docs/references/`)](references/)
Reusable engineering procedures and technical manuals:
- [`references/TECHNICAL_GUIDE.md`](references/TECHNICAL_GUIDE.md) — Core technical manual for local and cloud training/inference.
- [`references/REPORT_DESCRIPTION.md`](references/REPORT_DESCRIPTION.md) — Methodological specifications for forensic report generation.
- [`references/GIT_AND_RELEASE_BEST_PRACTICES.md`](references/GIT_AND_RELEASE_BEST_PRACTICES.md) — Version control conventions and branch lifecycle rules.
- [`references/OPTUNA_DB_GUIDE.md`](references/OPTUNA_DB_GUIDE.md) — Hyperparameter optimization setup via Optuna SQLite databases.

### 6. [Shared Standards & SOPs (`docs/shared/`)](shared/)
Cross-cutting standard operating procedures:
- [`shared/HOW_TO_SETUP_AI_AGENT.md`](shared/HOW_TO_SETUP_AI_AGENT.md) — AI agent workflow, lifecycle, and tool protocol.
- [`shared/ML_PIPELINE_REFERENCE_v3.md`](shared/ML_PIPELINE_REFERENCE_v3.md) — Comprehensive machine learning pipeline reference architecture.
- [`shared/HANDOFF_TEMPLATE.md`](shared/HANDOFF_TEMPLATE.md) — Agent session handoff template.

---

## 📊 Evaluation Notebooks Linkage

The live interactive Jupyter Notebooks corresponding to these reports are located in [`notebooks/`](../notebooks/):
- [`coursework_deepfake.ipynb`](../notebooks/coursework_deepfake.ipynb) — Master comparative benchmark running live GPU inference across 44 methods.
- [`coursework_eda.ipynb`](../notebooks/coursework_eda.ipynb) — Multi-dimensional exploratory data analysis and optical signal decomposition.
- [`predict_image.ipynb`](../notebooks/predict_image.ipynb) — Single-image interactive inference playground.
- [`deepfake_forensics_report.ipynb`](../notebooks/deepfake_forensics_report.ipynb) — Research evaluation notebook with embedded 5-chart per-method breakdown.
- [`coursework_deepfake_plus_v3_s1_best.ipynb`](../notebooks/coursework_deepfake_plus_v3_s1_best.ipynb) — GPU benchmark and error diagnostic suite for ViT-Plus s1_best.
