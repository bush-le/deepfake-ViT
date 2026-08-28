# Agent AI Knowledge Base & Control Layer — deepfake-ViT

Welcome to the **Agent AI Knowledge Base** for the `deepfake-ViT` repository (Anti-Deepfake Face Detection using Meta DINOv3 ViT-Small/16 and DINOv3 ConvNeXt-Tiny).
This directory serves as the centralized "second brain", memory, architecture reference, and governance layer for all AI agents and engineers working on this project.

---

## 🏗️ Architecture & Pipeline Overview

`agents/` governs the complete ML lifecycle across data, models, training, evaluation, and documentation:

```
Data (data/splits/) → Models (src/models/) → Training Scripts (src/training/)
    → Evaluation & Benchmarking (src/eval/, scripts/)
    → Interactive Notebooks (notebooks/coursework_deepfake.ipynb, notebooks/coursework_eda.ipynb, notebooks/predict_image.ipynb)
                 ↕ Results & Artifacts (experiments/checkpoints/, experiments/results/) ↕
Governance (agents/): rules → phases → progress → experiments → bugs → references
```

### 🧠 Core Project Pillars:
1. **Model Architectures:**
   - **Primary Model (Transformer):** Meta DINOv3 ViT-Small/16 (`embed_dim=384`, 12 layers, 6 heads, 4 register tokens, `Linear(384, 2)` head, ~21.60M parameters). Checkpoint: [`best_model_v3.pt`](../experiments/checkpoints/best_model_v3.pt).
   - **Matched Baseline (Modern CNN):** Meta DINOv3 ConvNeXt-Tiny (stages=[3,3,9,3], dims=[96,192,384,768], 2-layer GELU MLP head `768 → 384 → 2`, ~28.12M parameters). Checkpoint: [`convnext_weakfix_v3.pt`](../experiments/checkpoints/convnext_weakfix_v3.pt).
   - **Ensemble & Adapters:** Joint inference (`EnsembleClassifier` $0.65 \cdot P_{ViT} + 0.35 \cdot P_{CNN}$) and Parameter-Efficient Fine-Tuning (`LoRA`).
2. **Datasets & 3-Tier Zero-Leakage Protocol:**
   - **Fixed Train Split:** [`train_v5_weakfix_v3.csv`](../data/splits/train_v5_weakfix_v3.csv) (**129,884 images** across 51 subsets, 44 fake methods + 7 real sources).
   - **Validation Split:** [`val_v5_combined_universal_kaggle_boost.csv`](../data/splits/val_v5_combined_universal_kaggle_boost.csv) (**6,000 images**).
   - **Test CourseWork Balanced (1:1):** [`test_coursework_44methods_balanced_zero_leakage.csv`](../data/splits/test_coursework_44methods_balanced_zero_leakage.csv) (**21,446 images** — 10,723 Real : 10,723 Fake, ~300 images/method, 0.00% leak).
   - **Test CourseWork Full Suite:** [`test_coursework_44methods_full_zero_leakage.csv`](../data/splits/test_coursework_44methods_full_zero_leakage.csv) (**50,084 images** — 25,042 Real : 25,042 Fake, ~600–1500 images/method, 0.00% leak).
   - **Leakage Auditing:** Complete deduplication eliminating 5,680 path overlaps and 2,261 MD5 byte-level collisions between Train and Test splits.
3. **Primary Authoritative Notebooks:**
   - [`coursework_eda.ipynb`](../notebooks/coursework_eda.ipynb): Master 46 cells (20 sections) exploratory data analysis notebook featuring multi-split census (207.4k samples), 5-family generative taxonomy (44 methods + 7 real sources), physical color space metrics, 2D Fourier FFT spectral analysis, 1D Radial PSD decay curves, GLCM skin textures, and Kolmogorov-Smirnov (KS-Test) statistical verification.
   - [`coursework_deepfake.ipynb`](../notebooks/coursework_deepfake.ipynb): Master 35-cell evaluation notebook featuring 3-tier zero-leakage auditing, live GPU inference, side-by-side ViT vs. ConvNeXt benchmarking, 5 post-training visualization charts, Youden's J threshold optimization, and deep error diagnostics + visual gallery.
   - [`predict_image.ipynb`](../notebooks/predict_image.ipynb): Standalone interactive single-image tester with prediction probability visualization.

---

## 📁 Directory Architecture

```
agents/
├── README.md                      # Entry point & knowledge base guide (this file)
├── OVERVIEW.md                    # Core project overview, goals, architecture & roadmap
├── PURPOSE.md                     # Project brief, requirements & success criteria
├── TECHNICAL_GUIDE.md             # Deep technical specifications, models, & run guides
├── DATA_SPLIT_SUMMARIZE.md        # Comprehensive dataset census (129.8k train, 21.4k bal, 50k full)
├── DATA_PREP_SUMMARY_REPORT.md    # Extraction, cleaning, and zero-leakage pipeline report
├── EDA_DATA_INVENTORY.md          # 44 Fake methods + 7 real domains taxonomy & inventory
├── CODEBASE_AUDIT.md              # System-wide codebase audit and integrity scorecard
├── HOW_TO_SETUP_AI_AGENT.md       # AI agent workflow & configuration manual
├── ML_PIPELINE_REFERENCE_v3.md    # Detailed 18-step end-to-end ML pipeline reference
│
├── rules/                         # Mandatory conventions & behavioral guidelines
│   ├── AGENT_AI.md                # Agent AI behavioral contracts & pair-programming rules
│   ├── CODEBASE_AUDIT.md          # Pre-task codebase verification checklist
│   ├── FOLDER_STRUCTURE.md        # Official repository directory layout specification
│   ├── LOGGING_CHECKPOINT_RULES.md# Checkpoint formatting, safety & logging rules
│   ├── MD_CONVENTION.md           # Markdown documentation standard
│   ├── NAMING_CONVENTION.md       # Standardized file, function & variable naming
│   ├── NOTEBOOK_HEADER_CONVENTION.md# Notebook metadata & cell header conventions
│   └── WORKSPACE_INTEGRITY.md     # Multi-agent workspace safety & conflict resolution
│
├── phases/                        # Project phase blueprints & technical specs
│   ├── DATA_PREP.md               # Data collection, cleaning, splits & zero-leakage auditing
│   ├── MODEL.md                   # Model architecture, backbones, heads & LoRA specs
│   ├── TRAINING_INFO.md           # Training loops, loss functions, optimizers & LLRD schedules
│   ├── EVAL.md                    # Evaluation metrics, protocols, scripts & test suites
│   ├── EXPERIMENTS.md             # Experiment tracking & historical results archive
│   ├── CROSS_DOMAIN_BALANCED_PROTOCOL.md # Cross-domain dataset construction protocol
│   └── DOMAIN_AGNOSTIC_AUGMENTATION_SPEC.md # Forensic-safe data augmentation pipeline
│
├── progress/                      # Live execution status & milestones
│   ├── MASTER_STATUS.md           # Global project status & phase progress overview
│   ├── DATA_PREP_STATUS.md        # Data preparation & split status tracker
│   ├── MODEL_STATUS.md            # Model loading & weight verification tracker
│   ├── TRAINING_STATUS.md         # Training runs & checkpoint status tracker
│   ├── EVAL_STATUS.md             # Benchmark evaluation & test results tracker
│   ├── CODEBASE_AUDIT_STATUS.md   # System audit & file tree verification status
│   ├── EXP02_STATUS.md            # EXP-02 milestone & validation tracker
│   └── EXP03_STATUS.md            # EXP-03 / EXP-04 benchmark progress tracker
│
├── experiments/                   # Detailed experiment logs & reports
│   ├── README.md                  # Experiments directory guide & master index
│   ├── EXP_01_ACCURACY_OPTIMIZATION_PLAN.md # Baseline optimization plan
│   ├── EXP_02_ACCURACY_IMPROVEMENT_PLAN.md # Weak method diagnostics & roadmap
│   ├── EXP_03_ZERO_LEAKAGE_AND_BALANCED_TRAINING_PLAN.md # Zero-leakage training plan
│   ├── EXP_03_06_CONTINUED_IMPROVEMENT_PLAN.md # Multi-pillar optimization plan
│   └── EXP_04_COURSEWORK_44METHODS_BENCHMARK.md # 44-Methods zero-leakage benchmark report
│
└── bugs/                          # Incident post-mortems & bug documentation
    ├── README.md                  # Bugs directory guide & resolved issues index
    ├── BUG_01_CELEBDF_TEST_LABEL_FIX.md # Celeb-DF test labeling bug & resolution
    ├── BUG_02_IMGDS_PICKLE_FORKSERVER.md # Python 3.14 forkserver unpickling fix
    └── BUG_03_ZERO_DIVISION_TEST_OLD.md # Zero division guard on legacy benchmark CSVs
```

---

## 🔗 Quick Links to Primary Artifacts

| Resource | File Path / Link |
| :--- | :--- |
| **Master EDA Notebook** | [`notebooks/coursework_eda.ipynb`](../notebooks/coursework_eda.ipynb) |
| **Master Benchmark Notebook** | [`notebooks/coursework_deepfake.ipynb`](../notebooks/coursework_deepfake.ipynb) |
| **Single-Image Predictor** | [`notebooks/predict_image.ipynb`](../notebooks/predict_image.ipynb) |
| **Fixed Train Split (129.8k)** | [`data/splits/train_v5_weakfix_v3.csv`](../data/splits/train_v5_weakfix_v3.csv) |
| **Test Balanced Split (21.4k)** | [`data/splits/test_coursework_44methods_balanced_zero_leakage.csv`](../data/splits/test_coursework_44methods_balanced_zero_leakage.csv) |
| **Test Full Split (50.0k)** | [`data/splits/test_coursework_44methods_full_zero_leakage.csv`](../data/splits/test_coursework_44methods_full_zero_leakage.csv) |
| **ViT Checkpoint** | [`experiments/checkpoints/best_model_v3.pt`](../experiments/checkpoints/best_model_v3.pt) |
| **ConvNeXt Checkpoint** | [`experiments/checkpoints/convnext_weakfix_v3.pt`](../experiments/checkpoints/convnext_weakfix_v3.pt) |
