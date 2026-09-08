# FOLDER_STRUCTURE.md — Repository Layout Specification

This document is the official source of truth for the directory layout of `deepfake-ViT`. All agents and contributors must follow this structure.

```
deepfake-ViT/
├── docs/                         # Human-readable Technical Documentation & Research Reports
│   ├── README.md                 # Master documentation portal
│   ├── DEEPFAKE_FORENSICS_REPORT.md # Master 5-part forensic & experimental research report
│   ├── THEORY_AND_MODEL_COMPARISON.md # Theoretical foundations & inductive bias comparison
│   ├── MODELS.md                 # Checkpoint specifications & parameter inventory
│   ├── RUNPOD.md                 # GPU cloud infrastructure execution runbook
│   └── DESCRIPTION_NOTES.md      # Method & dataset synthesis notes
│
├── agents/                       # Centralized Agent AI Knowledge Base & Control Layer
│   ├── README.md                 # Entry point & navigation guide
│   ├── OVERVIEW.md               # Core project overview & strategic roadmap
│   ├── PURPOSE.md                # Problem brief & academic rubric requirements
│   ├── TECHNICAL_GUIDE.md        # Technical architecture, loading & evaluation guide
│   ├── DATA_SPLIT_SUMMARIZE.md   # Official dataset census & zero-leakage breakdown
│   ├── DATA_PREP_SUMMARY_REPORT.md # Extraction, cleaning & test set report
│   ├── EDA_DATA_INVENTORY.md     # 44 Fake methods & 7 real domains taxonomy
│   ├── CODEBASE_AUDIT.md         # Codebase health & security audit report
│   ├── HOW_TO_SETUP_AI_AGENT.md  # Agent workflow setup manual
│   ├── ML_PIPELINE_REFERENCE_v3.md # End-to-end 18-step ML pipeline reference
│   ├── rules/                    # Mandatory guidelines & standards
│   ├── phases/                   # Pipeline phase specifications
│   ├── progress/                 # Live phase milestone tracking
│   ├── experiments/              # Experiment proposals & research logs
│   ├── bugs/                     # Bug reports & troubleshooting guides
│   ├── references/               # External reference guides
│   └── templates/                # Standard documentation skeletons
│
├── data/
│   ├── splits/                   # Official CSV splits (train_v5, val_v5, test_coursework)
│   ├── raw/                      # Read-only raw data partitions
│   └── processed/                # Extracted face crops and metadata manifests
│
├── src/
│   ├── data/                     # Dataset loaders, split builders & augmentations
│   ├── models/                   # DINOv3 ViT, ConvNeXt, LoRA & Ensemble definitions
│   ├── training/                 # Standalone script-only training loops & LLRD
│   ├── eval/                     # Evaluation routines, metrics & benchmarking
│   └── utils/                    # Seeding, device management & logging helpers
│
├── scripts/                      # Standalone evaluation & analysis execution scripts
│   └── eval_v5_weakfix_v3_report.py
│
├── notebooks/                    # Interactive evaluation & visualization notebooks
│   ├── coursework_deepfake.ipynb # Master live GPU evaluation & ViT vs ConvNeXt benchmark (35 cells)
│   ├── coursework_eda.ipynb# Master training dataset EDA, FFT & taxonomy census (24 cells)
│   ├── predict_image.ipynb       # Interactive single-image prediction playground
│   └── archived/                 # Historical exploratory notebooks (00 to 15)
│
├── experiments/
│   ├── checkpoints/              # Model weights (best_model_v3.pt, convnext_weakfix_v3.pt)
│   ├── results/                  # Evaluation reports, JSON results & CSV tables
│   ├── plots/                    # Confusion matrices, ROC curves & attention maps
│   └── runs/                     # Training run logs & telemetry
│
└── tests/                        # Automated smoke tests & data verification tests
```

---

## 🔒 Mandatory Repository Rules

1. **No New Root Folders:** Never create arbitrary top-level directories without explicit approval.
2. **Immutable Raw Data:** `data/raw/` is strictly read-only.
3. **No Training in Notebooks:** Interactive notebooks are strictly for evaluation, metric visualization, and single-image testing. All training must occur via `src/training/` scripts.
4. **Frozen Checkpoints:** Checkpoints `experiments/checkpoints/best_model_v3.pt` and `experiments/checkpoints/convnext_weakfix_v3.pt` are frozen.
