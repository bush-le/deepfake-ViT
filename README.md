# deepfake-ViT: DINOv3 ViT vs ConvNeXt Deepfake Benchmark

- **Motivation/Background**: Provide a production-grade, reproducible research repository comparing Vision Transformers (Meta DINOv3 ViT-S/16) against Convolutional Networks (ConvNeXt-Tiny) on face deepfake detection across 44 manipulation methods.
- **Purpose**: Serve as the primary entry point and high-level architectural documentation for the deepfake-ViT project.
- **Overview Pipeline**: DINOv3 ViT & ConvNeXt training and evaluation pipeline on 44-method zero-leakage benchmarks.
- **Detailed Plan**: §1 Overview & Highlights; §2 Benchmark Performance; §3 Repository Structure; §4 Quickstart & Setup; §5 Training & Evaluation SOP; §6 Results & Notebooks.
- **References**: `src/`, `docs/`, `experiments/`, `notebooks/`, `requirements.txt`.
- **Created**: 2026-08-10T23:29:24+07:00
- **Last Updated**: 2026-09-08T10:15:00+07:00

---

Academic Coursework & Research Project: Detecting facial deepfakes using Vision Transformers (DINOv3) — self-supervised ViT developed by Meta AI.

---

## 📚 Documentation & Research Reports (`docs/`)

All human-readable technical documentation, academic research reports, and theoretical guides are organized in the [`docs/`](docs/) directory:
- [**`docs/DEEPFAKE_FORENSICS_REPORT.md`**](docs/DEEPFAKE_FORENSICS_REPORT.md): 🔬 Master Forensic & Experimental Research Report (207k Census, 16 EDA techniques, 4-Model Per-Method Benchmarks).
- [**`docs/THEORY_AND_MODEL_COMPARISON.md`**](docs/THEORY_AND_MODEL_COMPARISON.md): 📐 Theoretical Foundations & ViT vs. CNN Inductive Bias Analysis.
- [**`docs/MODELS.md`**](docs/MODELS.md): 🧠 Model Architecture Specifications, Checkpoints & Parameter Counts.
- [**`docs/RUNPOD.md`**](docs/RUNPOD.md): 🚀 GPU Infrastructure & Cloud Training Execution Guide.
- [**`docs/DESCRIPTION_NOTES.md`**](docs/DESCRIPTION_NOTES.md): 📝 High-Level Project & Dataset Synthesis Notes.

---

## 📁 Repository Directory Structure

```
deepfake-ViT/
├── .venv/              # Virtual environment (Python ≥ 3.10, standardized on 3.11/3.14)
├── docs/               # Human-readable documentation, reports, and theoretical guides
├── agents/             # Agent AI knowledge base (architecture, rules, phases, progress, experiments)
├── configs/            # Experiment configuration files (YAML)
├── data/
│   ├── raw/            # Raw datasets (DF40, FaceForensics++, Celeb-DF, etc.)
│   ├── processed/      # Processed data (cropped, aligned, resized faces)
│   ├── external/       # External datasets & benchmarks
│   ├── splits/         # Authoritative CSV splits (Train v3 Clean, Val v5 Boost, Test Bal 21k, Test Full 50k)
│   └── test/           # Local test images
├── src/                # Primary Python package & core implementations
│   ├── data/           # Dataset loaders, transforms, split & build utilities
│   ├── models/         # DINOv3 ViT-Small/16, DINOv3 ConvNeXt-Tiny, LoRA adapters
│   ├── training/       # Training loops, loss weighting, fine-tuning scripts
│   ├── eval/           # Evaluation harnesses, benchmark runners, metric aggregators
│   ├── experiments/    # Comparison scripts, figure generators, statistical analyzers
│   └── utils/          # Logging, hardware helpers, shell utilities
├── experiments/
│   ├── checkpoints/    # Frozen model weights (best_model_v3.pt, convnext_weakfix_v3.pt)
│   ├── results/        # Evaluation reports, JSON benchmarks, metrics, research docs
│   ├── plots/          # Generated figures, charts, and visualizations
│   └── runs/           # Run logs and training history
├── notebooks/          # Authoritative Coursework Notebooks
│   ├── coursework_eda.ipynb       # Master Multi-Split EDA & Physical Forensics (46 cells (20 sections))
│   ├── coursework_deepfake.ipynb  # Master Benchmark, ViT vs. CNN & Error Analysis (35 cells)
│   └── predict_image.ipynb        # Interactive Single-Image Deepfake Predictor (9 cells)
└── tests/              # Unit and smoke test suite
```

---

## 🚀 Environment Setup

```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
source .venv/bin/activate

# 1) PyTorch + Torchvision with CUDA 12.4 index (installed separately):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
#   (DINOv3 requires PyTorch >= 2.7.1 / 2.6.0; on macOS: pip install torch torchvision)

# 2) Remaining Python dependencies (pinned reproducible dependencies):
pip install -r requirements.lock.txt
#   (or `pip install -r requirements.txt` to pull latest valid floor releases)
```

---

## 📓 Primary Coursework Notebooks

1. [**`notebooks/coursework_eda.ipynb`**](notebooks/coursework_eda.ipynb) *(46 cells (20 sections))*:
   - **Master Exploratory Data Analysis (EDA)** across **207,414 image samples** (Train 129.8k, Val 6.0k, Test Bal 21.4k, Test Full 50.0k).
   - Complete **54-Methods Taxonomy** across 5 Generative Paradigms + Real Domains.
   - Physical & forensic signal analysis: 2D FFT Power Spectrum, 1D Radial PSD decay curves, High-Pass noise residuals, GLCM skin texture metrics, and Kolmogorov-Smirnov (KS-Test) statistical hypothesis verification.
   - Certified 4-Tier Zero-Leakage audit (0 path overlap, 127k MD5 hashes).

2. [**`notebooks/coursework_deepfake.ipynb`**](notebooks/coursework_deepfake.ipynb) *(35 Cells)*:
   - **Master Comparative Benchmark & Evaluation** running live GPU inference.
   - Head-to-head evaluation of **Meta DINOv3 ViT-Small/16** vs. **Meta DINOv3 ConvNeXt-Tiny** vs. **Joint Ensemble** on the 44-Methods Zero-Leakage test suite.
   - Post-training visualizations: Side-by-side Confusion Matrices, Probability Density Distributions, Tri-Curve Suite (ROC, PR, ECE Calibration), 5-Category Breakdown, Horizontal 44-Methods Ranking, and Inductive Bias Scatter Correlation.
   - Deep error diagnostics, Youden's J Index threshold optimization ($\tau^*$), and Top Visual Error Gallery.

3. [**`notebooks/predict_image.ipynb`**](notebooks/predict_image.ipynb) *(9 Cells)*:
   - **Interactive Single-Image Predictor** for user-supplied face images.

---

## 🛠️ Key CLI Commands (Local & Server Execution)

All commands are executed from the **repository root directory**. Each script provides a `--help` flag detailing arguments. Default device flag: `--device auto` (CUDA if available, otherwise MPS or CPU).

### Smoke & Unit Tests

```bash
python -m pytest tests/                 # Structural smoke and unit tests
```

### Data Pipeline & Dataset Building

```bash
# Build balanced subsets
python src/data/build_df40_balanced.py --n 750

# Test set construction & restructuring
python src/data/build_test_data.py --n 1000
python src/data/restructure_test_data_v3.py
python src/data/split_dataset.py        # Generate train/val/test CSV splits
```

### Model Training & Fine-Tuning

```bash
# Full fine-tune DINOv3 ViT (backbone + head)
python src/training/train.py \
  --train-csv data/splits/train_v5_weakfix_v3.csv \
  --val-csv data/splits/val_v5_combined_universal_kaggle_boost.csv \
  --epochs 5 --batch-size 32 --amp --num-workers 0

# LoRA parameter-efficient fine-tuning
python src/training/finetune_lora.py \
  --train-csv data/splits/train_v5_weakfix_v3.csv \
  --val-csv data/splits/val_v5_combined_universal_kaggle_boost.csv \
  --lora-rank 16 --lora-alpha 32 --amp

# ViT vs. CNN comparison training
python src/training/finetune_compare.py --model-type vit --amp
python src/training/finetune_compare.py --model-type cnn --amp
```

### Evaluation & Inference

```bash
# Evaluate on 44 methods zero-leakage benchmark
python scripts/eval_v5_weakfix_v3_report.py

# Single-image CLI prediction
python src/eval/predict.py --image data/test/sample.jpg --device cuda

# Latency and throughput benchmarking
python src/eval/benchmark_inference.py --device cuda
```

---

## 📌 Technical Notes

- **Meta DINOv3 Backbone**: Requires PyTorch ≥ 2.7.1 / 2.6.0 and `timm` ≥ 1.0.20 (or HuggingFace `transformers` ≥ 4.56). Pretrained weights are downloaded from HuggingFace Hub (`facebook/dinov3-*`).
- **Python Compatibility**: Standardized on Python 3.11 / 3.14. Multi-processing defaults to `num_workers=0` in interactive/Jupyter environments to prevent POSIX `forkserver` unpickling conflicts.
- **Data Paths**: Dataset CSV files contain relative or flexible path structures. The dynamic `resolve_path()` helper automatically locates local desktop and workspace image directories.
- **Reproducibility**: Pinned dependencies are maintained in `requirements.lock.txt`.
