# 📚 Project Documentation Portal (`docs/`)

Welcome to the human-readable documentation portal for the **High-Generalization Facial Deepfake Forensics & Detection** project (`deepfake-ViT`).

---

## 📑 Core Documentation Index

| Document | Description | Key Focus & Scope |
| :--- | :--- | :--- |
| [**`DEEPFAKE_FORENSICS_REPORT.md`**](DEEPFAKE_FORENSICS_REPORT.md) | **🔬 Master Forensic & Experimental Research Report** | Comprehensive 5-part research report covering 207k dataset census, 16 physical forensic EDA techniques, zero-leakage firewall protocols, 4-model per-method diagnostic charts (38 methods), and scorecard benchmarks. |
| [**`THEORY_AND_MODEL_COMPARISON.md`**](THEORY_AND_MODEL_COMPARISON.md) | **📐 Theoretical Foundations & Architectural Analysis** | Mathematical analysis of inductive biases (Global Self-Attention vs. Depthwise Convolutions), computational scaling laws, PyTorch implementations, and receptive field dynamics. |
| [**`MODELS.md`**](MODELS.md) | **🧠 Model Architecture & Checkpoint Specifications** | Complete inventory of checkpoints (`plus_v3_s1_best.pt`, `convnext_weakfix_v3.pt`), parameter counts, latency metrics (FPS), and classification head structures. |
| [**`THEORY_AND_MODEL_COMPARISON.pdf`**](THEORY_AND_MODEL_COMPARISON.pdf) | **📑 Academic Research Paper (PDF)** | Formatted academic publication PDF with theoretical derivations, math models, and full architecture comparisons. |
| [**`presentation.tex`**](presentation.tex) | **📊 Presentation Slides (LaTeX Beamer)** | Conference / coursework presentation slide deck covering theoretical foundations, model architectures, and results. |
| [**`RUNPOD.md`**](RUNPOD.md) | **🚀 GPU Infrastructure & Cloud Training Runbook** | Step-by-step operational guide for provisioning RunPod instances, mounting Hugging Face datasets via rclone, running AMP training loops, and synchronizing artifacts. |
| [**`DESCRIPTION_NOTES.md`**](DESCRIPTION_NOTES.md) | **📝 Project Synthesis & Evaluation Notes** | High-level summary of model performance, dataset splits, and benchmarking methodologies. |

---

## 📊 Evaluation Notebooks Linkage

The live interactive Jupyter Notebooks corresponding to these reports are located in [`notebooks/`](../notebooks/):
- [`deepfake_forensics_report.ipynb`](../notebooks/deepfake_forensics_report.ipynb) — Master research evaluation notebook with embedded 5-chart per-method breakdown.
- [`coursework_deepfake_plus_v3_s1_best.ipynb`](../notebooks/coursework_deepfake_plus_v3_s1_best.ipynb) — GPU benchmark and error diagnostic suite for ViT-Plus s1_best vs. ConvNeXt.
- [`coursework_eda.ipynb`](../notebooks/coursework_eda.ipynb) — Multi-dimensional exploratory data analysis and optical signal decomposition.
- [`predict_image.ipynb`](../notebooks/predict_image.ipynb) — Single-image interactive inference playground.
