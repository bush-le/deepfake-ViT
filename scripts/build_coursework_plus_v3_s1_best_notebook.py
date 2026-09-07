#!/usr/bin/env python3
"""
Script to build, execute and extract plots for notebooks/coursework_deepfake_plus_v3_s1_best.ipynb
Evaluates Meta DINOv3 ViT-Small/16 Plus (plus_v3_s1_best.pt) vs Meta DINOv3 ConvNeXt-Tiny (convnext_weakfix_v3.pt)
"""

import os, sys, json, copy, base64, io, shutil
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOKS_DIR = REPO_ROOT / "notebooks"
OUT_DIR = REPO_ROOT / "experiments/results/plus_v3_s1_best_eval"
OUT_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_EXTRACTED = REPO_ROOT / "experiments/plots/notebook_extracted"
PLOTS_EXTRACTED.mkdir(parents=True, exist_ok=True)

# Pre-copy existing s1_best npz caches if available to speed up execution
src_cache = REPO_ROOT / "experiments/results/courseWorkCheck"
for npz_file in ["vit_test_bal.npz", "vit_test_full.npz", "cnn_test_bal.npz", "cnn_test_full.npz"]:
    if (src_cache / npz_file).exists() and not (OUT_DIR / npz_file).exists():
        shutil.copy(src_cache / npz_file, OUT_DIR / npz_file)
        print(f"📦 Copied cache {npz_file} to {OUT_DIR}")

# Load base notebook
base_nb_path = NOTEBOOKS_DIR / "coursework_deepfake.ipynb"
with open(base_nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

# Update notebook metadata and header
title_md = """# 🎓 CourseWork Deepfake Benchmark — Master Evaluation & Post-Training Visual Diagnostics (Meta DINOv3 ViT-Small/16 Plus s1_best vs ConvNeXt)

## 🔬 Subtitle & Objective
**Objective**: Unified evaluation harness, live GPU inference, and comparative benchmarking between **Meta DINOv3 ViT-Small/16 Plus** (`plus_v3_s1_best.pt`, SwiGLU Gated MLP, 28.69M params) and **Meta DINOv3 ConvNeXt-Tiny** (`convnext_weakfix_v3.pt`, Modern CNN Baseline, 28.12M params) on the expanded **44-Methods Zero-Leakage Test Suites** (**20,846 Balanced** and **48,064 Full Suite**). Features comprehensive post-training visual diagnostics (ROC, PR, KDE, Calibration, 5-Category breakdown, Horizontal ranking, Inductive bias scatter, Youden J optimization, and Top Visual Error Gallery).

---

### 🗺️ Evaluation Roadmap

| Step | Section Title | Description & Methodological Scope | Import Path / Module |
| :---: | :--- | :--- | :---: |
| **0** | **Setup & Environment** | GPU configuration, path resolution, batch size & multi-worker safety settings. | [`src/utils/seeding.py`](../src/utils/seeding.py), [`src/utils/run_logger.py`](../src/utils/run_logger.py) |
| **1** | **Data & Zero-Leakage Audit** | Load 3 splits (Train 129.8k, Test Bal 20.8k, Test Full 48.1k), 3-tier zero-leakage verification. | [`data/splits/train_v5_weakfix_v3.csv`](../data/splits/train_v5_weakfix_v3.csv), [`data/splits/test_coursework_44methods_balanced_zero_leakage.csv`](../data/splits/test_coursework_44methods_balanced_zero_leakage.csv) |
| **2** | **Model Loading** | Load frozen weights: DINOv3 ViT-S/16 Plus (`plus_v3_s1_best.pt`) and ConvNeXt-Tiny (`convnext_weakfix_v3.pt`). | [`src/models/dinov3_vit.py`](../src/models/dinov3_vit.py), [`src/models/dinov3_convnext.py`](../src/models/dinov3_convnext.py) |
| **3** | **Live GPU Benchmarking** | Live inference on Test Balanced (20.8k) and Test Full (48.1k), side-by-side Confusion Matrices. | [`src/eval/evaluate.py`](../src/eval/evaluate.py), [`src/training/train.py`](../src/training/train.py) |
| **4** | **Post-Training Visualizations** | KDE score distribution, Tri-Curve Suite (ROC, PR, ECE), 5-Category breakdown, 44-Methods ranking, Inductive bias scatter. | [`src/experiments/make_report_figures.py`](../src/experiments/make_report_figures.py), [`src/experiments/compare_models.py`](../src/experiments/compare_models.py) |
| **5** | **Deep Error Diagnostics** | Threshold optimization (Youden J Index), hard case blindspot analysis, Top Visual Error Gallery (FN, FP). | [`src/eval/analyze_threshold.py`](../src/eval/analyze_threshold.py), [`src/eval/predict.py`](../src/eval/predict.py) |

---

### 📚 References & Documentation Links
- **Evaluation Specs**: [`EVAL.md`](../agents/phases/EVAL.md), [`EXP_04_COURSEWORK_44METHODS_BENCHMARK.md`](../agents/experiments/EXP_04_COURSEWORK_44METHODS_BENCHMARK.md)
- **EDA & Data Specs**: [`coursework_eda.ipynb`](coursework_eda.ipynb), [`DATA_PREP.md`](../agents/phases/DATA_PREP.md), [`EDA_DATA_INVENTORY.md`](../agents/EDA_DATA_INVENTORY.md)
- **Model Checkpoints**: [`plus_v3_s1_best.pt`](../experiments/checkpoints/plus_v3_s1_best.pt), [`convnext_weakfix_v3.pt`](../experiments/checkpoints/convnext_weakfix_v3.pt)"""

nb['cells'][0]['source'] = [line + "\n" for line in title_md.split("\n")]

# Adjust setup cell to output to experiments/results/plus_v3_s1_best_eval
setup_src = "".join(nb['cells'][2]['source'])
setup_src = setup_src.replace('OUT = HT / "experiments/results/courseWorkCheck"', 'OUT = HT / "experiments/results/plus_v3_s1_best_eval"')
setup_src = setup_src.replace("OUT = HT / 'experiments/results/courseWorkCheck'", 'OUT = HT / "experiments/results/plus_v3_s1_best_eval"')
nb['cells'][2]['source'] = [line + "\n" for line in setup_src.split("\n")]

# Also replace any other occurrences in code cells
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        c_src = "".join(cell['source'])
        if "courseWorkCheck" in c_src:
            c_src = c_src.replace("courseWorkCheck", "plus_v3_s1_best_eval")
            cell['source'] = [line + "\n" for line in c_src.split("\n")]

# Write initial notebook file
target_nb_path = NOTEBOOKS_DIR / "coursework_deepfake_plus_v3_s1_best.ipynb"
with open(target_nb_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print(f"✅ Created target notebook template {target_nb_path}")

# Run cells in python and update outputs
env_globals = {"__name__": "__main__", "__file__": str(target_nb_path)}
exec_count = 1

for idx, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        code_str = "".join(cell['source'])
        print(f"▶ Executing Cell {idx}...")
        
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        redirected_output = io.StringIO()
        redirected_error = io.StringIO()
        sys.stdout = redirected_output
        sys.stderr = redirected_error
        
        plt.close('all')
        cell_outputs = []
        try:
            def custom_display(*args, **kwargs):
                for a in args:
                    if isinstance(a, pd.DataFrame):
                        html_table = a.to_html()
                        text_table = a.to_string()
                        cell_outputs.append({
                            "data": {
                                "text/html": [line + "\n" for line in html_table.split("\n")],
                                "text/plain": [line + "\n" for line in text_table.split("\n")]
                            },
                            "metadata": {},
                            "output_type": "display_data"
                        })
                    elif hasattr(a, "_repr_html_"):
                        cell_outputs.append({
                            "data": {
                                "text/html": [a._repr_html_()],
                                "text/plain": [str(a)]
                            },
                            "metadata": {},
                            "output_type": "display_data"
                        })
                    else:
                        sys.stdout.write(str(a) + "\n")
            
            env_globals["display"] = custom_display
            
            exec(code_str, env_globals)
            
            figs = [plt.figure(i) for i in plt.get_fignums()]
            for fig in figs:
                buf = io.BytesIO()
                fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
                buf.seek(0)
                img_b64 = base64.b64encode(buf.read()).decode('utf-8')
                cell_outputs.append({
                    "data": {
                        "image/png": img_b64,
                        "text/plain": ["<Figure size ... with ... Axes>"]
                    },
                    "metadata": {},
                    "output_type": "display_data"
                })
            plt.close('all')
            
        except Exception as e:
            sys.stderr.write(f"Error in cell {idx}: {e}\n")
            print(f"❌ Error in cell {idx}: {e}")
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr
        
        stdout_val = redirected_output.getvalue()
        stderr_val = redirected_error.getvalue()
        
        final_outputs = []
        if stderr_val:
            final_outputs.append({
                "name": "stderr",
                "output_type": "stream",
                "text": [line + "\n" for line in stderr_val.split("\n") if line]
            })
        if stdout_val:
            final_outputs.append({
                "name": "stdout",
                "output_type": "stream",
                "text": [line + "\n" for line in stdout_val.split("\n") if line]
            })
        final_outputs.extend(cell_outputs)
        
        cell['outputs'] = final_outputs
        cell['execution_count'] = exec_count
        exec_count += 1

with open(target_nb_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print(f"🎉 Fully executed and saved {target_nb_path}")

# Extract images to plots_extracted
img_idx = 1
for cell_idx, cell in enumerate(nb['cells']):
    for out in cell.get('outputs', []):
        data = out.get('data', {})
        if 'image/png' in data:
            img_data = base64.b64decode(data['image/png'])
            fname1 = f"coursework_deepfake_plus_v3_s1_best_cell_{cell_idx}_img_{img_idx}.png"
            (PLOTS_EXTRACTED / fname1).write_bytes(img_data)
            fname2 = f"coursework_deepfake_cell_{cell_idx}_img_{img_idx}.png"
            (PLOTS_EXTRACTED / fname2).write_bytes(img_data)
            print(f"📸 Extracted {fname1} ({len(img_data):,} bytes)")
            img_idx += 1

print("✨ All tasks completed successfully!")
