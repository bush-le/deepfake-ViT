# PR #5 Audit, Conflict Resolution, and Integration Process Report

- **Motivation/Background**: Pull Request #5 ("Enhance DINOv3 ViT Training and Evaluation Pipelines" from `bush-le:main` to `zombieTDV:main`, commit `46d073a`) introduced 21 commits adding multi-model benchmark evaluation, coursework reporting notebooks, and enhanced DINOv3 attention mechanisms. However, divergence between `main` and PR #5 resulted in 32 file merge conflicts and critical runtime regressions.
- **Purpose**: Provide an immutable, fully traceable record of all actions taken to audit, resolve merge conflicts, diagnose bugs, apply stability fixes, verify test suites, and merge PR #5 into `main`.
- **Overview Pipeline**: Fetch PR #5 branch -> isolate on integration branch `merge/pr-5-audit` -> resolve 32 file conflicts with technical rationales -> execute static/import audit -> identify 5 bugs -> apply stability and portability fixes -> run full test suite -> fast-forward merge into `main` (commit `ea6a2d1`).
- **Detailed Plan**: §1 Executive Summary & Integration Timeline; §2 32-File Conflict Resolution Matrix; §3 Bug Audit & Resolution Lifecycle (conforming to CODEBASE_AUDIT_TEMPLATE.md); §4 System Verification & Test Evidence; §5 Post-Merge Status & File Census.
- **References**: `docs/README.md`, `agents/rules/CODEBASE_AUDIT.md`, `agents/rules/FOLDER_STRUCTURE.md`, `requirements.txt`.
- **Created**: 2026-09-08T10:10:30+07:00
- **Last Updated**: 2026-09-08T10:10:30+07:00

---

## 1. Executive Summary & Integration Timeline

### 1.1 PR Overview
- **Pull Request ID**: GitHub PR #5
- **Source Branch / Fork**: `bush-le:main` (HEAD commit: `46d073a`)
- **Target Branch**: `zombieTDV:main` (base commit: `7fdd3ad`)
- **Title**: *Enhance DINOv3 ViT Training and Evaluation Pipelines*
- **Scope**: Added 21 commits introducing coursework benchmarking across 44 deepfake generation methods, ConvNeXt vs. ViT comparison experiments, probability density visualization scripts, and evaluation notebooks.

### 1.2 Step-by-Step Integration Timeline

1. **Pre-Merge State Audit**:
   - Inspected `zombieTDV:main` commit history and verified pre-merge baseline: `pytest tests/` passed (7 passed, 7 skipped).
   - Fetched `bush-le:main` as remote tracking ref `pr-5`.

2. **Integration Branch Isolation**:
   - Created dedicated integration branch `merge/pr-5-audit` branching off `main`.
   - Executed `git merge --no-ff pr-5`, surfacing 32 file merge conflicts across `.gitignore`, documentation, notebooks, training scripts, data preparation pipelines, and model definitions.

3. **Conflict Resolution Strategy**:
   - Root files (`.gitignore`, `RUNPOD.md`, `README.md`): Preserved `main` ignoring patterns; deleted root `RUNPOD.md` in favor of `docs/RUNPOD.md`; adopted `pr-5` documentation index.
   - Notebooks (10 files): Checked out `--theirs` (`pr-5`) to incorporate real execution outputs, cells, and visualization outputs.
   - Data & Training Scripts (10 files): Checked out `--ours` (`main`) to preserve portable path resolution (`Path(__file__).resolve().parents[2]`, `os.getenv("DF40_ROOT")`) and file existence guards added in commit `7fdd3ad`.
   - Model Architecture (`src/models/dinov3_vit.py`): Executed a composite merge preserving both `pr-5` FlashAttention (`F.scaled_dot_product_attention`) and `main`'s `EnhancedDinoViTClassifier` + `**kwargs` support.
   - Coursework Evaluation Plots (7 files): Checked out `--theirs` (`pr-5`) to retain latest model benchmarking charts.
   - Committed conflict resolution as `03cc98f`.

4. **Static & Runtime Bug Audit**:
   - Ran automated syntax compilation (`py_compile`) across all 99 Python files.
   - Performed dynamic module import audit across all `src/` modules.
   - Scanned all scripts for hardcoded Linux paths (`/workspace/...`) and unshielded `.cuda()` calls.
   - Uncovered 5 distinct bugs across dependencies, script portability, device safety, and module-level execution.

5. **Stability & Portability Remediation**:
   - Installed missing `pandas>=2.0` into `.venv`.
   - Refactored 8 scripts in `scripts/` to use relative repository paths and `os.getenv("DF40_ROOT")`.
   - Replaced unshielded `torch.autocast(device_type='cuda')` in `src/eval/eval_exp02_full.py` with device-aware guards.
   - Encapsulated module-level side effects and dataset sampling logic inside `def main():` entrypoints across 5 scripts.
   - Committed fixes as `ea6a2d1`.

6. **Verification & Merge to `main`**:
   - Re-ran compilation: 99/99 files compiled cleanly.
   - Re-ran import audit: 100% of `src/` modules imported without errors or side effects.
   - Re-ran test suite: `pytest tests/ -v` passed (7 passed, 7 skipped).
   - Checked out `main` and executed fast-forward merge: `git merge --ff-only merge/pr-5-audit`.
   - Removed temporary branch `merge/pr-5-audit`.

---

## 2. 32-File Conflict Resolution Matrix

The table below documents the exact path, conflict classification, chosen resolution source, technical rationale, and subsequent stabilization fix for all 32 conflicted files during the merge of PR #5.

| # | Conflicted Filepath | Conflict Type | Resolution Source | Technical Rationale | Post-Merge Fix Applied |
| :- | :--- | :--- | :--- | :--- | :--- |
| 1 | `.gitignore` | Both modified | `main` (`--ours`) | Retained `main` rules ignoring large data directories (`data_train/`, `data_train_local/`, `test_data_v3/`, `experiments/results/finetune/`). | None (rules intact). |
| 2 | `RUNPOD.md` | Deleted in main, modified in pr-5 | `git rm` (Superseded) | Redundant with `docs/RUNPOD.md`. Removed root copy to prevent root directory pollution. | File removed from repo root. |
| 3 | `README.md` | Both modified | `pr-5` (`--theirs`) | Adopted `pr-5` updated documentation index, quick-start commands, and experiment overview. | Verified relative links. |
| 4 | `agents/rules/FOLDER_STRUCTURE.md` | Both modified | `pr-5` (`--theirs`) | Adopted latest folder structure reference from PR #5. | Scheduled for template alignment. |
| 5 | `src/models/dinov3_vit.py` | Both modified | Composite (`ours` + `theirs`) | Kept `EnhancedDinoViTClassifier` (required by `train_exp02..04`) while integrating `F.scaled_dot_product_attention` and `**kwargs`. | Combined signatures in `build_dinov3_classifier`. |
| 6 | `src/data/build_coursework_44methods_test_set.py` | Both added | `main` (`--ours`) | `main` had portable path resolution and file guards; `pr-5` hardcoded `/workspace/...` Linux paths. | Encapsulated in `def main():` and guarded empty candidates. |
| 7 | `src/data/build_coursework_scaled_test_set.py` | Both added | `main` (`--ours`) | `main` resolved paths via `parents[2]`; `pr-5` assumed local RunPod disk paths. | Encapsulated in `def main():` and guarded empty candidates. |
| 8 | `src/data/build_dataset_v3_clean.py` | Both added | `main` (`--ours`) | Preserved portable environment variable `DF40_ROOT` and graceful fallback if raw data is missing. | Main guard already present. |
| 9 | `src/data/build_expanded_44method_test_set.py` | Both added | `main` (`--ours`) | Retained portable path resolution; avoided hardcoded host paths. | Main guard already present. |
| 10 | `src/data/build_kaggle_midjourney_boost.py` | Both added | `main` (`--ours`) | Preserved `PROJECT_ROOT` and `DATA_ROOT` dynamic discovery. | Main guard already present. |
| 11 | `src/data/build_universal_balanced_v4.py` | Both added | `main` (`--ours`) | Retained portable path resolution. | Encapsulated in `def main():` and guarded zero division. |
| 12 | `src/data/evaluate_expanded_test_sets.py` | Both added | `main` (`--ours`) | Kept portable dataset paths. | Deferred model and checkpoint loading into `get_model()`. |
| 13 | `src/data/verify_zero_leakage.py` | Both added | `main` (`--ours`) | Maintained portable paths and existence checks. | Main guard already present. |
| 14 | `src/training/train_kaggle_boost.py` | Both added | `main` (`--ours`) | Preserved portable paths and CLI argument defaults. | Installed `pandas` in `.venv`. |
| 15 | `src/training/train_v5_combined.py` | Both added | `main` (`--ours`) | Kept portable paths and checkpointing routines. | Installed `pandas` in `.venv`. |
| 16 | `notebooks/coursework_deepfake.ipynb` | Both added | `pr-5` (`--theirs`) | PR #5 contained full coursework evaluation outputs, plots, and analysis tables. | Checked path references. |
| 17 | `notebooks/coursework_eda.ipynb` | Both added | `pr-5` (`--theirs`) | PR #5 had executed exploratory data analysis across 44 methods. | Validated cell execution JSON. |
| 18 | `notebooks/predict_image.ipynb` | Both added | `pr-5` (`--theirs`) | PR #5 provided single-image inference and interactive Grad-CAM/attention visualizer. | Validated cell execution JSON. |
| 19 | `notebooks/archived/05_midjourney_vs_traditional_deepfakes_eda.ipynb` | Both added | `pr-5` (`--theirs`) | Retained archival experiment record from PR #5. | Verified valid notebook JSON. |
| 20 | `notebooks/archived/06_comprehensive_dataset_splits_and_method_distribution_audit.ipynb` | Both added | `pr-5` (`--theirs`) | Retained archival split census notebook from PR #5. | Verified valid notebook JSON. |
| 21 | `notebooks/archived/08_data_leakage_audit_and_eda_exp02.ipynb` | Both added | `pr-5` (`--theirs`) | Retained archival leakage audit notebook from PR #5. | Verified valid notebook JSON. |
| 22 | `notebooks/archived/09_kaggle_dataset_eda_and_midjourney_enhancement.ipynb` | Both added | `pr-5` (`--theirs`) | Retained archival Kaggle Midjourney enhancement notebook from PR #5. | Verified valid notebook JSON. |
| 23 | `notebooks/archived/10_shared_zero_leakage_audit_and_eda_verification.ipynb` | Both added | `pr-5` (`--theirs`) | Retained archival zero-leakage verification notebook from PR #5. | Verified valid notebook JSON. |
| 24 | `notebooks/archived/14_master_workspace_data_inventory_and_unified_split_eda.ipynb` | Both added | `pr-5` (`--theirs`) | Retained master workspace data census notebook from PR #5. | Verified valid notebook JSON. |
| 25 | `notebooks/archived/15_expanded_46methods_test_set_eda.ipynb` | Both added | `pr-5` (`--theirs`) | Retained 46-method test set EDA notebook from PR #5. | Verified valid notebook JSON. |
| 26 | `experiments/results/courseWorkCheck/category_performance_breakdown.png` | Both modified | `pr-5` (`--theirs`) | Retained latest rendered category performance chart. | Binary asset staged. |
| 27 | `experiments/results/courseWorkCheck/cm_comparison_test_coursework_balanced.png` | Both modified | `pr-5` (`--theirs`) | Retained updated confusion matrix comparison plot. | Binary asset staged. |
| 28 | `experiments/results/courseWorkCheck/per_method_accuracy_horizontal_ranking.png` | Both modified | `pr-5` (`--theirs`) | Retained updated per-method accuracy horizontal ranking. | Binary asset staged. |
| 29 | `experiments/results/courseWorkCheck/probability_density_distribution.png` | Both modified | `pr-5` (`--theirs`) | Retained latest prediction probability density distribution chart. | Binary asset staged. |
| 30 | `experiments/results/courseWorkCheck/roc_pr_calibration_triad.png` | Both modified | `pr-5` (`--theirs`) | Retained latest ROC, PR, and Calibration triad curves. | Binary asset staged. |
| 31 | `experiments/results/courseWorkCheck/threshold_sensitivity_curves.png` | Both modified | `pr-5` (`--theirs`) | Retained latest threshold sensitivity sweep visualization. | Binary asset staged. |
| 32 | `experiments/results/courseWorkCheck/vit_vs_convnext_scatter_correlation.png` | Both modified | `pr-5` (`--theirs`) | Retained latest ViT vs ConvNeXt inductive bias correlation scatter plot. | Binary asset staged. |

---

## 3. Bug Audit & Resolution Lifecycle

This section conforms to the standard `CODEBASE_AUDIT_TEMPLATE.md` finding lifecycle. Every issue discovered during the merge audit is cataloged with severity, status (`RESOLVED`), evidence, and permanent resolution details.

### 3.1 Findings Summary Table

| Finding ID | Category | Severity | Description | Status | Resolution Summary |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **BUG-01** | Dependency | HIGH | `pandas` missing from active virtual environment despite declaration in `requirements.txt`. | **RESOLVED** | Installed `pandas-3.0.5` into `.venv` using explicit virtualenv pip path. |
| **BUG-02** | Portability | HIGH | 8 scripts in `scripts/` hardcoded Linux RunPod absolute paths (`/workspace/hoangtuan/...`, `/workspace/data/...`). | **RESOLVED** | Refactored paths to dynamic resolution using `Path(__file__).resolve().parents[1]` and `os.getenv("DF40_ROOT")`. |
| **BUG-03** | Device Safety | MEDIUM | Unshielded `torch.autocast(device_type='cuda')` in `src/eval/eval_exp02_full.py` crashes on CPU systems. | **RESOLVED** | Replaced with `torch.autocast(device_type=device.type, enabled=(device.type == 'cuda'))`. |
| **BUG-04** | Execution Safety | HIGH | Top-level execution in `src/data/build_universal_balanced_v4.py` caused `ZeroDivisionError` on module import. | **RESOLVED** | Encapsulated execution in `def main():` and added empty candidate / zero method guards. |
| **BUG-05** | Modularity | MEDIUM | Multiple `build_*.py` and evaluation scripts executed heavy side effects at module import time. | **RESOLVED** | Wrapped all top-level side effects inside `def main():` entrypoints guarded by `if __name__ == '__main__':`. |

---

### 3.2 Detailed Finding Reports

#### Finding BUG-01: Missing `pandas` Dependency in Runtime Environment
- **Severity**: HIGH
- **Status**: **RESOLVED**
- **Affected Files**: All training and data preparation scripts (`src/training/*.py`, `src/data/*.py`).
- **Description**: Running automated import checks failed across all training and data scripts with `ModuleNotFoundError: No module named 'pandas'`. While `requirements.txt` correctly declared `pandas>=2.0`, the virtual environment had not installed it.
- **Resolution & Evidence**: Installed `pandas-3.0.5` and `tzdata-2026.3` into `C:\document\Study documents\deepfake-ViT\.venv`. Verified via `python -c "import pandas; print(pandas.__version__)"` returning `3.0.5`.

#### Finding BUG-02: Hardcoded RunPod Paths Across `scripts/*.py`
- **Severity**: HIGH
- **Status**: **RESOLVED**
- **Affected Files**:
  - `scripts/audit_data_leakage.py`
  - `scripts/build_finetune_v5_weakfix.py`
  - `scripts/eval_v5_weakfix_report.py`
  - `scripts/eval_v5_weakfix_v3_report.py`
  - `scripts/expand_faceswap_v3.py`
  - `scripts/finetune_v5_weakfix.py`
  - `scripts/finetune_v5_weakfix_v3.py`
  - `scripts/prepare_celebvhq_frames.py`
- **Description**: PR #5 authored scripts directly on RunPod using hardcoded Linux strings (`/workspace/hoangtuan/deepfake-ViT`, `/workspace/data/`). Executing these scripts on local workstations, CI/CD runners, or alternative cloud instances caused immediate `FileNotFoundError`.
- **Resolution & Evidence**: Refactored all paths to resolve relative to `Path(__file__).resolve().parents[1]` and dynamically check environment variables:
  ```python
  ROOT = Path(os.getenv("REPO_ROOT", Path(__file__).resolve().parents[1]))
  DATA = Path(os.getenv("DF40_ROOT", ROOT / "data"))
  ```
  Ran `scan_paths_and_devices.py`: confirmed 0 hardcoded Linux paths remain.

#### Finding BUG-03: Unshielded CUDA Autocast in Evaluation Pipeline
- **Severity**: MEDIUM
- **Status**: **RESOLVED**
- **Affected Files**: `src/eval/eval_exp02_full.py`
- **Description**: Line 125 hardcoded `with torch.autocast(device_type='cuda', dtype=torch.bfloat16):`. When run on machines without an active NVIDIA CUDA GPU, PyTorch raised a fatal `RuntimeError: User specified an unsupported autocast device_type 'cuda' for CPU`.
- **Resolution & Evidence**: Updated autocast context to be device-aware:
  ```python
  with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == 'cuda')):
  ```
  Verified module compiles and imports on CPU without errors.

#### Finding BUG-04: Top-Level `ZeroDivisionError` on Import in `build_universal_balanced_v4.py`
- **Severity**: HIGH
- **Status**: **RESOLVED**
- **Affected Files**: `src/data/build_universal_balanced_v4.py`
- **Description**: Lines 182-207 executed sampling logic at import time. When dataset folders were absent on the local system, `active_fake_methods` evaluated to an empty list `[]`. Line 185 executed `base_quota = target_total // len(methods)`, raising `ZeroDivisionError: integer division or modulo by zero` simply by importing the module.
- **Resolution & Evidence**: Wrapped entire script logic in `def main():` with an explicit guard:
  ```python
  active_fake_methods = sorted([m for m in all_fake_pools.keys() if len(all_fake_pools[m]) >= 50])
  if not active_fake_methods:
      print("⚠️ No active fake methods found with >= 50 samples. Exiting.")
      return
  ```
  Verified safe import via `python -c "import src.data.build_universal_balanced_v4"`.

#### Finding BUG-05: Missing Main Guards Across `src/data/` and `src/eval/`
- **Severity**: MEDIUM
- **Status**: **RESOLVED**
- **Affected Files**:
  - `src/data/build_coursework_44methods_test_set.py`
  - `src/data/build_coursework_scaled_test_set.py`
  - `src/data/build_domain_balanced_v2.py`
  - `src/data/evaluate_expanded_test_sets.py`
  - `src/eval/eval_exp02_full.py`
  - `src/experiments/assemble_attention_figure.py`
  - `src/experiments/model_param_count.py`
- **Description**: Files located in `src/` are treated as library modules by Python tools, IDEs, and test runners. Running hardware initialization, dataset reads, and plotting at the module top level led to side effects and crashes when testing or importing.
- **Resolution & Evidence**:
  - Wrapped dataset builds in `def main():` and `if __name__ == '__main__': main()`.
  - Added empty candidate checks preventing `KeyError: 'label'` when candidate lists are empty.
  - Deferred model loading in `evaluate_expanded_test_sets.py` into `get_model()`.
  - Wrapped `pytorchvideo` import in `model_param_count.py` in `try...except ImportError`.
  - Ran comprehensive import audit: all `src` modules now import cleanly in isolation.

---

## 4. System Verification & Test Evidence

Following conflict resolution and bug remediation on `merge/pr-5-audit`, four automated quality gates were executed before merging into `main`.

### 4.1 Gate 1: Syntax & Compilation Audit
- **Command**: `audit_imports_and_syntax.py` (`py_compile` on all `.py` files)
- **Scope**: 99 Python files across `src/`, `scripts/`, and `tests/`
- **Result**:
  ```text
  Checking syntax/compilation for 99 Python files...
  All Python files compiled successfully!
  ```

### 4.2 Gate 2: Dynamic Module Import Audit
- **Command**: `audit_imports_and_syntax.py` (`importlib.import_module` on all `src/` modules)
- **Scope**: All 27 modules under `src/`
- **Result**:
  ```text
  Checking imports for src modules...
  All src modules imported successfully!
  ```

### 4.3 Gate 3: Path & Device Portability Scan
- **Command**: `scan_paths_and_devices.py` (regex scan for `/workspace/`, `/root/`, unshielded `.cuda()`)
- **Scope**: All files in `src/` and `scripts/`
- **Result**:
  ```text
  Scanning for hardcoded absolute Linux paths and unshielded .cuda()...
  0 hardcoded Linux paths found.
  0 unshielded .cuda() calls found.
  ```

### 4.4 Gate 4: Test Suite Execution
- **Command**: `pytest tests/ -v`
- **Environment**: Python 3.12.10 (Windows)
- **Result**:
  ```text
  ============================= test session starts =============================
  platform win32 -- Python 3.12.10, pytest-9.1.1, pluggy-1.6.0
  rootdir: C:\document\Study documents\deepfake-ViT
  configfile: pytest.ini
  collected 14 items

  tests\test_data_prep.py sssssss                                          [ 50%]
  tests\test_smoke.py .......                                              [100%]

  ======================== 7 passed, 7 skipped in 4.74s =========================
  ```

---

## 5. Post-Merge Repository State

### 5.1 Git Commit Provenance
- `03cc98f`: `merge: resolve conflicts between main and PR #5 (bush-le:main)`
- `ea6a2d1`: `fix: resolve hardcoded paths, unshielded execution, and import bugs found in audit`
- Fast-forward merge to `main`: `main` is now at commit `ea6a2d1`.

### 5.2 Working Tree Status
- Working tree clean, 0 uncommitted changes.
- Virtual environment verified and active.
- Ready for formal refactoring to `Deep_learning_template` Archetype A standard.
