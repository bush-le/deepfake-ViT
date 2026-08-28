# 🐛 Bug Reports & Troubleshooting Directory

This directory contains documented bug reports, root cause analyses, and verified resolution guides encountered during the `deepfake-ViT` project lifecycle.

---

## 🏗️ Bug Governance Lifecycle

```
Runtime Anomaly / Audit Detection
        ↓
Root Cause Analysis & Minimal Reproducer (agents/bugs/BUG_*.md)
        ↓
Code Patch & Defensive Guard Implementation
        ↓
Automated Test Verification & Smoke Test
        ↓
Status Update & Architecture Documentation Sync
```

---

## 📋 Bug Index

| Bug ID | Title & Summary | Component / Module | Severity | Status | Detailed Document |
| :---: | :--- | :--- | :---: | :---: | :--- |
| **`BUG-01`** | **Celeb-DF Test Set Label Inversion Fix** | `src/data/prepare_df40_splits.py` | Critical | Resolved ✅ | [`BUG_01_CELEBDF_TEST_LABEL_FIX.md`](BUG_01_CELEBDF_TEST_LABEL_FIX.md) |
| **`BUG-02`** | **Python 3.14+ Forkserver DataLoader Pickle Error** (`ImgDS`) | `notebooks/coursework_deepfake.ipynb` | High | Resolved ✅ | [`BUG_02_IMGDS_PICKLE_FORKSERVER.md`](BUG_02_IMGDS_PICKLE_FORKSERVER.md) |
| **`BUG-03`** | **Missing Benchmark CSV Division by Zero Guard** (`ZeroDivisionError`) | `notebooks/coursework_deepfake.ipynb` | Medium | Resolved ✅ | [`BUG_03_ZERO_DIVISION_TEST_OLD.md`](BUG_03_ZERO_DIVISION_TEST_OLD.md) |

---

## 🛠️ General Troubleshooting Reference

1. **`AttributeError: module '__main__' has no attribute 'ImgDS'` (Python 3.14+ Forkserver / Spawn):**
   - **Cause:** Python 3.14 on Linux uses `forkserver` multiprocessing by default, preventing worker processes from unpickling dynamically declared dataset classes in interactive environments.
   - **Fix:** Set `num_workers = 0` in interactive `DataLoader` definitions ([`BUG-02`](BUG_02_IMGDS_PICKLE_FORKSERVER.md)).
2. **`ZeroDivisionError: division by zero` on missing historical splits:**
   - **Cause:** Missing optional legacy benchmark files (`len(test_old) == 0`).
   - **Fix:** Use conditional checks `if len(df) > 0` before computing percentages ([`BUG-03`](BUG_03_ZERO_DIVISION_TEST_OLD.md)).
3. **Dynamic `PROJECT_ROOT` Resolution:**
   - Always resolve root path dynamically:
   ```python
   import sys, os
   from pathlib import Path
   ROOT = Path(os.getcwd()).resolve()
   if not (ROOT / "src").exists() and (ROOT.parent / "src").exists():
       ROOT = ROOT.parent
   if str(ROOT) not in sys.path:
       sys.path.insert(0, str(ROOT))
   ```
