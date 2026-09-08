# BUG-03 — Missing Historical Benchmark CSV Division-by-Zero Guard (`ZeroDivisionError`)

- **Bug ID:** `BUG-03`
- **Title:** `ZeroDivisionError: division by zero` when computing relative metrics against optional legacy benchmark datasets
- **Component:** `notebooks/coursework_deepfake.ipynb` / Interactive Dataset Audit & Comparison
- **Date Reported:** 2026-08-28
- **Date Resolved:** 2026-08-28
- **Severity:** Medium
- **Status:** **Resolved & Verified ✅**

---

## 1. Symptoms & Error Traceback

During the initial execution of Section 1 in the interactive evaluation notebook, when calculating data split ratios against legacy test sets (`test_old`), the notebook threw a runtime exception:

```text
ZeroDivisionError: division by zero
  ratio = len(test_bal) / len(test_old)
```

This occurred in clean environments where historical interim test sets were omitted or cleaned up, leaving `len(test_old) == 0`.

---

## 2. Root Cause Analysis

1. **Assumed Presence of Legacy Artifacts:**
   The notebook contained comparison cells contrasting the newly created 44-Methods Zero-Leakage test suite against legacy interim test sets (`test_balanced_fixed_zero_leakage.csv` or `test_v5_combined_universal.csv`).
2. **Unguarded Float Division:**
   When optional historical CSV files were absent, `df_old` evaluated to an empty DataFrame, causing division by zero when calculating percentage changes ($\Delta\%$).

---

## 3. Resolution & Defensive Implementation

1. **Added Defensive Length Guards & Fallbacks:**
   All summary calculations and audit tables now implement strict conditional length checking:
   ```python
   def safe_ratio(num, den, default="N/A"):
       if den is not None and len(den) > 0:
           return f"{(len(num) / len(den) * 100):.1f}%"
       return default
   ```
2. **Self-Contained 3-Split Structure:**
   The notebook's census and audit now focus primarily on the 3 active, authoritative splits:
   - `train_v5_weakfix_v3.csv` (129,884 samples)
   - `test_coursework_44methods_balanced_zero_leakage.csv` (21,446 samples)
   - `test_coursework_44methods_full_zero_leakage.csv` (50,084 samples)

---

## 4. Verification & Testing

- Verified Section 1 executes completely without errors in both fresh and existing directory layouts.
- Output tables display clean counts and percentages regardless of legacy split availability.
