# BUG-03 — Missing Historical Benchmark CSV Division-by-Zero Guard (`ZeroDivisionError`)

- **Motivation/Background**: Document bug diagnosis, root cause analysis, and regression prevention for BUG_03_ZERO_DIVISION_TEST_OLD.
- **Purpose**: Track lifecycle and remediation evidence for issue BUG_03_ZERO_DIVISION_TEST_OLD.
- **Overview Pipeline**: Bug discovery -> root cause analysis -> patch verification -> regression testing.
- **Detailed Plan**: §1 Bug Description & Symptoms; §2 Root Cause Diagnosis; §3 Remediation & Code Changes; §4 Verification & Prevention.
- **References**: `src/`, `tests/`, `agents/rules/CODEBASE_AUDIT.md`.
- **Created**: 2026-08-28T11:00:31+07:00
- **Last Updated**: 2026-09-08T10:15:00+07:00

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
