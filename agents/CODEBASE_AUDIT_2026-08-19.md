# Security & Code Quality Audit — Delta since `7c6f75d`

- **Motivation/Background**: After the audit report and remediation work, this
  reviews every commit since baseline `7c6f75d` (the codebase audit report) to
  catch issues introduced or left open by the remediation.
- **Purpose**: Assess vulnerabilities, bugs, performance, best-practice
  compliance, and regressions in the `7c6f75d..HEAD` delta.
- **Overview Pipeline**: Commit/diff enumeration → full code-diff review →
  targeted checks (download paths, README setup, unused imports, model
  signatures) → report.
- **Detailed Plan**: §1 summary; §2 scope; §3 security; §4 bugs & regressions;
  §5 performance; §6 compliance; §7 findings table; §8 action plan.
- **References**: `git`, repo rulebase (`agents/rules/*`), prior
  [CODEBASE_AUDIT.md](CODEBASE_AUDIT.md).

---

## Table of Contents

- [1. Executive Summary](#1-executive-summary)
- [2. Scope & Method](#2-scope--method)
- [3. Security](#3-security)
- [4. Bugs & Regressions](#4-bugs--regressions)
- [5. Performance](#5-performance)
- [6. Compliance & Best Practices](#6-compliance--best-practices)
- [7. Findings Summary](#7-findings-summary)
- [8. Action Plan](#8-action-plan)

---

## 1. Executive Summary

**Verdict: the delta is a net improvement with no critical/high findings.**
The remediation hardened deserialization (`weights_only=True`, no
`allow_pickle`), fixed a real `--device` bug in two training scripts, added a
CUDA-capable `--device` to `analyze_threshold.py`, aligned eval checkpoint
defaults to where training saves, added smoke tests, and improved training
performance. Two minor bugs remain (a README venv-setup inconsistency and a
`data/DF40` vs `data/raw/DF40` path mismatch), plus the known open ARCH-1
checkpointing gap. No functional regression in training/eval logic was
introduced.

## 2. Scope & Method

- **Baseline:** `7c6f75d` · **Head:** `0c17245` · **31 commits** since baseline.
- **Changed code files:** `src/training/{train,finetune_lora,finetune_compare}.py`,
  `src/utils/seeding.py`, `src/eval/{analyze_threshold,eval_df40_fake,
  eval_df40_vit_cnn,eval_finetuned,eval_finetuned_identity_disjoint,evaluate,
  predict}.py`, `src/data/{build_*,download_df40}.py`, `tests/`, `pytest.ini`,
  `requirements.lock.txt`, `.gitignore`.
- Method: full `git diff 7c6f75d..HEAD` review + targeted runtime checks
  (paths, imports, model signatures).

## 3. Security

**No new vulnerabilities introduced.** Positives:

- **`weights_only=True` added to every `torch.load`** in changed files
  (`train.py`, `finetune_compare.py`, `analyze_threshold.py`, `eval_df40_fake.py`,
  `eval_finetuned.py`, `eval_finetuned_identity_disjoint.py`) — closes the
  pickle-RCE surface (SEC-1).
- **`allow_pickle` removed** from `evaluate.py`/`predict.py` (SEC-2).
- No secrets/keys/tokens introduced; `.gitignore` correctly added
  `data/raw/DF40/`, `experiments/results/features/`, `.feynman/` to keep large
  generated data out of git.

Remaining (Low):

- **SEC-LOW-1 — retained Google Drive downloader is path-inconsistent and
  unignored.** `src/data/download_df40.py` still writes to `BASE=data/DF40`
  (its docstring now says `data/DF40`), while every build script and the HF
  recommendation use `DF40_ROOT` default `data/raw/DF40`. Running it then
  building without `DF40_ROOT` set fails to find data; and `data/DF40` is not
  gitignored (only `data/raw/DF40/` is). Deprecated but present.

## 4. Bugs & Regressions

- **BUG-MED-1 — README setup instructions are broken (real setup bug).**
  [README.md](README.md) L35 runs `python -m venv venv` (creates `venv/`), but
  L38 activates `source .venv/bin/activate` (uses `.venv/`) and the tree comment
  says `.venv/`. A user following the docs hits "No such file or directory:
  .venv/bin/activate". Introduced by commit `13308bf`. **Fix:** make the create
  and activate steps use the same name (`.venv`).
- **BUG-LOW-2 — `data/DF40` vs `data/raw/DF40` path mismatch** (see SEC-LOW-1).
- **Fixed in this delta (no longer bugs):** the missing `else` for `--device`
  in `finetune_lora.py`/`finetune_compare.py` (previously crashed with
  `UnboundLocalError` on explicit `--device`); `analyze_threshold.py` hardcoded
  to mps/cpu and rejecting `--device`; eval checkpoint defaults pointing at
  `experiments/checkpoints/...` that training no longer writes to.
- **No functional regression** in training/eval logic. `DinoViT(img_size=256,
  patch_size=16)` (used by the smoke test) is a valid constructor call —
  verified.

## 5. Performance

- **Improvement (positive):** DataLoaders now use `num_workers=2` +
  `pin_memory=True` (was 0), and an opt-in `--amp` bfloat16 autocast wraps the
  forward+loss. On the 8 GB card this helps throughput/memory.
- **PERF-LOW-1 — `num_workers=2` default with spawn on Windows/macOS.** The
  scripts have the `if __name__ == "__main__"` guard, so multi-worker loads
  work, but each worker re-imports `src.models` (heavy torch imports) → slower
  process startup and higher memory. Not a bug; consider `--num-workers 0` in
  docs for constrained boxes, or a lower default.

## 6. Compliance & Best Practices

- **COMP-MED-1 — ARCH-1 remains open.** Training still saves best-state-only
  dicts (no optimizer/scheduler/RNG state, no `_last.pt`, no resume, no JSONL
  history, no `RunLogger`), and `agents/rules/LOGGING_CHECKPOINT_RULES.md` still
  references nonexistent modules (`run_logger.py`, `checkpoint_utils.py`,
  `train_model.py`). This delta aligned checkpoint *paths* but not the *format*.
  Documented in [CODEBASE_AUDIT_STATUS.md](progress/CODEBASE_AUDIT_STATUS.md) as
  the single open P0.
- **COMP-LOW-1 — unused `import os`** in `tests/test_smoke.py` (lint-only; no
  behavior impact).
- **Positive:** smoke tests + `conftest.py` added; `requirements.lock.txt`
  added with the corrected lockfile-regeneration instruction (excluding
  torch/torchvision); `.gitignore` covers new generated dirs.

## 7. Findings Summary

| ID | Sev | Area | Finding |
|---|---|---|---|
| BUG-MED-1 | **Medium** | Bugs | README venv setup: `python -m venv venv` ≠ `source .venv/bin/activate` |
| COMP-MED-1 | **Medium** | Compliance | ARCH-1 open: checkpoints best-state-only; rules reference nonexistent modules |
| SEC-LOW-1 | Low | Security/Data | `download_df40.py` saves to `data/DF40`; inconsistent + unignored |
| BUG-LOW-2 | Low | Bugs | `data/DF40` vs `data/raw/DF40` path mismatch |
| PERF-LOW-1 | Low | Performance | `num_workers=2` spawn startup cost on Windows/macOS |
| COMP-LOW-1 | Low | Compliance | Unused `import os` in `tests/test_smoke.py` |

Positive: deserialization hardening complete; `--device` bug fixed; eval ckpt
defaults aligned; `analyze_threshold` now CUDA-capable; smoke tests added;
training perf improved. No critical/high findings; no functional regression.

## 8. Action Plan

1. **P1 — fix the README venv bug** (`python -m venv .venv` so create/activate
   match). Small, user-facing.
2. **P1 — decide ARCH-1** (implement full-state checkpointing in `train.py` or
   rescope `LOGGING_CHECKPOINT_RULES`) before the deadline.
3. **P2 — reconcile `download_df40.py` path** with `DF40_ROOT`
   (`data/raw/DF40`) or deprecate it, and gitignore `data/DF40`.
4. **P2 — optional** `--num-workers` guidance and remove unused `import os` in
   the smoke test.

---

## References

- [CODEBASE_AUDIT.md](CODEBASE_AUDIT.md) — prior full audit
- [CODEBASE_AUDIT_STATUS.md](progress/CODEBASE_AUDIT_STATUS.md) — remediation tracker
- `agents/rules/*` — project rulebase
- `git log 7c6f75d..HEAD` — 31 commits reviewed
