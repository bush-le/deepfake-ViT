# Cross-Domain Balanced Dataset Protocol

- **Motivation/Background**: Establish technical specification, requirements, and deliverables for pipeline phase CROSS_DOMAIN_BALANCED_PROTOCOL.
- **Purpose**: Guide the execution and quality gates of phase CROSS_DOMAIN_BALANCED_PROTOCOL.
- **Overview Pipeline**: Phase scope definition -> implementation guidelines -> verification gates -> status sign-off.
- **Detailed Plan**: §1 Phase Overview & Scope; §2 Technical Specification; §3 Deliverables & Artifacts; §4 Verification Protocol.
- **References**: `docs/OVERVIEW.md`, `docs/PURPOSE.md`, `agents/rules/`.
- **Created**: 2026-08-22T21:44:36+07:00
- **Last Updated**: 2026-09-08T10:15:00+07:00

---

This document defines the standard protocol for cross-domain balanced dataset construction and evaluation in `deepfake-ViT`.

---

## 1. Objectives & Guarantees
1. **Equal Class Representation**: Maintain a strict 1:1 binary class ratio ($N_{\text{Real}} = N_{\text{Fake}}$) across validation and benchmark test sets.
2. **Multi-Domain Diversity**: Balance sampling across 5 Generative Paradigms (*Diffusion, GANs, Face Swapping, Reenactment, Editing*) and 7 Real Domains.
3. **Zero Identity & Video Leakage**: Strictly isolate person identities, YouTube video sequences, and image MD5 byte hashes between training and testing partitions.

---

## 2. Standard CSV Split Schema
All splits must adhere to the standardized CSV columns:
- `path`: Relative or absolute file path to the face image ($256 \times 256$).
- `label`: Integer binary label (`0` = Real, `1` = Fake).
- `method`: Specific generator name or real domain identifier (e.g., `midjourney_v5`, `faceswap`, `ffhq_real`).
- `domain`: High-level domain code (`efs`, `ffc`, `cdc`, `oth`, `fe`, `real`).
- `split`: Dataset partition identifier (`train`, `val`, `test`).

---

## 3. Official Master Splits
- **Train Split (129,884 samples)**: [`data/splits/train_v5_weakfix_v3.csv`](../../data/splits/train_v5_weakfix_v3.csv)
- **Validation Split (6,000 samples)**: [`data/splits/val_v5_combined_universal_kaggle_boost.csv`](../../data/splits/val_v5_combined_universal_kaggle_boost.csv)
- **Test Balanced Suite (21,446 samples)**: [`data/splits/test_coursework_44methods_balanced_zero_leakage.csv`](../../data/splits/test_coursework_44methods_balanced_zero_leakage.csv)
- **Test Full Suite (50,084 samples)**: [`data/splits/test_coursework_44methods_full_zero_leakage.csv`](../../data/splits/test_coursework_44methods_full_zero_leakage.csv)

---

## 4. Sampling & Dataloader Constraints
- During training, dynamic **Balanced Batch Sampling** or **Class Loss Weighting** ($W_{\text{Real}}=1.0, W_{\text{Fake}}=0.35$) must be employed to mitigate inherent training imbalance.
- Interactive notebooks must set `num_workers = 0` to preserve compatibility with Python 3.14+ `forkserver` environments.
