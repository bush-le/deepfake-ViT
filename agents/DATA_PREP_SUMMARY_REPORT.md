# DATA_PREP_SUMMARY_REPORT.md — Summary Report: Data Preparation & Partitioning Pipeline

- **Motivation/Background**: Training and evaluating facial deepfake detection models (**deepfake-ViT**) requires rigorous data curation: multi-domain integration (FaceForensics++, Celeb-DF v1 & v2, DF40, SFHQ, FFHQ, Kaggle Boost, Midjourney), class balance preservation, and zero data leakage (Zero Identity/Video/Byte Leakage).
- **Purpose**: Comprehensive technical specification of data partitioning structures, extraction mechanisms, MD5 deduplication audits, and the authoritative split manifests used for model training and evaluation.
- **Key Pipeline Artifacts**:
  - `train_v5_weakfix_v3.csv` (**129,884 samples**)
  - `val_v5_combined_universal_kaggle_boost.csv` (**6,000 samples**)
  - `test_coursework_44methods_balanced_zero_leakage.csv` (**21,446 samples**)
  - `test_coursework_44methods_full_zero_leakage.csv` (**50,084 samples**)
  - `train_v5_weakfix_v3_hashes.json` (**127,185 MD5 hashes**)

---

## 📑 Table of Contents

1. [Full Pipeline Execution Summary](#1-full-pipeline-execution-summary)
2. [Multi-Source Data Curation & Extraction](#2-multi-source-data-curation--extraction)
3. [44-Methods Zero-Leakage Test Suite Construction](#3-44-methods-zero-leakage-test-suite-construction)
4. [Three-Tier Zero-Leakage Audit (Path / Identity / MD5 Collision)](#4-three-tier-zero-leakage-audit-path--identity--md5-collision)
5. [Authoritative Training Split (Train v3 - 129,884 Images)](#5-authoritative-training-split-train-v3---129884-images)
6. [Integration into Evaluation Systems & Notebooks](#6-integration-into-evaluation-systems--notebooks)

---

## 1. Full Pipeline Execution Summary

The `deepfake-ViT` dataset curation pipeline operates across 5 core stages:

```
[Raw Sources: FF++, Celeb-DF, DF40, FFHQ, SFHQ, Midjourney]
                         ↓ (Extraction & Face Alignment 256x256)
             [Processed Clean Image Pool]
                         ↓ (Partitioning & Identity Isolation)
    ┌─────────────────────────────────────────────────────────────┐
    │                                                             │
    ▼                                                             ▼
[Train Split v3: 129,884 imgs]                [Candidate Test Suite: 58,025 imgs]
    │                                                             │
    │ (Extract 127,185 MD5 Hashes)                                │ (3-Tier Leakage Filter)
    │                                                             ↓
    │                                              - Remove 5,680 Path Overlaps
    │                                              - Remove 2,261 MD5 Collisions
    │                                                             ↓
    │                                             [Audited Zero-Leakage Test Suites]
    │                                             • Test Balanced: 21,446 imgs (1:1)
    │                                             • Test Full:     50,084 imgs (1:1)
    └─────────────────────────────────────────────────────────────┘
```

---

## 2. Multi-Source Data Curation & Extraction

| Data Source | Volume / Scale | Structure / Properties | Role in Pipeline |
| :--- | :--- | :--- | :--- |
| **FaceForensics++ (FF++)** | 999 videos, 31,949 frames | PNG $256 \times 256$ cropped face frames from YouTube videos. | Baseline clean Real frames and classic DeepFake manipulations. |
| **Celeb-DF-v2** | 890 Real videos, 5,639 Fake videos | High-resolution celebrity YouTube interview videos. | Benchmark Real frames and Celeb-DF face swaps. |
| **FFHQ & CelebA-HQ** | 70,000 + 30,000 studio images | High-quality $1024 \times 1024$ studio portraits. | Unconditional Real reference faces for GAN/Diffusion synthesis. |
| **DF40 Benchmark** | 40 synthetic methods | Collection of 40 Deepfake algorithms (GAN, Diffusion, Reenactment, FaceSwap). | Multi-method cross-generator evaluation benchmark. |
| **Kaggle & Midjourney Boost** | 12,000+ supplementary images | Midjourney v5/v6, Stable Diffusion XL, HeyGen, DeepFaceLab. | State-of-the-art diffusion coverage to prevent out-of-domain blindspots. |

---

## 3. 44-Methods Zero-Leakage Test Suite Construction

To provide fair and unbiased evaluation, two standardized test suites were engineered:

1. **Test CourseWork Balanced (1:1 Ratio — 21,446 images):**
   - **Real:** 10,723 images from 7 independent real domains.
   - **Fake:** 10,723 images uniformly sampled across all 44 methods (~300 images/method).
   - Guarantees unbiased evaluation without method volume skew.
2. **Test CourseWork Full Suite (50,084 images):**
   - **Real:** 25,042 images.
   - **Fake:** 25,042 images (~600–1,500 images/method).
   - Large-scale benchmark evaluating overall statistical robustness and throughput.

---

## 4. Three-Tier Zero-Leakage Audit (Path / Identity / MD5 Collision)

| Audit Tier | Methodology | Verified Result | Certification Status |
| :--- | :--- | :--- | :--- |
| **Tier 1: Path Disjointness** | Set intersection $Train \cap Test$ | $0$ overlapping image filepaths | ✅ **PASSED (0.00% Leak)** |
| **Tier 2: Identity Disjointness** | Strict partitioning by celebrity & video sequence ID | 0 overlapping persons/videos | ✅ **PASSED (0.00% Leak)** |
| **Tier 3: MD5 Collision Deduplication** | Match test byte hashes against 127k Train MD5 hashes | Filtered 5,680 path and 2,261 byte collisions | ✅ **PASSED (0.0000% Leak)** |

---

## 5. Authoritative Training Split (Train v3 - 129,884 Images)

The official training split `train_v5_weakfix_v3.csv` comprises **129,884 samples** structured across 51 balanced subsets:
- **31,006 Real images** from 7 independent domains.
- **98,878 Fake images** spanning 44 deepfake generation algorithms.
- Trained using a **Balanced Batch Sampler** (Domain-balanced / FaceSwap-boosted) ensuring each batch has equal class representation.

---

## 6. Integration into Evaluation Systems & Notebooks

- **Master EDA Notebook:** [`notebooks/coursework_eda.ipynb`](../notebooks/coursework_eda.ipynb) conducts multi-split physical signal and statistical forensic analysis.
- **Master Evaluation Notebook:** [`notebooks/coursework_deepfake.ipynb`](../notebooks/coursework_deepfake.ipynb) loads frozen checkpoints and verifies Zero-Leakage integrity before live inference.
- **Interactive Single-Image Predictor:** [`notebooks/predict_image.ipynb`](../notebooks/predict_image.ipynb) allows interactive testing on any image.
