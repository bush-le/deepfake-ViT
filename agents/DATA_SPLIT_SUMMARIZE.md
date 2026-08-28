# DATA_SPLIT_SUMMARIZE.md — Unified Data Architecture & Zero-Leakage Split Summary

- **Motivation/Background**: Developing high-accuracy, highly generalizable facial deepfake detection models (**deepfake-ViT**) requires integrating diverse benchmark domains (FaceForensics++, Celeb-DF v1 & v2, DF40, SFHQ, FFHQ, Kaggle Boost, Midjourney Boost) under strict class balance and **0% identity/video/byte data leakage**.
- **Purpose**: Document the unified data partitioning architecture, sample census across Train/Val/Test splits, formal mathematical proof of the 3-Tier Zero-Leakage protocol, and the comprehensive catalog of 44 evaluated deepfake methods.
- **Key References**:
  - `data/splits/train_v5_weakfix_v3.csv` (**129,884 samples**)
  - `data/splits/val_v5_combined_universal_kaggle_boost.csv` (**6,000 samples**)
  - `test_coursework_44methods_balanced_zero_leakage.csv` (**21,446 samples**)
  - `test_coursework_44methods_full_zero_leakage.csv` (**50,084 samples**)

---

## 📑 Table of Contents

1. [Unified Data Infrastructure Overview](#1-unified-data-infrastructure-overview)
2. [Master Dataset Split Census](#2-master-dataset-split-census)
3. [Mathematical Proof of 3-Tier Zero-Leakage Protocol](#3-mathematical-proof-of-3-tier-zero-leakage-protocol)
4. [Catalog of 44 Deepfake Methods + 7 Real Sources](#4-catalog-of-44-deepfake-methods--7-real-sources)
5. [Integration in Notebooks & Evaluation Tools](#5-integration-in-notebooks--evaluation-tools)

---

## 1. Unified Data Infrastructure Overview

The `deepfake-ViT` repository unifies raw assets from 7 real domains and 44 deepfake generation algorithms into an isolated, multi-split evaluation pipeline:

```
                                          ┌─────────────────────────────────────────────────────────────┐
                                          │             MULTI-DATASET SOURCE INVENTORY                  │
                                          ├─────────────────────────────────────────────────────────────┤
                                          │ • FaceForensics++ (31,949 frames PNG Real)                  │
                                          │ • Celeb-DF-v2 (890 videos Real + 5,639 videos Fake)         │
                                          │ • Celeb-DF-v1 (408 videos Real + 795 videos Fake)           │
                                          │ • FFHQ & CelebA-HQ & SFHQ (High-Res Studio Reals)           │
                                          │ • DF40 Benchmark Suite (40 generative & manipulation fakes) │
                                          │ • Kaggle & Midjourney Enhancement Boost (v5 Trio)           │
                                          └──────────────────────────────┬──────────────────────────────┘
                                                                         │
                                 ┌───────────────────────────────────────┴───────────────────────────────────────┐
                                 │                                                                               │
                                 ▼                                                                               ▼
              ┌────────────────────────────────────────┐                      ┌──────────────────────────────────────────────┐
              │     OFFICIAL TRAINING SET (TRAIN)      │                      │     COURSEWORK BENCHMARK SUITE (0% LEAK)     │
              ├────────────────────────────────────────┤                      ├──────────────────────────────────────────────┤
              │ • train_v5_weakfix_v3.csv              │                      │ • test_coursework_balanced (21,446 images)   │
              │   - Total: 129,884 samples             │                      │   - 10,723 Real : 10,723 Fake (Exact 1:1)    │
              │   - Real: 31,006 (7 real domains)      │                      │ • test_coursework_full (50,084 images)       │
              │   - Fake: 98,878 (44 fake methods)     │                      │   - 25,042 Real : 25,042 Fake (Full Suite)   │
              │ • val_v5_combined_universal (6,000 imgs│                      │ • 0% Identity, Video & MD5 Byte Collision    │
              └────────────────────────────────────────┘                      └──────────────────────────────────────────────┘
```

---

## 2. Master Dataset Split Census

| Split Name | CSV File Path | Total Images | Real (0) | Fake (1) | Balance Ratio | Underlying Composition |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **Train v3 Clean** | `train_v5_weakfix_v3.csv` | **129,884** | 31,006 | 98,878 | Multi-subset Balanced Sampler | 44 Fake Methods + 7 Real Domains |
| **Validation v5** | `val_v5_combined_universal_kaggle_boost.csv` | **6,000** | 3,000 | 3,000 | **1.00 : 1** | Multi-domain balanced (Validation & Tuning) |
| **Test CourseWork Balanced** | `test_coursework_44methods_balanced_zero_leakage.csv` | **21,446** | **10,723** | **10,723** | **1.00 : 1** | ~300 images / method across all 44 methods |
| **Test CourseWork Full Suite** | `test_coursework_44methods_full_zero_leakage.csv` | **50,084** | **25,042** | **25,042** | **1.00 : 1** | ~600–1,500 images / method across all 44 methods |

---

## 3. Mathematical Proof of 3-Tier Zero-Leakage Protocol

### 3.1 Tier 1: Absolute Filepath Disjointness
Zero image file paths overlap between Train and Test splits:
$$\mathcal{S}_{train} \cap \mathcal{S}_{test\_bal} = \emptyset, \quad \mathcal{S}_{train} \cap \mathcal{S}_{test\_full} = \emptyset \quad (|Train \cap Test| = 0)$$

### 3.2 Tier 2: Identity & Video Sequence Isolation
All person identities and video sequence IDs present in the training set are strictly segregated from evaluation splits. Held-out test frames originate exclusively from unseen subjects and independent video sequences.

### 3.3 Tier 3: Byte-Level MD5 Hash Collision Deduplication
Cross-referenced **127,185 MD5 byte hashes** from the complete training dataset:
- Identified and eliminated **5,680 path overlaps**.
- Identified and eliminated **2,261 exact MD5 byte collisions**.
- Final audit result: **✅ 0.0000% LEAK (Absolute Zero-Leakage Standard)**.

---

## 4. Catalog of 44 Deepfake Methods + 7 Real Sources

### 4.1 44 Evaluated Deepfake Generation Methods:
1. **Face Swap & Blending (12 methods):** `faceswap`, `simswap`, `facedancer`, `blendface`, `inswap`, `mobileswap`, `fsgan`, `uniface`, `one_shot_free`, `tpsm`, `deepfake_faceswap`, `deepfacelab`.
2. **Facial Reenactment & Animation (11 methods):** `sadtalker`, `wav2lip`, `pirender`, `lia`, `hyperreenact`, `facevid2vid`, `MRAA`, `danet`, `fomm`, `heygen`, `heygen_new`.
3. **Generative Adversarial Networks (8 methods):** `StyleGAN2`, `StyleGAN3`, `StyleGANXL`, `stargan`, `starganv2`, `VQGAN`, `whichfaceisreal`, `styleclip`.
4. **Diffusion Models & Flow Matching (11 methods):** `stable_diffusion`, `sd2.1`, `MidJourney`, `pixart`, `DiT`, `SiT`, `RDDM`, `ddim`, `CollabDiff`, `e4s`, `e4e`.
5. **Expression & Landmark Manipulation (2 methods):** `mcnet`, `sfhq_studio`.

### 4.2 7 Pristine Real Face Domains:
1. `ffhq_real` (Flickr-Faces-HQ Studio)
2. `celebvhq_real` (CelebV-HQ Video Frames)
3. `celeba_real` (CelebFaces Attributes)
4. `faceforensics_real` (FaceForensics++ Pristine Sequences)
5. `celebdf_real` (Celeb-DF v2 Real YouTube Interviews)
6. `vggface2_real` (VGGFace2 Cleaned)
7. `sfhq_real` (Synthetic Face HQ Pristine Studio)

---

## 5. Integration in Notebooks & Evaluation Tools

- **Master EDA Notebook:** [`notebooks/coursework_eda.ipynb`](../notebooks/coursework_eda.ipynb) conducts comprehensive multi-split exploratory data analysis and physical signal forensics.
- **Master Benchmark Notebook:** [`notebooks/coursework_deepfake.ipynb`](../notebooks/coursework_deepfake.ipynb) executes live GPU comparative evaluation between DINOv3 ViT-S/16 and ConvNeXt-Tiny.
- **Interactive Single-Image Predictor:** [`notebooks/predict_image.ipynb`](../notebooks/predict_image.ipynb) provides an interactive testing harness for custom face images.
