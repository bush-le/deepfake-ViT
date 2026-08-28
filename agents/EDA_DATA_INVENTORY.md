# 📊 EDA — Comprehensive Data Inventory, 44 Deepfake Methods & Split Accounting

- **Project:** `deepfake-ViT` (Anti-Deepfake Face Detection via Meta DINOv3 ViT & ConvNeXt)
- **Scope:** Complete data inventory across **44 Fake Methods** and **7 Real Sources**.
- **Official Splits:**
  - [`data/splits/train_v5_weakfix_v3.csv`](../data/splits/train_v5_weakfix_v3.csv) (**129,884 samples**)
  - [`data/splits/val_v5_combined_universal_kaggle_boost.csv`](../data/splits/val_v5_combined_universal_kaggle_boost.csv) (**6,000 samples**)
  - [`test_coursework_44methods_balanced_zero_leakage.csv`](../data/splits/test_coursework_44methods_balanced_zero_leakage.csv) (**21,446 samples**)
  - [`test_coursework_44methods_full_zero_leakage.csv`](../data/splits/test_coursework_44methods_full_zero_leakage.csv) (**50,084 samples**)

---

## 📑 Table of Contents

1. [Unified Data Infrastructure Overview](#1-unified-data-infrastructure-overview)
2. [Distribution of 44 Synthetic & Manipulation Methods (Fake)](#2-distribution-of-44-synthetic--manipulation-methods-fake)
3. [Distribution of 7 Pristine Real Face Sources](#3-distribution-of-7-pristine-real-face-sources)
4. [Cross-Split Accounting (Train, Test Balanced, Test Full)](#4-cross-split-accounting-train-test-balanced-test-full)
5. [Zero-Leakage Auditing & Hash Deduplication](#5-zero-leakage-auditing--hash-deduplication)

---

## 1. Unified Data Infrastructure Overview

| # | Data Source / Subset | Format | Technical Description | Role in Pipeline |
|---|---|---|---|---|
| 1 | **FaceForensics++ (FF++)** | `.png` $256\times 256$ | 999 video sequences (c23 compression) | Pristine Real frames + Classic DeepFakes |
| 2 | **Celeb-DF v2** | `.mp4` / `.png` | High-res YouTube celebrity interview sequences | Pristine Real frames + High-quality Face Swaps |
| 3 | **DF40 Benchmark Suite** | `.png` $256\times 256$ | 40 modern deepfake generation algorithms | Core multi-method evaluation benchmark |
| 4 | **FFHQ / CelebA-HQ / SFHQ** | `.jpg` / `.png` | Studio portrait photography ($1024^2$) | High-resolution Real reference portraits |
| 5 | **Midjourney & Kaggle Boost** | `.jpg` / `.png` | Midjourney v5/v6, Stable Diffusion XL | Modern diffusion expansion preventing OOD blindspots |

---

## 2. Distribution of 44 Synthetic & Manipulation Methods (Fake)

| Algorithmic Family | Method Count | Representative Included Methods |
| :--- | :---: | :--- |
| **Face Swap & Blending** | 12 | `faceswap`, `simswap`, `facedancer`, `blendface`, `inswap`, `mobileswap`, `fsgan`, `uniface`, `one_shot_free`, `tpsm`, `deepfake_faceswap`, `deepfacelab` |
| **Reenactment & Talking Head** | 11 | `sadtalker`, `wav2lip`, `pirender`, `lia`, `hyperreenact`, `facevid2vid`, `MRAA`, `danet`, `fomm`, `heygen`, `heygen_new` |
| **GAN Generative Synthesis** | 8 | `StyleGAN2`, `StyleGAN3`, `StyleGANXL`, `stargan`, `starganv2`, `VQGAN`, `whichfaceisreal`, `styleclip` |
| **Diffusion & Flow Matching** | 11 | `stable_diffusion`, `sd2.1`, `MidJourney`, `pixart`, `DiT`, `SiT`, `RDDM`, `ddim`, `CollabDiff`, `e4s`, `e4e` |
| **Landmark & Feature Edit** | 2 | `mcnet`, `sfhq_studio` |
| **TOTAL FAKE METHODS** | **44** | **Exhaustive coverage of 44 contemporary deepfake methods** |

---

## 3. Distribution of 7 Pristine Real Face Sources

| Real Source | Domain Characteristics | Train Samples | Test Balanced | Test Full |
| :--- | :--- | :---: | :---: | :---: |
| **`ffhq_real`** | Flickr-Faces-HQ (High-quality studio photography) | 10,000 | 1,532 | 3,576 |
| **`celebvhq_real`** | CelebV-HQ (Celebrity YouTube videos high-res) | 8,000 | 1,532 | 3,578 |
| **`celeba_real`** | CelebFaces Attributes (In-the-wild celebrity photos) | 4,000 | 1,532 | 3,578 |
| **`faceforensics_real`**| FaceForensics++ Pristine (YouTube interview sequences) | 4,000 | 1,532 | 3,578 |
| **`celebdf_real`** | Celeb-DF v2 Real (Celebrity interviews & talk shows) | 3,006 | 1,532 | 3,578 |
| **`vggface2_real`** | VGGFace2 Cleaned (Diverse poses, lighting, ages) | 1,000 | 1,532 | 3,578 |
| **`sfhq_real`** | Synthetic Face HQ Pristine (Neutral studio lighting) | 1,000 | 1,531 | 3,576 |
| **TOTAL REAL** | **7 Standardized Real Domains** | **31,006** | **10,723** | **25,042** |

---

## 4. Cross-Split Accounting (Train, Test Balanced, Test Full)

| # | Method Name | Label | Train (129.8k) | Test Balanced (21.4k) | Test Full (50.0k) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| 1 | `faceswap` | FAKE (1) | 3,000 | 300 | 658 |
| 2 | `sadtalker` | FAKE (1) | 3,000 | 300 | 632 |
| 3 | `facedancer` | FAKE (1) | 3,000 | 300 | 667 |
| 4 | `blendface` | FAKE (1) | 3,000 | 300 | 664 |
| 5 | `simswap` | FAKE (1) | 3,000 | 300 | 647 |
| 6 | `fsgan` | FAKE (1) | 3,000 | 300 | 614 |
| 7 | `e4s` | FAKE (1) | 3,000 | 300 | 356 |
| 8 | `wav2lip` | FAKE (1) | 3,000 | 300 | 528 |
| 9 | `inswap` | FAKE (1) | 3,000 | 300 | 447 |
| 10 | `lia` | FAKE (1) | 3,000 | 300 | 675 |
| 11 | `mobileswap` | FAKE (1) | 3,000 | 300 | 1,299 |
| 12 | `starganv2` | FAKE (1) | 2,500 | 300 | 673 |
| 13 | `pirender` | FAKE (1) | 3,000 | 300 | 641 |
| 14 | `uniface` | FAKE (1) | 3,000 | 300 | 669 |
| 15 | `one_shot_free` | FAKE (1) | 3,000 | 300 | 670 |
| 16 | `tpsm` | FAKE (1) | 3,000 | 300 | 592 |
| 17 | `VQGAN` | FAKE (1) | 2,500 | 300 | 903 |
| 18 | `pixart` | FAKE (1) | 2,500 | 300 | 602 |
| 19 | `StyleGAN2` | FAKE (1) | 2,500 | 300 | 674 |
| 20 | `mcnet` | FAKE (1) | 3,000 | 300 | 968 |
| 21 | `StyleGANXL` | FAKE (1) | 2,500 | 300 | 973 |
| 22 | `SiT` | FAKE (1) | 2,500 | 300 | 665 |
| 23 | `hyperreenact` | FAKE (1) | 3,000 | 300 | 662 |
| 24 | `facevid2vid` | FAKE (1) | 3,000 | 300 | 674 |
| 25 | `MRAA` | FAKE (1) | 3,000 | 300 | 543 |
| 26 | `RDDM` | FAKE (1) | 2,500 | 300 | 1,460 |
| 27 | `sd2.1` | FAKE (1) | 2,500 | 300 | 683 |
| 28 | `danet` | FAKE (1) | 3,000 | 300 | 973 |
| 29 | `StyleGAN3` | FAKE (1) | 2,500 | 300 | 593 |
| 30 | `ddim` | FAKE (1) | 2,500 | 300 | 669 |
| 31 | `fomm` | FAKE (1) | 3,000 | 300 | 982 |
| 32 | `CollabDiff` | FAKE (1) | 2,500 | 300 | 447 |
| 33 | `whichfaceisreal` | FAKE (1) | 2,500 | 300 | 451 |
| 34 | `DiT` | FAKE (1) | 2,500 | 300 | 455 |
| 35 | `stargan` | FAKE (1) | 2,500 | 300 | 187 |
| 36 | `styleclip` | FAKE (1) | 2,500 | 300 | 25 |
| 37 | `e4e` | FAKE (1) | 2,500 | 300 | 11 |
| 38 | `MidJourney` | FAKE (1) | 800 | 187 | 187 |
| 39 | `deepfacelab` | FAKE (1) | 25 | 25 | 25 |
| 40 | `heygen` | FAKE (1) | 11 | 11 | 11 |

---

## 5. Zero-Leakage Auditing & Hash Deduplication

- **Filepath Intersection:** Verified **0 overlapping image file paths** ($Train \cap Test_{Bal} = \emptyset$, $Train \cap Test_{Full} = \emptyset$).
- **MD5 Hash Deduplication:** Cross-audited **127,185 MD5 byte-level hashes**, eliminating historical collisions.
- **Master EDA Notebook:** [`notebooks/coursework_eda.ipynb`](../notebooks/coursework_eda.ipynb) contains complete reproducible audit code.
- **Master Benchmark Evaluation Notebook:** [`notebooks/coursework_deepfake.ipynb`](../notebooks/coursework_deepfake.ipynb) contains complete live benchmark evaluation code.
