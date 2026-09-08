# DATA_PREP_STATUS.md — Data Preparation Status Tracker

- **Phase:** Data Preparation & Multi-Domain Partitioning
- **Current Status:** **COMPLETED & ZERO-LEAKAGE CERTIFIED (0.0000%)**
- **Last Verification:** 2026-08-28

---

## 1. Split Inventory & Deliverables

| Artifact | Samples | Balance | Status | File Path |
| :--- | :---: | :---: | :---: | :--- |
| **Train Split (v3 Clean)** | 129,884 | Multi-Subset Balanced | ✅ Frozen | [`data/splits/train_v5_weakfix_v3.csv`](../../data/splits/train_v5_weakfix_v3.csv) |
| **Validation Split (v5 Boost)** | 6,000 | 1:1 (3k Real : 3k Fake) | ✅ Frozen | [`data/splits/val_v5_combined_universal_kaggle_boost.csv`](../../data/splits/val_v5_combined_universal_kaggle_boost.csv) |
| **Test CourseWork Balanced** | 21,446 | 1:1 (10.7k Real : 10.7k Fake) | ✅ Frozen | [`test_coursework_44methods_balanced_zero_leakage.csv`](../../data/splits/test_coursework_44methods_balanced_zero_leakage.csv) |
| **Test CourseWork Full Suite** | 50,084 | 1:1 (25.0k Real : 25.0k Fake) | ✅ Frozen | [`test_coursework_44methods_full_zero_leakage.csv`](../../data/splits/test_coursework_44methods_full_zero_leakage.csv) |
| **Hash Manifest (MD5)** | 127,185 | Deduplicated Hashes | ✅ Verified | [`train_v5_weakfix_v3_hashes.json`](../../data/splits/train_v5_weakfix_v3_hashes.json) |

---

## 2. Completed Milestones

- [x] Extracted high-resolution face crops ($256\times 256$) across FaceForensics++, Celeb-DF v2, DF40, FFHQ, SFHQ, and Kaggle Midjourney Boost.
- [x] Standardized 44 Deepfake methods and 7 Real face sources into unified schemas.
- [x] Eliminated 5,680 path overlaps and 2,261 exact byte collision duplicates.
- [x] Verified 0 path overlap ($Train \cap Test_{Bal} = \emptyset, Train \cap Test_{Full} = \emptyset$).
- [x] Integrated into Master EDA Notebook: [`notebooks/coursework_eda.ipynb`](../../notebooks/coursework_eda.ipynb).
- [x] Integrated into Master Benchmark Evaluation Notebook: [`notebooks/coursework_deepfake.ipynb`](../../notebooks/coursework_deepfake.ipynb).
