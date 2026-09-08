# DATA_PREP.md — Data Preparation Specification

- **Motivation/Background**: Establish technical specification, requirements, and deliverables for pipeline phase DATA_PREP.
- **Purpose**: Guide the execution and quality gates of phase DATA_PREP.
- **Overview Pipeline**: Phase scope definition -> implementation guidelines -> verification gates -> status sign-off.
- **Detailed Plan**: §1 Phase Overview & Scope; §2 Technical Specification; §3 Deliverables & Artifacts; §4 Verification Protocol.
- **References**: `docs/OVERVIEW.md`, `docs/PURPOSE.md`, `agents/rules/`.
- **Created**: 2026-08-18T11:19:39+07:00
- **Last Updated**: 2026-09-08T10:15:00+07:00

---

## 1. Background & Scope

Evaluating Deepfake detection requires diverse, cross-generator representation while strictly preventing train-test contamination. This phase aggregates real data (FF++, Celeb-DF v2, FFHQ, CelebA-HQ, SFHQ, CelebV-HQ, VGGFace2) and fake data (44 methods including GANs, Diffusion, FaceSwap, and Reenactment) into deterministic, zero-leakage splits.

---

## 2. Dataset Architecture & Deliverables

- **Input Data Sources:**
  - `FaceForensics++`: 999 pristine video sequences.
  - `Celeb-DF-v2`: 890 Real and 5,639 Fake high-resolution celebrity interview videos.
  - `DF40 Suite`: 40 generative and manipulation algorithms.
  - `High-Res Real Sources`: FFHQ, CelebA-HQ, SFHQ, CelebV-HQ, VGGFace2.
  - `Modern Generative Augmentations`: Midjourney v5/v6, Stable Diffusion XL, HeyGen, DeepFaceLab.
- **Output Splits (Frozen):**
  - **Fixed Train Split:** [`data/splits/train_v5_weakfix_v3.csv`](../../data/splits/train_v5_weakfix_v3.csv) (**129,884 images** across 51 subsets).
  - **Validation Split:** [`data/splits/val_v5_combined_universal_kaggle_boost.csv`](../../data/splits/val_v5_combined_universal_kaggle_boost.csv) (**6,000 images**).
  - **Test CourseWork Balanced (1:1):** [`test_coursework_44methods_balanced_zero_leakage.csv`](../../data/splits/test_coursework_44methods_balanced_zero_leakage.csv) (**21,446 images** — 10,723 Real : 10,723 Fake, ~300/method).
  - **Test CourseWork Full Suite:** [`test_coursework_44methods_full_zero_leakage.csv`](../../data/splits/test_coursework_44methods_full_zero_leakage.csv) (**50,084 images** — 25,042 Real : 25,042 Fake).
  - **Hash Manifest:** [`train_v5_weakfix_v3_hashes.json`](../../data/splits/train_v5_weakfix_v3_hashes.json) (127,185 byte-level MD5 hashes).

---

## 3. Zero-Leakage Auditing Protocols

1. **Protocol 1 (Path Disjoint):** $|Train \cap Test_{Bal}| = 0$, $|Train \cap Test_{Full}| = 0$.
2. **Protocol 2 (Identity & Video Disjoint):** Video sequence IDs and subject identities are strictly partitioned.
3. **Protocol 3 (MD5 Byte-Level Collision Audit):** 5,680 path overlaps and 2,261 exact byte collision duplicates were eliminated.

---

## 4. Preprocessing & Normalization Pipeline

- **Image Size:** $256 \times 256$ pixels (Bicubic interpolation).
- **Normalization:** ImageNet mean $\mu = [0.485, 0.456, 0.406]$, std $\sigma = [0.229, 0.224, 0.225]$.
- **Transforms:** `torchvision.transforms.Compose([Resize((256, 256)), ToTensor(), Normalize(mean, std)])`.

---

## 5. References & Direct Links

- Master Training EDA Notebook: [`coursework_eda.ipynb`](../../notebooks/coursework_eda.ipynb)
- Master Evaluation Benchmark Notebook: [`coursework_deepfake.ipynb`](../../notebooks/coursework_deepfake.ipynb)
- Master Data Summary: [`DATA_SPLIT_SUMMARIZE.md`](DATA_SPLIT_SUMMARIZE.md)
- Data Prep Report: [`DATA_PREP_SUMMARY_REPORT.md`](DATA_PREP_SUMMARY_REPORT.md)
- EDA Inventory: [`EDA_DATA_INVENTORY.md`](EDA_DATA_INVENTORY.md)
- Status Tracker: [`DATA_PREP_STATUS.md`](../progress/DATA_PREP_STATUS.md)
