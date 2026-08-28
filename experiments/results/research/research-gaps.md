# Identified Research Gaps & Proposed Solutions

---

## 1. Key Research Gaps in Deepfake Detection
1. **Out-of-Distribution Diffusion Generalization**: Detectors trained exclusively on GANs fail when deployed against modern diffusion models.
   - *Solution*: Integrated Kaggle Midjourney Boost with 11 distinct diffusion algorithms.
2. **Data Leakage & Inflated Benchmark Scores**: Subject and video overlap between splits distorts evaluation.
   - *Solution*: Engineered a certified 4-tier Zero-Leakage pipeline with 127k MD5 byte hash verification.
3. **Over-reliance on Superficial Compression Shortcuts**: Models classifying low-resolution images as fakes.
   - *Solution*: Implemented artifact-preserving data augmentations.
