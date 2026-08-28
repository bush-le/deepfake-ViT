# Comprehensive 40-Methods Evaluation Report (Identity-Disjoint Test v3)

Evaluation report comparing **Meta DINOv3 ViT-Small/16** and **Meta DINOv3 ConvNeXt-Tiny** across the multi-method benchmark suite.

---

## 1. Executive Summary

- **ViT-Small/16 Accuracy**: **97.66%** | ROC-AUC: **99.64%**
- **ConvNeXt-Tiny Accuracy**: **97.26%** | ROC-AUC: **99.58%**
- **Joint Ensemble Accuracy**: **97.80%** | ROC-AUC: **99.68%**

---

## 2. Categorical Performance Breakdown

1. **Diffusion Synthesis**: Both models achieve $>98.5\%$ detection rate on modern diffusion models (Midjourney v5/v6, Stable Diffusion XL, CollabDiff).
2. **Generative GANs**: StyleGAN2/3/XL and ProGAN achieve $>98.0\%$ detection due to pronounced high-frequency spectral artifacts.
3. **Face Swapping**: ConvNeXt-Tiny achieves slightly higher boundary detection on subtle blending seams, while ViT-S/16 provides superior global context modeling.
