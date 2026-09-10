# Model Architecture & Dataset Selection Rationale

---

## 1. Architectural Selection: Meta DINOv3 ViT-Small/16
- **Self-Supervised Feature Robustness**: Pretrained on billions of curated images without supervised label bias.
- **Global Spatial Self-Attention**: Captures non-local structural incoherence and semantic inconsistencies across synthetic faces.
- **Computational Efficiency**: 28.69M parameters (DINOv3 ViT-S/16 Plus with SwiGLU Gated MLP) with 256 visual tokens + 4 register tokens, strictly matched to ConvNeXt-Tiny (28.12M) under GPU memory constraints.

---

## 2. Baseline Architecture: Meta DINOv3 ConvNeXt-Tiny
- Modern convolutional network featuring $7	imes 7$ depthwise convolutions, serving as the local receptive field benchmark against ViT.
