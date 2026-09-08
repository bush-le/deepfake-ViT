# MODEL.md — Model Definitions & Architecture Specifications

- **Motivation/Background**: Establish technical specification, requirements, and deliverables for pipeline phase MODEL.
- **Purpose**: Guide the execution and quality gates of phase MODEL.
- **Overview Pipeline**: Phase scope definition -> implementation guidelines -> verification gates -> status sign-off.
- **Detailed Plan**: §1 Phase Overview & Scope; §2 Technical Specification; §3 Deliverables & Artifacts; §4 Verification Protocol.
- **References**: `docs/OVERVIEW.md`, `docs/PURPOSE.md`, `agents/rules/`.
- **Created**: 2026-08-18T11:19:39+07:00
- **Last Updated**: 2026-09-08T10:15:00+07:00

---

## 1. Background & Design Rationale

To assess deepfake detection capabilities under different inductive biases, the project implements two comparable architectures:
1. **Self-Supervised Vision Transformer (Meta DINOv3 ViT-Small/16):** Captures long-range spatial dependencies and global self-attention anomalies.
2. **Modern Convolutional Network (Meta DINOv3 ConvNeXt-Tiny):** Exploits localized convolutional inductive biases to spot boundary blending and high-frequency edge artifacts.

---

## 2. Model Specifications

### 2.1 Meta DINOv3 ViT-Small/16 (`src/models/dinov3_vit.py`)
- **Backbone:** Self-Supervised DINOv3 ViT-S/16 (`embed_dim=384`, `depth=12`, `num_heads=6`, 4 register tokens, SwiGLU MLP).
- **Classification Head:** Single linear layer `Linear(384, 2)` mapped to binary logits `[Real, Fake]`.
- **Parameter Count:** **21,602,306** (~21.60M params).
- **Checkpoint:** [`experiments/checkpoints/best_model_v3.pt`](../../experiments/checkpoints/best_model_v3.pt).

### 2.2 Meta DINOv3 ConvNeXt-Tiny (`src/models/dinov3_convnext.py` & `src/models/classifier_v2.py`)
- **Backbone:** ConvNeXt-Tiny architecture with 4 stages (`depths=[3, 3, 9, 3]`, `dims=[96, 192, 384, 768]`, $7\times 7$ depthwise convolutions, LayerScale $\gamma$).
- **Classification Head (`DinoConvNextClassifier`):**
  $$\text{LayerNorm(768)} \rightarrow \text{Dropout(0.2)} \rightarrow \text{Linear(768, 384)} \rightarrow \text{GELU} \rightarrow \text{Dropout(0.1)} \rightarrow \text{Linear(384, 2)}$$
- **Parameter Count:** **28,124,354** (~28.12M params).
- **Checkpoint:** [`experiments/checkpoints/convnext_weakfix_v3.pt`](../../experiments/checkpoints/convnext_weakfix_v3.pt).

### 2.3 Joint Inference Ensemble (`src/models/classifier_v2.py`)
- Combines probability outputs: $P_{\text{ensemble}} = 0.65 \cdot P_{\text{ViT}} + 0.35 \cdot P_{\text{CNN}}$.
- Fuses global attention contextual cues with sharp localized convolutional edge detection.

### 2.4 LoRA Adapter (`src/models/lora.py`)
- Low-Rank Adaptation applied to query/value projection matrices in ViT self-attention blocks ($r=8, \alpha=16$).

---

## 3. Loading & Execution Patterns

```python
# Load ViT
from src.models.dinov3_vit import build_dinov3_classifier
model_vit = build_dinov3_classifier(weights_path=None, num_classes=2, img_size=256, device="cuda")
model_vit.load_state_dict(torch.load("experiments/checkpoints/best_model_v3.pt")["model_state_dict"], strict=False)

# Load ConvNeXt
from src.models.dinov3_convnext import DinoConvNext
from src.models.classifier_v2 import DinoConvNextClassifier
model_cnn = DinoConvNextClassifier(DinoConvNext(), num_classes=2, hidden_dim=384)
model_cnn.load_state_dict(torch.load("experiments/checkpoints/convnext_weakfix_v3.pt")["model_state_dict"], strict=False)
```

---

## 4. Links & References

- Technical Guide: [`../TECHNICAL_GUIDE.md`](../TECHNICAL_GUIDE.md)
- Status Tracker: [`../progress/MODEL_STATUS.md`](../progress/MODEL_STATUS.md)
- Evaluation Notebook: [`../../notebooks/coursework_deepfake.ipynb`](../../notebooks/coursework_deepfake.ipynb)
