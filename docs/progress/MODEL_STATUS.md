# MODEL_STATUS.md — Model Definitions & Checkpoint Status

- **Title:** Model Definitions & Checkpoint Verification (DINOv3 ViT / ConvNeXt / LoRA / Ensemble)
- **Date Created:** 2026-08-18
- **Last Updated:** 2026-08-28
- **Description:** Status of model architectures, parameter counts, and saved checkpoints.
- **Status:** **Done & Verified**
- **Phase Doc:** [`../phases/MODEL.md`](../phases/MODEL.md)

---

## Log

- **2026-08-18:** DINOv3 ViT-Small/16, ConvNeXt-Tiny CNN, and LoRA adapters implemented and verified with `strict=True`.
- **2026-08-22:** Parameter count parity verified (ViT: 21.60M vs. ConvNeXt: 28.12M).
- **2026-08-24:** Integrated `DinoConvNextClassifier` with 2-layer GELU MLP head (`classifier_v2.py`).
- **2026-08-28:** Verified both checkpoints on disk:
  - `experiments/checkpoints/best_model_v3.pt` (DINOv3 ViT-Small/16, Best Val AUC: `0.9940`).
  - `experiments/checkpoints/convnext_weakfix_v3.pt` (DINOv3 ConvNeXt-Tiny, Best Val AUC: `0.9997`).
- **2026-08-28:** Integrated both models side-by-side into [`notebooks/coursework_deepfake.ipynb`](../../notebooks/coursework_deepfake.ipynb).

---

## Blockers

- None.

---

## Decisions

- Use frozen weights for both models during evaluation and benchmarking.
- Both models take $256 \times 256$ RGB inputs with standard ImageNet normalization.

---

## Links

- Phase Doc: [`../phases/MODEL.md`](../phases/MODEL.md)
- Technical Guide: [`../TECHNICAL_GUIDE.md`](../TECHNICAL_GUIDE.md)
- Checkpoints: `experiments/checkpoints/`
