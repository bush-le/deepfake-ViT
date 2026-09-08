# MODEL_STATUS.md — Model Definitions & Checkpoint Status

- **Motivation/Background**: Provide real-time progress tracking, checklist status, and milestone completion for MODEL_STATUS.
- **Purpose**: Maintain an accurate audit trail of completed tasks and active blockers for MODEL_STATUS.
- **Overview Pipeline**: Milestone tracking -> task checklist review -> verification status update.
- **Detailed Plan**: §1 Current Milestone Status; §2 Completed Deliverables; §3 Active Blockers; §4 Next Priorities.
- **References**: `docs/OVERVIEW.md`, `docs/phases/`.
- **Created**: 2026-08-18T11:19:39+07:00
- **Last Updated**: 2026-09-08T10:15:00+07:00

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
