# TRAINING_STATUS.md — Training Execution Status

- **Title:** Training Loops, Hyperparameter Optimization & Fine-tuning Status
- **Date Created:** 2026-08-18
- **Last Updated:** 2026-08-28
- **Description:** Status of training procedures, fine-tuning milestones, and saved checkpoints.
- **Status:** **Done & Checkpointed**
- **Phase Doc:** [`../phases/TRAINING_INFO.md`](../phases/TRAINING_INFO.md)

---

## Log

- **2026-08-18:** Full RNG seeding and differential learning rate fine-tuning implemented.
- **2026-08-22:** Implemented Layer-wise Learning Rate Decay (LLRD $\gamma = 0.80$) and Label Smoothing ($\epsilon = 0.05$).
- **2026-08-24:** Completed Universal Balanced Training on 129.8k samples (`train_v5_weakfix_v3.csv`) for both ViT and ConvNeXt backbones.
- **2026-08-28:** Final checkpoints validated and verified on disk:
  - ViT Checkpoint: `experiments/checkpoints/best_model_v3.pt` (Best Val AUC: `0.9940`).
  - ConvNeXt Checkpoint: `experiments/checkpoints/convnext_weakfix_v3.pt` (Best Val AUC: `0.9997`).

---

## Blockers

- None. Both models successfully converged and are checkpointed for offline evaluation.

---

## Decisions

- Retain frozen checkpoints `best_model_v3.pt` and `convnext_weakfix_v3.pt` for direct inference.
- Multi-worker setting: Use `num_workers=0` in interactive environments to prevent Python 3.14+ forkserver pickling restrictions.

---

## Links

- Phase Doc: [`../phases/TRAINING_INFO.md`](../phases/TRAINING_INFO.md)
- Checkpoints: `experiments/checkpoints/`
- Technical Guide: [`../TECHNICAL_GUIDE.md`](../TECHNICAL_GUIDE.md)
