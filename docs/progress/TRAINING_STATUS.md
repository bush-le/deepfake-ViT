# TRAINING_STATUS.md — Training Execution Status

- **Motivation/Background**: Provide real-time progress tracking, checklist status, and milestone completion for TRAINING_STATUS.
- **Purpose**: Maintain an accurate audit trail of completed tasks and active blockers for TRAINING_STATUS.
- **Overview Pipeline**: Milestone tracking -> task checklist review -> verification status update.
- **Detailed Plan**: §1 Current Milestone Status; §2 Completed Deliverables; §3 Active Blockers; §4 Next Priorities.
- **References**: `docs/OVERVIEW.md`, `docs/phases/`.
- **Created**: 2026-08-18T11:19:39+07:00
- **Last Updated**: 2026-09-08T10:15:00+07:00

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
