# TRAINING_INFO.md — Training Protocols & Optimization Procedures

- **Title:** Training Loops, Optimization Schedules & Checkpointing
- **Date Created:** 2026-08-18
- **Last Updated:** 2026-08-28
- **Description:** Standalone script training procedures, Layer-wise Learning Rate Decay (LLRD), Balanced Batch Samplers, Label Smoothing, and Mixed-Precision Optimization.
- **Status:** **Completed & Checkpointed**

---

## 1. Background & Principles

Training is executed strictly through reproducible command-line scripts located in `src/training/` and `scripts/`. Interactive notebooks never execute full training loops; they consume artifacts produced by training scripts.

---

## 2. Optimization Pipeline & Techniques

```
Train Dataset (129,884 samples across 51 subsets)
  ↓
Weighted / Focused Random Sampler (Balanced Real/Fake Batches)
  ↓
Layer-wise Learning Rate Decay (LLRD γ = 0.80 across 12 Transformer layers)
  ↓
AdamW Optimizer (Base LR: 1.5e-5, Head LR: 4.0e-4, Weight Decay: 0.05)
  ↓
Label-Smoothed Cross-Entropy Loss (ε = 0.05) + PyTorch AMP (bfloat16)
  ↓
Cosine Annealing Learning Rate Scheduler with Warmup
  ↓
Validation Checkpointing on ROC-AUC / F1-Score
```

---

## 3. Key Hyperparameter Specifications

| Parameter | DINOv3 ViT-Small/16 | DINOv3 ConvNeXt-Tiny | Description |
| :--- | :--- | :--- | :--- |
| **Input Resolution** | $256 \times 256$ | $256 \times 256$ | Standard bicubic resized face crops |
| **Batch Size** | 64 / 128 | 64 / 128 | Fits comfortably in 4GB–8GB VRAM |
| **Optimizer** | AdamW | AdamW | Betas: `(0.9, 0.999)`, eps: `1e-8` |
| **Base LR (Backbone)** | $1.5 \times 10^{-5}$ | $1.5 \times 10^{-5}$ | Preserves foundational representations |
| **Head LR** | $4.0 \times 10^{-4}$ | $4.0 \times 10^{-4}$ | Fast convergence for classification head |
| **LLRD Decay Factor ($\gamma$)** | $0.80$ | N/A (Stage-wise) | $\eta_l = \text{base\_lr} \cdot \gamma^{11-l}$ |
| **Weight Decay** | $0.05$ | $0.05$ | Prevents parameter explosion |
| **Label Smoothing ($\epsilon$)** | $0.05$ | $0.05$ | Softens binary cross-entropy targets |
| **Precision** | `bfloat16` / `fp16` | `bfloat16` / `fp16` | Accelerated mixed precision |

---

## 4. Primary Training Entry Points

```bash
# 1. Standard differential fine-tuning (ViT)
.venv/bin/python src/training/finetune_compare.py \
    --model vit \
    --train_csv data/splits/train_v5_weakfix_v3.csv \
    --val_csv data/splits/val_v5_combined_universal_kaggle_boost.csv \
    --epochs 5 --batch_size 64 --lr 1.5e-5 --amp

# 2. ConvNeXt fine-tuning
.venv/bin/python src/training/finetune_compare.py \
    --model convnext \
    --train_csv data/splits/train_v5_weakfix_v3.csv \
    --val_csv data/splits/val_v5_combined_universal_kaggle_boost.csv \
    --epochs 5 --batch_size 64 --lr 1.5e-5 --amp
```

---

## 5. Checkpoint Verification

- **ViT Best Checkpoint:** `experiments/checkpoints/best_model_v3.pt` (Epoch 3, Val AUC: `0.9940`).
- **ConvNeXt Best Checkpoint:** `experiments/checkpoints/convnext_weakfix_v3.pt` (Epoch 1, Val AUC: `0.9997`).

---

## 6. Links & References

- Technical Guide: [`../TECHNICAL_GUIDE.md`](../TECHNICAL_GUIDE.md)
- Status Tracker: [`../progress/TRAINING_STATUS.md`](../progress/TRAINING_STATUS.md)
- Logging Rules: [`../rules/LOGGING_CHECKPOINT_RULES.md`](../rules/LOGGING_CHECKPOINT_RULES.md)
