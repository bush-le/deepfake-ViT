# Domain-Agnostic Data Augmentation Specification

This document defines the official data augmentation pipeline for training robust deepfake detectors without destroying discriminative high-frequency artifacts.

---

## 1. Principles of Forensic-Safe Augmentation
1. **Artifact Preservation**: Do NOT apply aggressive Gaussian blur, strong downsampling, or severe JPEG compression ($\text{quality} < 60$), which destroys discriminative GAN checkerboard patterns and boundary blending seams.
2. **Photometric Robustness**: Apply mild brightness/contrast jittering to prevent the model from overfitting to specific camera sensor lighting profiles.
3. **Geometric Invariance**: Standard horizontal flips and subtle affine transformations maintain natural facial geometry while expanding spatial variety.

---

## 2. Recommended Training Augmentation Pipeline

```python
import torchvision.transforms as transforms

train_transform = transforms.Compose([
    transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(
        brightness=0.1,
        contrast=0.1,
        saturation=0.05,
        hue=0.02
    ),
    transforms.RandomAffine(
        degrees=5,
        translate=(0.02, 0.02),
        scale=(0.98, 1.02)
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])
```

---

## 3. Evaluation & Inference Transform

```python
eval_transform = transforms.Compose([
    transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])
```

---

## 4. Verification & Validation Metrics
- Ensure validation loss stability without augmentation-induced divergence.
- Validate that models trained with this pipeline maintain $>97\%$ accuracy across all 44 test methods in [`notebooks/coursework_deepfake.ipynb`](../../notebooks/coursework_deepfake.ipynb).
