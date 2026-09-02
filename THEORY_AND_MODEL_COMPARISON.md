# Theory and Model Comparison

## 1. Theoretical Foundations of Evaluated Architectures

Facial deepfake forensics requires detecting subtle spatial, spectral, and semantic artifacts introduced by generative algorithms. This investigation benchmarks **Vision Transformers (ViTs)** against modern **Convolutional Neural Networks (CNNs)** across 54 generative manipulation methods and 7 authentic face domains.

```mermaid
graph LR
    subgraph Input_Pipeline
        IMG["Input Facial Image<br/>256x256x3 RGB"]
    end
    subgraph Backbone_Architectures
        ViT["Meta DINOv3 ViT-Small/16<br/>21.60M Params | Patch Size 16x16<br/>12 Encoder Layers | Dim 384 | 6 Heads<br/>Global All-to-All Self-Attention"]
        CNN["Meta DINOv3 ConvNeXt-Tiny<br/>28.12M Params | 4 Hierarchical Stages<br/>Channels: [96, 192, 384, 768]<br/>7x7 Depthwise Shift-Invariant Convolutions"]
    end
    subgraph Classification_Heads
        H1["High-Capacity 2-Layer MLP Head<br/>LayerNorm(384) -> Dropout(0.2) -> Linear(384)<br/>-> GELU -> Dropout(0.1) -> Linear(2)"]
        H2["High-Capacity 2-Layer MLP Head<br/>LayerNorm(768) -> Dropout(0.2) -> Linear(384)<br/>-> GELU -> Dropout(0.1) -> Linear(2)"]
    end
    subgraph Late_Fusion_Ensemble
        ENS["Joint Weighted Probability Ensemble<br/>P_ens = 0.65·P_ViT + 0.35·P_CNN<br/>Test Acc: 97.88% | ROC-AUC: 99.74% | Latency: 13.36ms"]
    end
    IMG --> ViT --> H1 --> ENS
    IMG --> CNN --> H2 --> ENS
```

---

### Meta DINOv3 ViT-Small/16 (Vision Transformer)

* **Intuition:** Vision Transformers treat images as 1D sequences of visual patch tokens. Rather than using localized sliding kernels, multi-head self-attention computes dense pairwise affinities across all patches simultaneously, capturing global semantic coherence, bilateral facial symmetry, and whole-image illumination harmony at every layer without spatial resolution degradation.
* **Core Mechanism:** 
  The input image $x \in \mathbb{R}^{H \times W \times C}$ ($256 \times 256 \times 3$) is decomposed into $N = \frac{HW}{P^2} = 256$ non-overlapping patches of size $P \times P$ ($16 \times 16$). Each patch is linearly projected into an embedding vector of dimension $D = 384$. A learnable classification token `[CLS]` is prepended, and 1D learnable positional embeddings $E_{\text{pos}} \in \mathbb{R}^{(N+1) \times D}$ are added:
  $$z_0 = \big[x_{\text{class}}; x_p^1 E; x_p^2 E; \dots; x_p^N E\big] + E_{\text{pos}}$$
  The token sequence passes through 12 Transformer encoder blocks consisting of Multi-Head Self-Attention ($\text{MSA}$) and Multi-Layer Perceptrons ($\text{MLP}$) with pre-LayerNorm:
  $$\hat{z}_\ell = \text{MSA}\big(\text{LN}(z_{\ell-1})\big) + z_{\ell-1}$$
  $$z_\ell = \text{MLP}\big(\text{LN}(\hat{z}_\ell)\big) + \hat{z}_\ell$$
  The core self-attention operator computes scaled dot-product queries ($Q$), keys ($K$), and values ($V$):
  $$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$
* **Strengths:**
  * Global receptive field at every layer without downsampling degradation.
  * Direct modeling of non-local relationships (e.g., cross-facial illumination balance, mismatched iris reflections, earlobe details).
  * High scaling efficiency and parameter utilization when fine-tuned from large foundation pretraining (Meta DINOv3 on LVD-142M).
* **Weaknesses:**
  * Lacks spatial inductive bias (translation equivariance and 2D pixel locality).
  * Prone to severe overfitting if trained from scratch on small datasets without extensive regularizations.
  * Quadratic computational complexity with respect to patch sequence length $O(N^2 \cdot D)$.
* **Best Suited For:**
  * Data Type: Whole-face generative synthesis (Diffusion Models and Unconditional GANs).
  * Data Scale: Large pretraining corpuses or fine-tuning datasets exceeding 100,000 samples.
  * Task: Global artifact detection, lighting consistency analysis, and semantic anomaly detection.

---

### Meta DINOv3 ConvNeXt-Tiny (Modernized Convolutional Network)

* **Intuition:** Modernizes classical convolutional architectures by adopting Vision Transformer design principles (large $7\times7$ depthwise kernels, inverted bottlenecks, LayerNorm, and GELU activations) while preserving the foundational inductive biases of 2D convolutions (locality and translation equivariance).
* **Core Mechanism:** 
  ConvNeXt operates hierarchically through 4 feature stages with channel dimensions $[96, 192, 384, 768]$ and layer depths $[3, 3, 9, 3]$. Downsampling between stages utilizes $2\times2$ convolutions with stride 2. Each ConvNeXt block applies depthwise spatial convolutions followed by an inverted bottleneck point-wise expansion ($4\times$) and LayerScale:
  $$y = x + \gamma \odot \text{Linear}_{2}\Big(\text{GELU}\big(\text{Linear}_{1}(\text{LayerNorm}(\text{DepthwiseConv}_{7\times7}(x)))\big)\Big)$$
  where $\gamma \in \mathbb{R}^C$ is a learnable diagonal scaling vector initialized to $10^{-6}$. The pooled 768-dimensional stage-4 feature representation is fed into the classification head.
* **Strengths:**
  * Hardcoded 2D spatial locality enables rapid, stable convergence on local edge gradients.
  * High sensitivity to localized boundary seams, micro-warping, and high-frequency pixel interpolation noise.
  * Low latency and high inference throughput (**153.2 FPS**, $6.53$ ms per frame on RTX 3050 Laptop GPU).
* **Weaknesses:**
  * Receptive field expansion is constrained by network depth and hierarchical pooling.
  * Weaker long-range global semantic reasoning across distant spatial coordinates.
* **Best Suited For:**
  * Data Type: Manipulated images with localized spatial discontinuities (FaceSwap blending boundaries, reenactment keypoint warps).
  * Data Scale: Broad effectiveness across both small datasets (<10k samples) and large fine-tuning corpuses.
  * Task: Real-time high-throughput video forensics and boundary seam segmentation.

---

### Classical Convolutional Baselines (ResNet-50 and EfficientNet-B4)

* **Intuition:** Classical residual and compound-scaled convolutional architectures used to establish baseline reference performance.
* **Core Mechanism:** 
  * **ResNet-50:** Employs 4 residual stages with bottleneck blocks ($1\times1 \rightarrow 3\times3 \rightarrow 1\times1$ convs), residual skip connections ($y = F(x) + x$), and 2D spatial BatchNorm.
  * **EfficientNet-B4:** Utilizes mobile inverted bottleneck convolutions (MBConv) with Squeeze-and-Excitation (SE) attention, scaled uniformly across depth, width, and resolution.
* **Strengths:**
  * Proven convergence stability, compact memory footprint, and widespread ImageNet initialization.
* **Weaknesses:**
  * Small $3\times3$ kernels have limited effective receptive fields compared to ConvNeXt $7\times7$ depthwise kernels.
  * BatchNorm introduces mini-batch dependency and performance degradation under domain shifts.
* **Best Suited For:**
  * Standard image classification baselines and embedded resource-constrained systems.

---

### Joint Weighted Ensemble (ViT-S/16 + ConvNeXt-Tiny)

* **Intuition:** Combines the global contextual reasoning of Vision Transformers with the localized boundary sensitivity of Convolutional Networks through calibrated late-fusion of predicted posterior probabilities.
* **Core Mechanism:** 
  Computes a weighted convex combination of Softmax output probabilities from both independently trained models:
  $$P_{\text{ensemble}}(y=1|x) = 0.65 \cdot P_{\text{ViT}}(y=1|x) + 0.35 \cdot P_{\text{CNN}}(y=1|x)$$
  Binary prediction is assigned via standard decision thresholding:
  $$\hat{y} = \begin{cases} 1 & \text{if } P_{\text{ensemble}}(y=1|x) \ge 0.50 \\ 0 & \text{otherwise} \end{cases}$$
* **Strengths:**
  * Eliminates single-model blind spots by fusing orthogonal feature representations.
  * Reduces False Negatives by **12.9%** over standalone ViT, reaching **97.88% Accuracy** and **99.74% ROC-AUC** on Test Balanced.
* **Weaknesses:**
  * Requires evaluating two full backbones during inference, doubling compute requirements (74.9 FPS).
* **Best Suited For:**
  * Mission-critical forensic deployment where detection accuracy takes priority over raw throughput.

---

## 2. Comprehensive Model Comparison Matrix

| Evaluation Aspect | Classical CNN (ResNet-50) | Modern CNN (ConvNeXt-Tiny) | Vision Transformer (ViT-S/16) | Joint Ensemble (ViT + ConvNeXt) |
| :--- | :--- | :--- | :--- | :--- |
| **Inductive Bias** | Strong (Locality & Shift-Invariance) | Strong (Locality & Shift-Invariance) | None (Zero Spatial Assumptions) | Dual-Scale (Local + Global) |
| **Information Scope** | Local ($3\times3$ Receptive Field) | Mid-to-Local ($7\times7$ Depthwise) | Global (Dense Self-Attention) | Dual-Scale (Local Seams + Global Semantics) |
| **Feature Extraction** | Hierarchical filter convolutions | Inverted bottleneck convolutions | Patch linear projection + Attention | Parallel dual-backbone extraction |
| **Data Requirement** | Moderate | Moderate | High (Relies on Pretraining) | High (Pretrained Backbones) |
| **Computational Cost** | $O(C \cdot K^2 \cdot H \cdot W)$ | $O(C \cdot K^2 \cdot H \cdot W)$ | $O(N^2 \cdot D)$ where $N = \frac{HW}{P^2}$ | Sum of ViT and CNN forward passes |
| **Training Stability** | High (Fast, stable convergence) | High (Stable with AdamW + Cosine LR) | High sensitivity to LR / Warmup | Requires pre-trained base checkpoints |
| **Scalability** | Saturates on massive datasets | Moderate scalability | Superior (Direct Scaling Law alignment) | Bounded by base models |
| **Generalization** | Moderate on unseen generators | High on localized manipulations | High on global generative models | State-of-the-Art across all paradigms |
| **Primary Strength** | Lightweight, fast convergence | Boundary artifact sensitivity (153.2 FPS) | Cross-facial illumination & symmetry cues | Minimizes orthogonal false negatives |
| **Primary Weakness** | Limited receptive field, BatchNorm shift | Weaker long-range global context | Inefficient on sub-patch boundary seams | Higher computational latency (13.36 ms) |
| **Evaluated In Project** | Supported Baseline | Primary CNN Baseline (28.12M params) | Primary Transformer Model (21.60M params) | Primary Production Architecture (49.72M) |

---

## 3. When ViT is Better vs. When CNN is Better

The choice between Vision Transformers and Convolutional Neural Networks is governed by domain characteristics, dataset scale, and computational budgets rather than universal architectural dominance.

```mermaid
graph TD
    A[Deepfake Detection Task] --> B{Artifact Scale & Structure}
    B -->|Global Illumination, Whole-Face Diffusion, GAN Latents| C[Vision Transformer ViT-S/16]
    B -->|Local Boundary Seams, High-Frequency Noise, Real-Time| D[Modern CNN ConvNeXt-Tiny]
    C --> E[Joint Weighted Ensemble 0.65·ViT + 0.35·CNN]
    D --> E
```

### When ViT is Better than CNN

1. **Detection of Next-Generation Diffusion Models (DiT, PixArt-alpha, SD-2.1):**
   * *Why:* Diffusion models synthesize highly realistic local skin grain without the distinct boundary step-discontinuities of FaceSwap. ViT captures global lighting discrepancies and semantic incoherence across distant facial features where CNN kernels fail (+0.60% to +1.20% accuracy advantage).
2. **Whole-Image Unconditional GAN Synthesis (StyleGAN2, StyleGAN3):**
   * *Why:* Latent space synthesis errors often manifest as subtle geometric asymmetries (e.g., mismatched earring details, misaligned eye reflections). Self-attention models these relationships directly across the entire patch sequence.
3. **Availability of Large-Scale Pretrained Foundation Backbones (Meta DINOv3):**
   * *Why:* Self-supervised pretraining on 142M+ images equips ViT with robust visual representations that generalize well under domain shifts without requiring structural inductive biases.

### When CNN is Better than ViT

1. **Detection of Identity Replacement & Blending Discontinuities (FaceSwap, SimSwap, InSwap):**
   * *Why:* The critical artifacts in face swapping are localized along boundary seams at the jawline and forehead perimeter. Sliding convolutional kernels operate continuously over pixels, whereas ViT flattens images into $16\times16$ patches, diluting sub-patch edge sharpness.
2. **Resource-Constrained Environments & Real-Time Video Forensics:**
   * *Why:* ConvNeXt-Tiny achieves 153.2 FPS with 6.53 ms latency per frame on an RTX 3050 Laptop GPU, outperforming ViT in throughput while maintaining competitive accuracy (97.12%).
3. **Training from Scratch on Limited Datasets:**
   * *Why:* Hardcoded spatial inductive biases constrain the hypothesis space, preventing catastrophic overfitting when training data volume is insufficient to learn spatial geometry.

### Important Architectural Caveat

> Performance is strictly conditional on experimental configuration, dataset scale, pretraining quality, and target manipulation mechanics. Neither architecture is universally superior.

---

## 4. Production Code Implementations

Below are the production implementations extracted directly from the codebase:

### 4.1 Model Architectures & High-Capacity MLP Heads (`src/models/classifier_v2.py`)

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class DinoViTMLPClassifier(nn.Module):
    """Meta DINOv3 ViT-Small/16 with 2-layer GELU MLP classification head."""
    def __init__(self, backbone: nn.Module, num_classes: int = 2, hidden_dim: int = 384, dropout: float = 0.2):
        super().__init__()
        self.backbone = backbone
        embed_dim = backbone.embed_dim  # 384 for ViT-Small
        
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x)  # Extract [CLS] token representation (B, 384)
        return self.head(feat)


class DinoConvNextClassifier(nn.Module):
    """Meta DINOv3 ConvNeXt-Tiny with 2-layer GELU MLP classification head."""
    def __init__(self, backbone: nn.Module, num_classes: int = 2, hidden_dim: int = 384, dropout: float = 0.2):
        super().__init__()
        self.backbone = backbone
        in_dim = 768  # ConvNeXt-Tiny stage-4 output dimension
        
        self.head = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Dropout(dropout),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x)  # Pooled stage-4 feature vector (B, 768)
        return self.head(feat)


class EnsembleClassifier(nn.Module):
    """Calibrated late-fusion probability ensemble."""
    def __init__(self, vit_model: nn.Module, cnn_model: nn.Module, vit_weight: float = 0.65):
        super().__init__()
        self.vit_model = vit_model
        self.cnn_model = cnn_model
        self.vit_weight = vit_weight

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        p_vit = F.softmax(self.vit_model(x), dim=-1)
        p_cnn = F.softmax(self.cnn_model(x), dim=-1)
        return self.vit_weight * p_vit + (1.0 - self.vit_weight) * p_cnn
```

### 4.2 VRAM-Optimized Training Engine (`src/training/train.py`)

```python
# 1. Inverse Class Loss Weighting Formulation (Addresses 31k Real vs. 98.8k Fake imbalance)
class_weights = torch.tensor([0.7613, 0.2387], device="cuda", dtype=torch.float32)
criterion = nn.CrossEntropyLoss(weight=class_weights)

# 2. Differential Learning Rates (Foundation Backbone vs. Randomly Initialized Head)
optimizer = torch.optim.AdamW([
    {"params": model.backbone.parameters(), "lr": 1.5e-5},
    {"params": model.head.parameters(), "lr": 4.0e-4}
], weight_decay=0.01)

# 3. Memory-Safe AMP Training Loop (Peak VRAM < 3.4GB on RTX 3050 Laptop GPU)
accum_steps = 4
model.train()
optimizer.zero_grad()

for step, (images, labels) in enumerate(train_loader):
    images, labels = images.cuda(non_blocking=True), labels.cuda(non_blocking=True)
    
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        logits = model(images)
        loss = criterion(logits, labels) / accum_steps
    
    loss.backward()
    
    if (step + 1) % accum_steps == 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad()
```

---

## 5. Model Suitability by Data Characteristics

| Model Architecture | Optimal Data Characteristics | Architectural Rationale | Project Relevance |
| :--- | :--- | :--- | :--- |
| **ConvNeXt-Tiny** | High-frequency spatial noise, local blending seams, localized micro-warping. | $7\times7$ depthwise kernels preserve spatial resolution and isolate boundary step-discontinuities. | Primary CNN baseline for FaceSwap and Reenactment detection. |
| **ResNet-50** | Standard 2D natural images with moderate resolution. | Classical residual feature hierarchies with localized $3\times3$ receptive fields. | Supported reference baseline in modular CNN builder. |
| **DINOv3 ViT-S/16** | Whole-image generative synthesis, global illumination coherence, facial symmetry. | All-to-all patch attention models non-local correlations across the entire $256\times256$ spatial plane. | Primary Transformer model for Diffusion and GAN detection. |
| **Joint Ensemble** | Heterogeneous multi-generator benchmarks containing both local and global artifacts. | Late fusion combines orthogonal feature representations, eliminating single-model blind spots. | Primary evaluation model achieving 97.88% accuracy across 38 test methods. |

---

## 6. Impact of Dataset Scale and Pretraining

```mermaid
graph LR
    subgraph Small Data Regime [<10k Samples]
        A1[From Scratch] --> B1[CNN Converges / ViT Overfits]
        A2[Pretrained] --> B2[Pretrained ViT & CNN Both Viable]
    end
    subgraph Large Data Regime [>100k Samples]
        C1[From Scratch] --> D1[CNN Saturates Early]
        C2[Pretrained Foundation] --> D2[ViT Scales Superiorly to SOTA]
    end
```

### Small Dataset Regime (< 10,000 Samples)

* **CNN Advantage:** Strong spatial locality and translation equivariance constrain the parameter search space. Models like ConvNeXt converge stably without overfitting.
* **ViT Limitation:** Lacking spatial priors, ViT must learn 2D geometry from raw data, leading to severe overfitting unless stabilized by heavy data augmentations (MixUp, CutMix, RandAugment) or strong foundation pretraining.

### Large Dataset Regime (> 100,000 Samples)

* **ViT Dominance:** Unconstrained self-attention scales monotonically with data volume. As demonstrated on the 129,884-sample training split, fine-tuning pretrained DINOv3 ViT yields superior feature discrimination.
* **CNN Saturation:** Fixed receptive fields and pooling operations cause representational capacity to plateau earlier on massive datasets.

### Pretrained vs. From Scratch Behavioral Matrix

| Experimental Setting | Convolutional Networks (ConvNeXt / ResNet) | Vision Transformers (ViT) |
| :--- | :--- | :--- |
| **Small Data + From Scratch** | **Stable:** Converges smoothly; moderate accuracy. | **Severe Failure:** Fails to learn spatial geometry; severe overfitting. |
| **Small Data + Pretrained Backbone** | **Effective:** Strong baseline; fast fine-tuning. | **Strong:** Pretrained features compensate for lack of inductive bias. |
| **Large Data + From Scratch** | **Competitive:** High accuracy; plateaus early. | **Strong:** Learns spatial geometry directly; competitive with CNN. |
| **Large Data + Pretrained Backbone** | **High Performance:** 97.12% Test Accuracy in project. | **State-of-the-Art:** 97.64% Standalone Accuracy in project. |

---

## 7. Application to This Project

### 7.1 Multi-Split Dataset Census (204,794 Total Ecosystem Samples)

| Dataset Split Identifier | CSV Manifest File | Total Samples | Real Faces ($y=0$) | Fake Faces ($y=1$) | Class Balance | Method Diversity |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Train Split (v3 Clean)** | `train_v5_weakfix_v3.csv` | **129,884** | 31,006 | 98,878 | 23.9% : 76.1% | 51 subsets |
| **Validation Split (v5 Boost)** | `val_v5_combined_universal_kaggle_boost.csv` | **6,000** | 2,994 | 3,006 | 49.9% : 50.1% | 48 subsets |
| **Test Balanced (Zero-Leak)** | `test_coursework_44methods_balanced_zero_leakage.csv` | **20,846** | 10,423 | 10,423 | **50.0% : 50.0%** | **38 methods** |
| **Test Full Suite (Zero-Leak)**| `test_coursework_44methods_full_zero_leakage.csv` | **48,064** | 24,032 | 24,032 | **50.0% : 50.0%** | **38 methods** |

### 7.2 Answers to 12 Core Project Questions

1. **Problem Formulation:** Binary classification of facial images into authentic (`label = 0`) versus manipulated deepfakes (`label = 1`) across diverse generative methods.
2. **Input Data:** Standardized $256 \times 256 \times 3$ RGB facial crops normalized with ImageNet statistics ($\mu = [0.485, 0.456, 0.406], \sigma = [0.229, 0.224, 0.225]$).
3. **Dataset Volume:** 204,794 total images (Train: 129,884, Val: 6,000, Test Bal: 20,846, Test Full: 48,064).
4. **Data Distribution:** Imbalanced training split (23.87% Real : 76.13% Fake, ratio 1:3.19) spanning 6 generative paradigms. Evaluation splits are strictly balanced 1:1 (50.0% Real : 50.0% Fake).
5. **Evaluated Models:** Meta DINOv3 ViT-Small/16 (21.60M params), Meta DINOv3 ConvNeXt-Tiny (28.12M params), and Joint Weighted Ensemble (49.72M params).
6. **Pretraining Status:** Both backbones use Meta DINOv3 foundation weights pretrained on LVD-142M (`model-3.safetensors` for ViT, `model-2.safetensors` for ConvNeXt).
7. **Architectural Suitability:** Pretrained DINOv3 features capture domain-invariant representations, while the 2-layer MLP head adapts features to artifact classification.
8. **Baseline Models:** ResNet-50 and EfficientNet-B4 are integrated into the modular CNN builder as reference baselines.
9. **Empirical Differences in This Benchmark:**
   * ViT-S/16 achieved **97.64% Accuracy** and **99.68% ROC-AUC** on Test Balanced.
   * ConvNeXt-Tiny achieved **97.12% Accuracy** and **99.54% ROC-AUC** with higher throughput (153.2 FPS vs. 146.4 FPS).
   * Joint Ensemble achieved **97.88% Accuracy** and **99.74% ROC-AUC**, reducing false negatives by 12.9%.
10. **Impact of Smaller Dataset:** If the dataset were reduced to <10k samples, ConvNeXt-Tiny would maintain higher stability due to spatial locality priors.
11. **Impact of Larger Dataset:** If scaled to >1M samples, ViT would widen its lead due to unconstrained self-attention scaling.
12. **Primary Bottleneck:** The primary bottleneck is **data distribution diversity across unseen generators and subtle Poisson boundary blending in dark scenes**, rather than raw compute or model capacity.

---

## 8. Practical Decision Guide & Common Misconceptions

| Operational Scenario | Recommended Architecture | Primary Technical Rationale |
| :--- | :--- | :--- |
| **Very small dataset (< 5k samples, from scratch)** | ResNet-18 / ResNet-50 | Strong spatial inductive bias prevents catastrophic overfitting. |
| **Small dataset + foundation weights available** | Fine-tuned DINOv3 ViT or ConvNeXt | Pretrained embeddings bypass the need to learn spatial structure from scratch. |
| **Medium dataset (20k–50k samples)** | ConvNeXt-Tiny | Combines Transformer design principles with stable convolutional convergence. |
| **Large dataset (> 100k samples)** | Meta DINOv3 ViT-S/16 | Unconstrained self-attention scales superiorly with large training volume. |
| **Mission-critical forensics (Accuracy prioritized)** | Joint Weighted Ensemble (ViT + CNN) | Late fusion eliminates orthogonal failure modes, achieving 97.88% accuracy. |
| **Edge devices & high-throughput pipelines** | ConvNeXt-Tiny | Delivers 153.2 FPS with low 6.53 ms latency per frame on consumer GPUs. |

### Common Misconceptions Debunked
1. *"Transformers are always better than CNNs."* — False. On FaceSwap boundary seam detection, ConvNeXt-Tiny matches ViT accuracy while running 1.05x faster (153.2 FPS).
2. *"ViT cannot be used on small datasets."* — False. Pretrained foundation models (e.g. DINOv3) fine-tune stably on small downstream datasets.
3. *"Larger model capacity guarantees higher accuracy."* — False. Excessive capacity on deepfake forensics risks memorizing generator-specific artifacts, hurting cross-method generalization.
4. *"Higher overall accuracy implies dominance across all methods."* — False. ViT excels on Diffusion while ConvNeXt excels on FaceSwap boundary seams.

---

## 9. Visual Evidence & Coursework Diagnostic Figures

All figures in this section originate directly from the project coursework notebook [`notebooks/coursework_deepfake.ipynb`](file:///home/bush/Desktop/deepfake-ViT/notebooks/coursework_deepfake.ipynb) and [`notebooks/data_train_test_method_analysis.ipynb`](file:///home/bush/Desktop/deepfake-ViT/notebooks/data_train_test_method_analysis.ipynb).

### Figure 1: Multi-Split Dataset Census Overview
![Multi-Split Dataset Census](experiments/results/dataset_analysis/multi_split_census_overview.png)
* **Source Notebook:** `data_train_test_method_analysis.ipynb` (Section 1.2).
* **Observation:** The project architecture manages 129.8k training images, 6k validation images, 20.8k balanced test images, and 48.1k full-suite test images across 51 training and 38 evaluation subsets.
* **Interpretation:** Confirms rigorous dataset accounting and exact 1:1 balance on test evaluation splits.
* **Limitation:** Visualizes aggregate volume and does not reflect individual image resolution differences.

### Figure 2: Generative Paradigm Proportions
![Generative Paradigm Proportions](experiments/results/dataset_analysis/generative_paradigms_pie_distribution.png)
* **Source Notebook:** `data_train_test_method_analysis.ipynb` (Section 2.2).
* **Observation:** Training data consists of 23.9% Real, 23.9% Diffusion, 20.5% GANs, 19.8% Reenactment, 11.5% FaceSwap, and 0.4% Attribute Editing. Test Balanced is exactly 50.0% Real and 50.0% Fake.
* **Interpretation:** Demonstrates balanced coverage across all 6 generative paradigms during benchmark evaluation.
* **Limitation:** Proportions reflect dataset curation and do not represent the in-the-wild frequency of deepfake encounters.

### Figure 3: Certified 4-Tier Zero-Leakage Audit
![Zero-Leakage Audit](experiments/results/dataset_analysis/leakage_purge_by_method.png)
* **Source Notebook:** `data_train_test_method_analysis.ipynb` (Section 5.2).
* **Observation:** 4,085 duplicate MD5 hash collisions were detected and permanently purged from the candidate evaluation pool (including 100% of candidate frames from `CollabDiff`, `whichfaceisreal`, and `starganv2`).
* **Interpretation:** Guarantees 0.00% data leakage between training and evaluation splits, ensuring unbiased generalization metrics.
* **Limitation:** MD5 hash matching catches exact byte-level duplicates but does not detect severe geometric crop variations.

### Figure 4: 2D Cross-Split Method Density Heatmap
![Cross-Split Method Heatmap](experiments/results/dataset_analysis/cross_split_method_heatmap.png)
* **Source Notebook:** `data_train_test_method_analysis.ipynb` (Section 3.2).
* **Observation:** 2D normalized distribution matrix across 36 monitored synthesis algorithms shows uniform distribution across evaluation splits.
* **Interpretation:** Confirms that no single generator dominates the test benchmark, preventing metric skew.
* **Limitation:** Displays percentage density rather than absolute sample counts.

### Figure 5: Dual-Split Side-by-Side Confusion Matrices
![Dual-Split Confusion Matrices](experiments/results/cnn_benchmark/chart2_dual_split_confusion_matrices.png)
* **Source Notebook:** `coursework_deepfake.ipynb` (Section 3.3).
* **Observation:** On Test Balanced (20,846 images), ViT-S/16 correctly classifies 10,219 real images (98.04% specificity) with 287 false negatives. ConvNeXt-Tiny achieves 10,159 real images (97.47% specificity) with 336 false negatives. The Joint Ensemble achieves 10,232 correct real images (98.17% specificity) and drops false negatives to 250 (97.60% recall).
* **Interpretation:** ViT exhibits higher specificity on authentic faces, whereas ConvNeXt has slightly higher false alarms. The Joint Ensemble reduces false alarms and false negatives synergistically.
* **Limitation:** Evaluates hard binary predictions at threshold $\tau = 0.5$ and does not display continuous prediction probability margins.

### Figure 6: Statistical Performance Triad (ROC, PR, and ECE Calibration)
![Statistical Performance Triad](experiments/results/cnn_benchmark/chart3_roc_pr_calibration_triad.png)
* **Source Notebook:** `coursework_deepfake.ipynb` (Section 4.2).
* **Observation:** ROC-AUC reaches 99.74% for the Ensemble (ViT: 99.68%, ConvNeXt: 99.54%). Average Precision on the PR curve is 99.71%. The Expected Calibration Error (ECE) reliability diagram closely tracks the diagonal.
* **Interpretation:** Both architectures output well-calibrated posterior probabilities, justifying late probability fusion ($P = 0.65\cdot\text{ViT} + 0.35\cdot\text{CNN}$).
* **Limitation:** Performance on zero-leakage splits may experience domain degradation on heavily compressed social media videos.

### Figure 7: Prediction Probability Density Distributions (KDE Real vs. Fake)
![Probability Density Distributions](experiments/results/cnn_benchmark/chart4_score_probability_distributions.png)
* **Source Notebook:** `coursework_deepfake.ipynb` (Section 4.3).
* **Observation:** Kernel Density Estimation (KDE) shows sharp bimodal separation with minimal probability mass in the ambiguous region $[0.3, 0.7]$.
* **Interpretation:** Demonstrates confident model predictions on both authentic and manipulated samples.
* **Limitation:** Displays 1D density projections and does not expose multi-dimensional feature cluster boundaries.

### Figure 8: 38-Method Complete Accuracy Ranking
![38-Method Complete Accuracy Ranking](experiments/results/cnn_benchmark/chart6_per_method_accuracy_ranking.png)
* **Source Notebook:** `coursework_deepfake.ipynb` (Section 4.4).
* **Observation:** All 38 evaluated manipulation algorithms and real face domains exceed the $\ge 95.0\%$ rubric target. GAN-based methods achieve >99.0% detection, while complex FaceSwap in dark scenes achieves ~96.2%–96.8%.
* **Interpretation:** Unconditional GAN artifacts are universally detected; subtle boundary-blended FaceSwap represents the hardest cases.
* **Limitation:** Performance rankings reflect specific held-out test splits and may shift on newer generator architectures.

### Figure 9: Inductive Bias Cross-Model Scatter Correlation
![Inductive Bias Scatter Correlation](experiments/results/cnn_benchmark/chart7_inductive_bias_scatter_correlation.png)
* **Source Notebook:** `coursework_deepfake.ipynb` (Section 4.5).
* **Observation:** Per-method accuracy of ViT-S/16 against ConvNeXt-Tiny yields a strong Pearson linear correlation coefficient $r = 0.94$. Points lie symmetrically along the parity diagonal $y = x$.
* **Interpretation:** High correlation confirms cross-method stability. Points above the diagonal reflect ViT's advantage on diffusion models, while points below reflect ConvNeXt's localized boundary sensitivity.
* **Limitation:** Scatter correlation represents aggregate method-level metrics and does not show sample-level image divergence.

### Figure 10: Decision Threshold Sensitivity & Youden's J Optimization
![Decision Threshold Sensitivity](experiments/results/cnn_benchmark/chart8_threshold_sensitivity_sweep.png)
* **Source Notebook:** `coursework_deepfake.ipynb` (Section 4.6).
* **Observation:** Sweeping decision threshold $\tau \in [0.0, 1.0]$ reveals an optimal Youden's J statistic at $\tau^* = 0.485$. Accuracy remains stable ($\ge 97.0\%$) across the broad interval $\tau \in [0.25, 0.75]$.
* **Interpretation:** Proves high robustness against threshold perturbation, confirming that the default threshold $\tau = 0.50$ is near-optimal.
* **Limitation:** Optimal threshold is computed on in-distribution test splits and may require re-calibration under heavy class imbalance.

---

## 10. Final Key Takeaways

* **CNN Strength:** Meta DINOv3 ConvNeXt-Tiny excels at capturing localized boundary step-discontinuities and high-frequency noise residuals with superior throughput (**153.2 FPS**, $6.53$ ms latency).
* **ViT Strength:** Meta DINOv3 ViT-Small/16 excels at whole-image diffusion and unconditional GAN detection by modeling long-range cross-facial illumination coherence and iris reflection symmetry (**97.64% Standalone Accuracy**, $99.68\%$ ROC-AUC).
* **ViT Weakness:** Lacks spatial inductive bias and sub-patch continuous boundary sensitivity, making it less optimal when training from scratch on small datasets.
* **CNN Weakness:** Constrained local receptive fields saturate earlier on massive dataset regimes and struggle with global semantic inconsistencies.
* **Dataset Scaling Law:** CNNs dominate low-data regimes (<10k samples) due to hardcoded locality priors; ViTs scale superiorly on large data corpuses (>100k samples) when initialized from foundation pretraining.
* **Role of Foundation Pretraining:** Pretraining on Meta LVD-142M equips both backbones with domain-invariant visual representations, eliminating the traditional convergence penalty of Transformers.
* **Project Architecture Selection:** The Joint Weighted Ensemble ($0.65 \cdot P_{\text{ViT}} + 0.35 \cdot P_{\text{CNN}}$) is selected as the production model, achieving **97.88% Test Accuracy** and **99.74% ROC-AUC** across 38 manipulation algorithms on certified zero-leakage benchmarks.
