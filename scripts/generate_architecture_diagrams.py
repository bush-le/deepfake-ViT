"""
Generate crisp, publication-grade architectural PNG diagrams corresponding to all Mermaid diagrams.
Saves to: experiments/results/diagrams/
1. diagram1_forensics_pipeline.png
2. diagram2_artifact_scale_decision.png
3. diagram3_data_regime_scaling.png
"""

from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches

REPO_ROOT = Path(r"C:\document\Study documents\deepfake-ViT")
OUT_DIR = REPO_ROOT / "experiments" / "results" / "diagrams"
OUT_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#bdc3c7'
plt.rcParams['axes.linewidth'] = 0.8

PRIMARY = "#1b3a4b"
SECONDARY = "#2980b9"
ACCENT = "#27ae60"
LIGHT_BG = "#f8f9fa"
DARK_TEXT = "#2c3e50"
BORDER = "#bdc3c7"

# -------------------------------------------------------------------------
# DIAGRAM 1: End-to-End Forensics Pipeline
# -------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 4.8), dpi=300)
ax.set_xlim(0, 100)
ax.set_ylim(0, 50)
ax.axis('off')

# Title
ax.text(50, 47.5, "End-to-End Deepfake Forensics Pipeline Architecture", 
        fontsize=12, fontweight='bold', ha='center', color=PRIMARY)

# Box 1: Input
b1 = patches.FancyBboxPatch((3, 16), 18, 18, boxstyle="round,pad=1", 
                            fc=LIGHT_BG, ec=SECONDARY, lw=1.5)
ax.add_patch(b1)
ax.text(12, 28, "Input Facial Crop", fontsize=9, fontweight='bold', ha='center', color=PRIMARY)
ax.text(12, 24, "256x256x3 RGB", fontsize=8, ha='center', color=DARK_TEXT)
ax.text(12, 20, "ImageNet Normalization", fontsize=7.5, ha='center', color="#7f8c8d")

# Arrows from Input to Backbones
ax.annotate("", xy=(28, 33), xytext=(21, 28),
            arrowprops=dict(arrowstyle="->", color=SECONDARY, lw=1.5, mutation_scale=12))
ax.annotate("", xy=(28, 17), xytext=(21, 22),
            arrowprops=dict(arrowstyle="->", color=SECONDARY, lw=1.5, mutation_scale=12))

# Box 2A: ViT Backbone
b2a = patches.FancyBboxPatch((28, 25), 26, 16, boxstyle="round,pad=1", 
                             fc="#ebf5fb", ec=SECONDARY, lw=1.5)
ax.add_patch(b2a)
ax.text(41, 37, "Meta DINOv3 ViT-Plus A1", fontsize=9, fontweight='bold', ha='center', color=SECONDARY)
ax.text(41, 33, "28.69M Params | Patch 16x16", fontsize=7.5, ha='center', color=DARK_TEXT)
ax.text(41, 29, "12 Layers | 6 Heads | Dim 384", fontsize=7.5, ha='center', color=DARK_TEXT)
ax.text(41, 26, "SwiGLU Gated MLP | SDPA", fontsize=7.5, fontweight='bold', ha='center', color=PRIMARY)

# Box 2B: ConvNeXt Backbone
b2b = patches.FancyBboxPatch((28, 7), 26, 16, boxstyle="round,pad=1", 
                             fc="#eafaf1", ec=ACCENT, lw=1.5)
ax.add_patch(b2b)
ax.text(41, 19, "Meta DINOv3 ConvNeXt-Tiny", fontsize=9, fontweight='bold', ha='center', color=ACCENT)
ax.text(41, 15, "28.12M Params | 4 Stages", fontsize=7.5, ha='center', color=DARK_TEXT)
ax.text(41, 11, "Channels: [96, 192, 384, 768]", fontsize=7.5, ha='center', color=DARK_TEXT)
ax.text(41, 8, "7x7 Depthwise Conv", fontsize=7.5, fontweight='bold', ha='center', color=PRIMARY)

# Arrows from Backbones to Heads
ax.annotate("", xy=(58, 33), xytext=(54, 33),
            arrowprops=dict(arrowstyle="->", color=SECONDARY, lw=1.5, mutation_scale=12))
ax.annotate("", xy=(58, 15), xytext=(54, 15),
            arrowprops=dict(arrowstyle="->", color=ACCENT, lw=1.5, mutation_scale=12))

# Box 3A: ViT Head
b3a = patches.FancyBboxPatch((58, 27), 16, 12, boxstyle="round,pad=1", 
                             fc=LIGHT_BG, ec=SECONDARY, lw=1.2)
ax.add_patch(b3a)
ax.text(66, 35, "2-Layer MLP Head", fontsize=8, fontweight='bold', ha='center', color=PRIMARY)
ax.text(66, 31, "LayerNorm -> GELU", fontsize=7.5, ha='center', color=DARK_TEXT)
ax.text(66, 28, "Linear(384, 2)", fontsize=7.5, ha='center', color=DARK_TEXT)

# Box 3B: ConvNeXt Head
b3b = patches.FancyBboxPatch((58, 9), 16, 12, boxstyle="round,pad=1", 
                             fc=LIGHT_BG, ec=ACCENT, lw=1.2)
ax.add_patch(b3b)
ax.text(66, 17, "2-Layer MLP Head", fontsize=8, fontweight='bold', ha='center', color=PRIMARY)
ax.text(66, 13, "LayerNorm -> GELU", fontsize=7.5, ha='center', color=DARK_TEXT)
ax.text(66, 10, "Linear(768, 2)", fontsize=7.5, ha='center', color=DARK_TEXT)

# Arrows from Heads to Ensemble
ax.annotate("", xy=(78, 27), xytext=(74, 31),
            arrowprops=dict(arrowstyle="->", color=SECONDARY, lw=1.5, mutation_scale=12))
ax.annotate("", xy=(78, 23), xytext=(74, 17),
            arrowprops=dict(arrowstyle="->", color=ACCENT, lw=1.5, mutation_scale=12))

# Box 4: Ensemble
b4 = patches.FancyBboxPatch((78, 13), 20, 24, boxstyle="round,pad=1", 
                            fc="#f4ecf7", ec="#8e44ad", lw=1.8)
ax.add_patch(b4)
ax.text(88, 33, "Joint Weighted Ensemble", fontsize=8.5, fontweight='bold', ha='center', color="#8e44ad")
ax.text(88, 29, "P = 0.65·ViT + 0.35·CNN", fontsize=7.5, fontweight='bold', ha='center', color=PRIMARY)
ax.text(88, 25, "Test Acc: 99.28%", fontsize=8, fontweight='bold', ha='center', color=ACCENT)
ax.text(88, 21.5, "ROC-AUC: 99.97%", fontsize=8, fontweight='bold', ha='center', color=SECONDARY)
ax.text(88, 18, "Fake Recall: 99.47%", fontsize=7.5, fontweight='bold', ha='center', color="#8e44ad")
ax.text(88, 14.5, "Latency: 13.36 ms", fontsize=7, ha='center', color=DARK_TEXT)

plt.tight_layout()
p1 = OUT_DIR / "diagram1_forensics_pipeline.png"
plt.savefig(p1, bbox_inches='tight', dpi=300)
plt.close()
print(f"Generated: {p1}")


# -------------------------------------------------------------------------
# DIAGRAM 2: Artifact Scale & Architecture Selection
# -------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 4), dpi=300)
ax.set_xlim(0, 100)
ax.set_ylim(0, 45)
ax.axis('off')

ax.text(50, 42, "Forensic Artifact Scale & Architecture Alignment", 
        fontsize=11.5, fontweight='bold', ha='center', color=PRIMARY)

# Root Box
r = patches.FancyBboxPatch((35, 28), 30, 10, boxstyle="round,pad=1", fc=LIGHT_BG, ec=PRIMARY, lw=1.5)
ax.add_patch(r)
ax.text(50, 34, "Deepfake Detection Task", fontsize=9, fontweight='bold', ha='center', color=PRIMARY)
ax.text(50, 30, "Identify Artifact Profile", fontsize=8, ha='center', color=DARK_TEXT)

# Left Branch: Global
ax.annotate("", xy=(22, 23), xytext=(40, 28),
            arrowprops=dict(arrowstyle="->", color=SECONDARY, lw=1.5, mutation_scale=12))
b_left = patches.FancyBboxPatch((4, 7), 36, 16, boxstyle="round,pad=1", fc="#ebf5fb", ec=SECONDARY, lw=1.5)
ax.add_patch(b_left)
ax.text(22, 19, "Global / Semantic Artifacts", fontsize=8.5, fontweight='bold', ha='center', color=SECONDARY)
ax.text(22, 16, "• Whole-Face Diffusion (DiT, PixArt, SD)", fontsize=7.5, ha='center', color=DARK_TEXT)
ax.text(22, 13, "• Unconditional GANs (StyleGAN2/3)", fontsize=7.5, ha='center', color=DARK_TEXT)
ax.text(22, 10, "• Illumination & Iris Symmetry Asymmetry", fontsize=7.5, ha='center', color=DARK_TEXT)
ax.text(22, 7.5, "=> Recommended: Meta DINOv3 ViT-Plus A1 (28.69M)", fontsize=7, fontweight='bold', ha='center', color=PRIMARY)

# Right Branch: Local
ax.annotate("", xy=(78, 23), xytext=(60, 28),
            arrowprops=dict(arrowstyle="->", color=ACCENT, lw=1.5, mutation_scale=12))
b_right = patches.FancyBboxPatch((60, 7), 36, 16, boxstyle="round,pad=1", fc="#eafaf1", ec=ACCENT, lw=1.5)
ax.add_patch(b_right)
ax.text(78, 19, "Local / Boundary Artifacts", fontsize=8.5, fontweight='bold', ha='center', color=ACCENT)
ax.text(78, 16, "• FaceSwap Blending Seams (InSwap, SimSwap)", fontsize=7.5, ha='center', color=DARK_TEXT)
ax.text(78, 13, "• High-Frequency Noise & JPEG Artifacts", fontsize=7.5, ha='center', color=DARK_TEXT)
ax.text(78, 10, "• Real-Time Video Streaming Constraints", fontsize=7.5, ha='center', color=DARK_TEXT)
ax.text(78, 7.5, "=> Recommended: Meta DINOv3 ConvNeXt-Tiny (28.12M)", fontsize=7, fontweight='bold', ha='center', color=PRIMARY)

plt.tight_layout()
p2 = OUT_DIR / "diagram2_artifact_scale_decision.png"
plt.savefig(p2, bbox_inches='tight', dpi=300)
plt.close()
print(f"Generated: {p2}")


# -------------------------------------------------------------------------
# DIAGRAM 3: Small vs. Large Data Regime Scaling Map
# -------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 4), dpi=300)
ax.set_xlim(0, 100)
ax.set_ylim(0, 45)
ax.axis('off')

ax.text(50, 42, "Dataset Scale & Pretraining Scaling Regime Map", 
        fontsize=11.5, fontweight='bold', ha='center', color=PRIMARY)

# Left Side: Small Data
b_s = patches.FancyBboxPatch((4, 4), 43, 34, boxstyle="round,pad=1", fc=LIGHT_BG, ec=PRIMARY, lw=1.5)
ax.add_patch(b_s)
ax.text(25.5, 34, "Small Data Regime (< 10k Samples)", fontsize=9, fontweight='bold', ha='center', color=PRIMARY)

b_s1 = patches.FancyBboxPatch((6, 20), 39, 11, boxstyle="round,pad=0.5", fc="#fadbd8", ec="#e74c3c", lw=1)
ax.add_patch(b_s1)
ax.text(25.5, 27, "Training From Scratch", fontsize=8, fontweight='bold', ha='center', color="#c0392b")
ax.text(25.5, 23, "• CNN: Converges stably (Inductive Bias)", fontsize=7.5, ha='center', color=DARK_TEXT)
ax.text(25.5, 20.5, "• ViT: Fails / Overfits severely", fontsize=7.5, fontweight='bold', ha='center', color="#c0392b")

b_s2 = patches.FancyBboxPatch((6, 6), 39, 11, boxstyle="round,pad=0.5", fc="#d5f5e3", ec=ACCENT, lw=1)
ax.add_patch(b_s2)
ax.text(25.5, 13, "Pretrained Foundation Model", fontsize=8, fontweight='bold', ha='center', color=ACCENT)
ax.text(25.5, 9.5, "• CNN: Effective, fast fine-tuning", fontsize=7.5, ha='center', color=DARK_TEXT)
ax.text(25.5, 7, "• ViT: Strong performance (Features Transfer)", fontsize=7.5, fontweight='bold', ha='center', color=ACCENT)

# Right Side: Large Data
b_l = patches.FancyBboxPatch((53, 4), 43, 34, boxstyle="round,pad=1", fc=LIGHT_BG, ec=PRIMARY, lw=1.5)
ax.add_patch(b_l)
ax.text(74.5, 34, "Large Data Regime (> 100k Samples)", fontsize=9, fontweight='bold', ha='center', color=PRIMARY)

b_l1 = patches.FancyBboxPatch((55, 20), 39, 11, boxstyle="round,pad=0.5", fc="#fef9e7", ec="#f39c12", lw=1)
ax.add_patch(b_l1)
ax.text(74.5, 27, "Training From Scratch", fontsize=8, fontweight='bold', ha='center', color="#d35400")
ax.text(74.5, 23, "• CNN: Competitive, but saturates early", fontsize=7.5, ha='center', color=DARK_TEXT)
ax.text(74.5, 20.5, "• ViT: Learns 2D geometry directly", fontsize=7.5, ha='center', color=DARK_TEXT)

b_l2 = patches.FancyBboxPatch((55, 6), 39, 11, boxstyle="round,pad=0.5", fc="#d4e6f1", ec=SECONDARY, lw=1)
ax.add_patch(b_l2)
ax.text(74.5, 13, "Pretrained Foundation Model", fontsize=8, fontweight='bold', ha='center', color=SECONDARY)
ax.text(74.5, 9.5, "• CNN: 99.49% Acc (Superior Real Specificity)", fontsize=7.5, ha='center', color=DARK_TEXT)
ax.text(74.5, 7, "• ViT: 98.53% Acc (Superior Diffusion Recall)", fontsize=7.5, fontweight='bold', ha='center', color=SECONDARY)

plt.tight_layout()
p3 = OUT_DIR / "diagram3_data_regime_scaling.png"
plt.savefig(p3, bbox_inches='tight', dpi=300)
plt.close()
print(f"Generated: {p3}")

# Also write to repo scripts directory
repo_script = REPO_ROOT / "scripts" / "generate_architecture_diagrams.py"
repo_script.write_text(Path(__file__).read_text(encoding="utf-8"), encoding="utf-8")
print(f"Also saved script to: {repo_script}")
