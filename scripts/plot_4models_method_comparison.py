#!/usr/bin/env python3
"""
Generate separate individual charts for the 4-model per-method benchmark.
Splits the 5 subplots into 5 separate standalone high-quality charts:
1. chart1_vit_probe_per_method.png
2. chart2_convnext_probe_per_method.png
3. chart3_vit_plus_finetune_per_method.png
4. chart4_convnext_finetuned_per_method.png
5. chart5_all4_models_scatter_comparison.png
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# 38 methods in exact order (top to bottom)
METHODS = [
    "StyleGAN2",
    "StyleGAN3",
    "hyperreenact",
    "RDDM",
    "facevid2vid",
    "ddim",
    "fomm",
    "sd2.1",
    "e4e",
    "VQGAN",
    "one_shot_free",
    "pixart",
    "prender",
    "StyleGANXL",
    "MRAA",
    "stargan",
    "DiT",
    "SiT",
    "uniface",
    "e4s",
    "danet",
    "tpsm",
    "mcnet",
    "fsgan*",
    "mobileswap*",
    "styleclip",
    "deepfake_faceswap*",
    "deepfacelab",
    "blendface",
    "faceswap*",
    "facedancer*",
    "simswap",
    "inswap*",
    "sadtalker*",
    "Midjourney",
    "lia",
    "wav2lip*",
    "heygen"
]

WEAK_METHODS = {
    "fsgan*", "mobileswap*", "deepfake_faceswap*", "faceswap*",
    "facedancer*", "inswap*", "sadtalker*", "wav2lip*"
}

SCORES = {
    # Method: [ViT probe, ConvNeXt probe, ViT-Plus finetune A1, ConvNeXt finetuned]
    "StyleGAN2":          [1.000, 1.000, 1.000, 1.000],
    "StyleGAN3":          [1.000, 1.000, 1.000, 1.000],
    "hyperreenact":       [1.000, 1.000, 1.000, 1.000],
    "RDDM":               [0.985, 1.000, 1.000, 1.000],
    "facevid2vid":        [0.990, 0.983, 1.000, 1.000],
    "ddim":               [0.987, 0.986, 0.993, 1.000],
    "fomm":               [0.991, 0.986, 0.991, 1.000],
    "sd2.1":              [0.983, 0.986, 0.997, 1.000],
    "e4e":                [0.983, 0.976, 1.000, 1.000],
    "VQGAN":              [0.993, 0.969, 0.991, 1.000],
    "one_shot_free":      [1.000, 0.966, 0.997, 1.000],
    "pixart":             [0.976, 0.966, 0.987, 0.993],
    "prender":            [0.980, 0.945, 0.993, 1.000],
    "StyleGANXL":         [0.991, 0.929, 0.993, 0.997],
    "MRAA":               [0.939, 0.927, 0.984, 0.912],
    "stargan":            [0.970, 0.796, 0.984, 0.999],
    "DiT":                [0.937, 0.793, 0.997, 0.997],
    "SiT":                [0.921, 0.806, 0.997, 0.993],
    "uniface":            [0.943, 0.756, 1.000, 1.000],
    "e4s":                [0.884, 0.816, 0.991, 0.993],
    "danet":              [0.824, 0.833, 1.000, 0.997],
    "tpsm":               [0.827, 0.789, 1.000, 0.997],
    "mcnet":              [0.820, 0.789, 0.997, 0.997],
    "fsgan*":             [0.894, 0.826, 0.970, 0.903],
    "mobileswap*":        [0.880, 0.736, 0.987, 0.986],
    "styleclip":          [0.757, 0.863, 0.970, 0.999],
    "deepfake_faceswap*": [0.860, 0.802, 0.969, 0.945],
    "deepfacelab":        [0.921, 0.640, 1.000, 1.000],
    "blendface":          [0.876, 0.699, 0.984, 0.997],
    "faceswap*":          [0.779, 0.833, 0.960, 0.973],
    "facedancer*":        [0.850, 0.769, 0.974, 0.945],
    "simswap":            [0.847, 0.642, 1.000, 0.997],
    "inswap*":            [0.827, 0.683, 0.993, 0.976],
    "sadtalker*":         [0.803, 0.703, 0.984, 0.960],
    "Midjourney":         [0.455, 0.994, 1.000, 1.000],
    "lia":                [0.816, 0.666, 0.991, 0.963],
    "wav2lip*":           [0.827, 0.609, 0.984, 0.993],
    "heygen":             [0.455, 0.545, 1.000, 1.000],
}

# Colors
C_VIT_PROBE = "#ba8b4c"    # Golden brown / mustard tan
C_CNN_PROBE = "#cb6899"    # Dusty rose / pinkish magenta
C_VIT_FINE  = "#4d74b2"    # Steel blue
C_CNN_FINE  = "#2ea267"    # Jade green

def plot_single_bar_chart(vals, color, title, subtitle, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    methods_rev = METHODS[::-1]
    y_pos_rev = np.arange(len(methods_rev))

    fig, ax = plt.subplots(figsize=(8.5, 11), dpi=200)
    ax.barh(y_pos_rev, vals, height=0.72, color=color, alpha=0.95)
    ax.set_xlim(0.0, 1.02)
    ax.set_ylim(-0.6, len(METHODS) - 0.4)
    ax.set_xticks(np.arange(0.0, 1.1, 0.2))
    ax.set_xticklabels([f"{x:.1f}" for x in np.arange(0.0, 1.1, 0.2)], fontsize=10)
    ax.set_xlabel("detection rate (fake, 300 ảnh/method)", fontsize=10.5, labelpad=7)
    
    # Title & Subtitle
    ax.set_title(title, fontsize=12, fontweight="bold", pad=14)
    if subtitle:
        fig.text(0.5, 0.96, subtitle, ha="center", va="top", fontsize=9.5, color="#444444")
    
    ax.set_yticks(y_pos_rev)
    ax.set_yticklabels(methods_rev, fontsize=9)
    for tick_label in ax.get_yticklabels():
        if tick_label.get_text() in WEAK_METHODS:
            tick_label.set_fontweight("bold")

    ax.grid(axis="x", alpha=0.4, color="#d8d8d8", linestyle="-", linewidth=0.7)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color("#222222")
        spine.set_linewidth(0.8)

    fig.tight_layout(rect=[0.02, 0.02, 0.98, 0.94 if subtitle else 0.97])
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"✅ Saved: {output_path}")

def plot_scatter_comparison_chart(output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    methods_rev = METHODS[::-1]
    y_pos_rev = np.arange(len(methods_rev))

    v_probe_vals = [SCORES[m][0] for m in methods_rev]
    c_probe_vals = [SCORES[m][1] for m in methods_rev]
    v_fine_vals  = [SCORES[m][2] for m in methods_rev]
    c_fine_vals  = [SCORES[m][3] for m in methods_rev]

    fig, ax = plt.subplots(figsize=(10.5, 11.5), dpi=200)
    ax.set_title("So sánh cả 4 — điểm mỗi method; nét đứt = mean det của từng model",
                 fontsize=12, fontweight="bold", loc="left", pad=12)
    ax.set_xlim(0.0, 1.02)
    ax.set_ylim(-0.6, len(METHODS) - 0.4)
    ax.set_xticks(np.arange(0.0, 1.1, 0.2))
    ax.set_xticklabels([f"{x:.1f}" for x in np.arange(0.0, 1.1, 0.2)], fontsize=10)
    ax.set_xlabel("detection rate (fake, 300 ảnh/method)", fontsize=10.5, labelpad=7)
    ax.set_yticks(y_pos_rev)
    ax.set_yticklabels(methods_rev, fontsize=9)
    for tick_label in ax.get_yticklabels():
        if tick_label.get_text() in WEAK_METHODS:
            tick_label.set_fontweight("bold")

    ax.grid(axis="x", alpha=0.4, color="#d8d8d8", linestyle="-", linewidth=0.7)
    ax.set_axisbelow(True)

    # Vertical dashed lines for mean det
    ax.axvline(x=0.887, color=C_VIT_PROBE, linestyle="--", linewidth=1.1, alpha=0.9)
    ax.axvline(x=0.843, color=C_CNN_PROBE, linestyle="--", linewidth=1.1, alpha=0.9)
    ax.axvline(x=0.991, color=C_VIT_FINE,  linestyle="--", linewidth=1.1, alpha=0.9)
    ax.axvline(x=0.987, color=C_CNN_FINE,  linestyle="--", linewidth=1.1, alpha=0.9)

    # Scatter dots for each model
    ax.scatter(v_probe_vals, y_pos_rev, color=C_VIT_PROBE, s=36, alpha=0.9, label="ViT-S/16+ pretrained (probe)", zorder=3)
    ax.scatter(c_probe_vals, y_pos_rev, color=C_CNN_PROBE, s=36, alpha=0.9, label="ConvNeXt pretrained (probe)", zorder=3)
    ax.scatter(v_fine_vals,  y_pos_rev, color=C_VIT_FINE,  s=36, alpha=0.9, label="ViT-Plus finetune A1", zorder=3)
    ax.scatter(c_fine_vals,  y_pos_rev, color=C_CNN_FINE,  s=36, alpha=0.9, label="ConvNeXt finetuned", zorder=3)

    # Legend at bottom
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.06), ncol=4, frameon=True, fontsize=9.5,
              handletextpad=0.4, columnspacing=1.5)

    for spine in ax.spines.values():
        spine.set_color("#222222")
        spine.set_linewidth(0.8)

    fig.tight_layout(rect=[0.02, 0.05, 0.98, 0.97])
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"✅ Saved: {output_path}")

def generate_all_separate_charts(out_dir="experiments/plots"):
    os.makedirs(out_dir, exist_ok=True)
    methods_rev = METHODS[::-1]
    subtitle = "* = 8 method yếu (mục tiêu sampler A1) — cùng thứ tự method ở mọi ô (mạnh nhất trên cùng)"

    # 1. ViT Probe
    v_probe_vals = [SCORES[m][0] for m in methods_rev]
    p1 = os.path.join(out_dir, "chart1_vit_probe_per_method.png")
    plot_single_bar_chart(v_probe_vals, C_VIT_PROBE,
                          "ViT-S/16+ pretrained (probe)   ·   acc 89.5%   ·   mean det 88.7%",
                          subtitle, p1)

    # 2. ConvNeXt Probe
    c_probe_vals = [SCORES[m][1] for m in methods_rev]
    p2 = os.path.join(out_dir, "chart2_convnext_probe_per_method.png")
    plot_single_bar_chart(c_probe_vals, C_CNN_PROBE,
                          "ConvNeXt pretrained (probe)   ·   acc 87.8%   ·   mean det 84.3%",
                          subtitle, p2)

    # 3. ViT-Plus Finetune A1
    v_fine_vals = [SCORES[m][2] for m in methods_rev]
    p3 = os.path.join(out_dir, "chart3_vit_plus_finetune_per_method.png")
    plot_single_bar_chart(v_fine_vals, C_VIT_FINE,
                          "ViT-Plus finetune A1   ·   acc 98.5%   ·   mean det 99.1%",
                          None, p3)

    # 4. ConvNeXt Finetuned
    c_fine_vals = [SCORES[m][3] for m in methods_rev]
    p4 = os.path.join(out_dir, "chart4_convnext_finetuned_per_method.png")
    plot_single_bar_chart(c_fine_vals, C_CNN_FINE,
                          "ConvNeXt finetuned   ·   acc 99.2%   ·   mean det 98.7%",
                          None, p4)

    # 5. Scatter comparison
    p5 = os.path.join(out_dir, "chart5_all4_models_scatter_comparison.png")
    plot_scatter_comparison_chart(p5)

    return [p1, p2, p3, p4, p5]

if __name__ == "__main__":
    generate_all_separate_charts()
