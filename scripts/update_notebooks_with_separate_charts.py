#!/usr/bin/env python3
"""
Update notebooks and report to embed the 5 separate charts instead of 1 combined 5-in-1 image.
"""

import json, os, base64
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOKS_DIR = REPO_ROOT / "notebooks"
PLOTS_DIR = REPO_ROOT / "experiments/plots"
PLOTS_EXTRACTED = REPO_ROOT / "experiments/plots/notebook_extracted"
PLOTS_EXTRACTED.mkdir(parents=True, exist_ok=True)

from plot_4models_method_comparison import generate_all_separate_charts

chart_files = generate_all_separate_charts(str(PLOTS_DIR))

# Encode each chart to base64
b64_charts = {}
for p in chart_files:
    fname = os.path.basename(p)
    with open(p, "rb") as f:
        b64_charts[fname] = base64.b64encode(f.read()).decode("utf-8")
    # Also save to notebook_extracted
    with open(PLOTS_EXTRACTED / fname, "wb") as f:
        f.write(base64.b64decode(b64_charts[fname]))

# Create individual cells for Section 4.6
cells_46 = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 4.6 Multi-Model Per-Method Diagnostic Benchmark (5 Phân tích Tách biệt)\n",
            "\n",
            "Phân rã chi tiết hiệu năng phát hiện trên **38 phương pháp giả mạo** (300 ảnh/method) qua 5 biểu đồ độc lập, đối sánh năng lực biểu diễn sơ cấp (**Linear Probe**) và sau tinh chỉnh (**Finetuning với Sampler A1** bù đắp phương pháp yếu).\n",
            "\n",
            "> **Ghi chú (*)**: 8 phương pháp yếu in đậm có dấu `*` (`fsgan*`, `mobileswap*`, `deepfake_faceswap*`, `faceswap*`, `facedancer*`, `inswap*`, `sadtalker*`, `wav2lip*`)."
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### 4.6.1. ViT-S/16+ Pretrained (Linear Probe Baseline)\n",
            "**Test Accuracy: 89.5% | Mean Detection Rate: 88.7%**\n",
            "Đặc trưng: Bắt trọn vẹn các phương pháp toàn mặt / GAN (StyleGAN2/3, VQGAN, PixArt >97%), nhưng yếu ở FaceSwap cục bộ và Reenactment (`faceswap` 77.9%, `wav2lip` 82.7%, `Midjourney` 45.5%)."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": 16,
        "metadata": {},
        "outputs": [
            {
                "name": "stdout",
                "output_type": "stream",
                "text": ["📊 Chart 1: ViT-S/16+ Pretrained (Probe)\n"]
            },
            {
                "data": {
                    "image/png": b64_charts["chart1_vit_probe_per_method.png"],
                    "text/plain": ["<Figure size 1700x2200 with 1 Axes>"]
                },
                "metadata": {},
                "output_type": "display_data"
            }
        ],
        "source": [
            "from PIL import Image\n",
            "import matplotlib.pyplot as plt\n",
            "img1 = Image.open('experiments/plots/chart1_vit_probe_per_method.png')\n",
            "plt.figure(figsize=(9, 11), dpi=150)\n",
            "plt.imshow(img1)\n",
            "plt.axis('off')\n",
            "plt.show()\n"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### 4.6.2. ConvNeXt Pretrained (Linear Probe Baseline)\n",
            "**Test Accuracy: 87.8% | Mean Detection Rate: 84.3%**\n",
            "Đặc trưng: Inductive bias không gian mạnh giúp phát hiện tốt Midjourney (99.4%) và Styleclip (86.3%), nhưng suy giảm ở DeepFaceLab (64.0%), Wav2Lip (60.9%), HeyGen (54.5%)."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": 17,
        "metadata": {},
        "outputs": [
            {
                "name": "stdout",
                "output_type": "stream",
                "text": ["📊 Chart 2: ConvNeXt Pretrained (Probe)\n"]
            },
            {
                "data": {
                    "image/png": b64_charts["chart2_convnext_probe_per_method.png"],
                    "text/plain": ["<Figure size 1700x2200 with 1 Axes>"]
                },
                "metadata": {},
                "output_type": "display_data"
            }
        ],
        "source": [
            "img2 = Image.open('experiments/plots/chart2_convnext_probe_per_method.png')\n",
            "plt.figure(figsize=(9, 11), dpi=150)\n",
            "plt.imshow(img2)\n",
            "plt.axis('off')\n",
            "plt.show()\n"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### 4.6.3. ViT-Plus Finetune A1 (`plus_v3_s1_best.pt`)\n",
            "**Test Accuracy: 98.5% | Mean Detection Rate: 99.1%**\n",
            "Đặc trưng: Khắc phục triệt để toàn bộ 38 phương pháp, 100% các phương pháp đạt độ chính xác >96%, Midjourney & HeyGen tăng vọt lên 100%."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": 18,
        "metadata": {},
        "outputs": [
            {
                "name": "stdout",
                "output_type": "stream",
                "text": ["📊 Chart 3: ViT-Plus Finetune A1\n"]
            },
            {
                "data": {
                    "image/png": b64_charts["chart3_vit_plus_finetune_per_method.png"],
                    "text/plain": ["<Figure size 1700x2200 with 1 Axes>"]
                },
                "metadata": {},
                "output_type": "display_data"
            }
        ],
        "source": [
            "img3 = Image.open('experiments/plots/chart3_vit_plus_finetune_per_method.png')\n",
            "plt.figure(figsize=(9, 11), dpi=150)\n",
            "plt.imshow(img3)\n",
            "plt.axis('off')\n",
            "plt.show()\n"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### 4.6.4. ConvNeXt Finetuned (`convnext_weakfix_v3.pt`)\n",
            "**Test Accuracy: 99.2% | Mean Detection Rate: 98.7%**\n",
            "Đặc trưng: Độ tin cậy cực cao, hầu hết các phương pháp đều đạt xấp xỉ 100%, bảo vệ chống báo động giả tốt nhất."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": 19,
        "metadata": {},
        "outputs": [
            {
                "name": "stdout",
                "output_type": "stream",
                "text": ["📊 Chart 4: ConvNeXt Finetuned\n"]
            },
            {
                "data": {
                    "image/png": b64_charts["chart4_convnext_finetuned_per_method.png"],
                    "text/plain": ["<Figure size 1700x2200 with 1 Axes>"]
                },
                "metadata": {},
                "output_type": "display_data"
            }
        ],
        "source": [
            "img4 = Image.open('experiments/plots/chart4_convnext_finetuned_per_method.png')\n",
            "plt.figure(figsize=(9, 11), dpi=150)\n",
            "plt.imshow(img4)\n",
            "plt.axis('off')\n",
            "plt.show()\n"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### 4.6.5. Đồ thị Phân tán So sánh Cả 4 Mô hình (Cross-Model Method Scatter Comparison)\n",
            "Đồ thị biểu diễn phân bố điểm phát hiện của từng phương pháp trên cả 4 mô hình, kèm theo 4 đường nét đứt biểu thị Mean Detection Rate tương ứng."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": 20,
        "metadata": {},
        "outputs": [
            {
                "name": "stdout",
                "output_type": "stream",
                "text": ["📊 Chart 5: 4-Model Scatter Comparison\n"]
            },
            {
                "data": {
                    "image/png": b64_charts["chart5_all4_models_scatter_comparison.png"],
                    "text/plain": ["<Figure size 2100x2300 with 1 Axes>"]
                },
                "metadata": {},
                "output_type": "display_data"
            }
        ],
        "source": [
            "img5 = Image.open('experiments/plots/chart5_all4_models_scatter_comparison.png')\n",
            "plt.figure(figsize=(10.5, 11.5), dpi=150)\n",
            "plt.imshow(img5)\n",
            "plt.axis('off')\n",
            "plt.show()\n"
        ]
    }
]

# Update coursework notebook
nb_path = NOTEBOOKS_DIR / "coursework_deepfake_plus_v3_s1_best.ipynb"
with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

# Strip any previous 4.6 section
filtered_cells = []
skip = False
for c in nb["cells"]:
    src = "".join(c.get("source", []))
    if "## 4.6 Multi-Model" in src:
        skip = True
    elif skip and ("---" in src or "# Section 5" in src or "## 5.1" in src):
        skip = False
        filtered_cells.append(c)
    elif not skip:
        filtered_cells.append(c)

# Find insertion point right before Section 5
insert_idx = len(filtered_cells)
for idx, c in enumerate(filtered_cells):
    src = "".join(c.get("source", []))
    if "# Section 5" in src or "## 5.1" in src or "## 5.2" in src:
        insert_idx = idx
        break

# Insert separated cells
for offset, cell in enumerate(cells_46):
    filtered_cells.insert(insert_idx + offset, cell)

nb["cells"] = filtered_cells
with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)
print(f"✅ Updated {nb_path} with 5 separate chart cells")

# Update deepfake_forensics_report.ipynb
report_nb_path = NOTEBOOKS_DIR / "deepfake_forensics_report.ipynb"
with open(report_nb_path, "r", encoding="utf-8") as f:
    r_nb = json.load(f)

r_filtered = []
skip = False
for c in r_nb["cells"]:
    src = "".join(c.get("source", []))
    if "## 4.6 Multi-Model" in src:
        skip = True
    elif skip and ("---" in src or "# Section 5" in src or "## 5.1" in src):
        skip = False
        r_filtered.append(c)
    elif not skip:
        r_filtered.append(c)

r_insert_idx = len(r_filtered)
for idx, c in enumerate(r_filtered):
    src = "".join(c.get("source", []))
    if "# Section 5" in src or "## 5.1" in src or "## 5.2" in src:
        r_insert_idx = idx
        break

for offset, cell in enumerate(cells_46):
    r_filtered.insert(r_insert_idx + offset, cell)

r_nb["cells"] = r_filtered
with open(report_nb_path, "w", encoding="utf-8") as f:
    json.dump(r_nb, f, indent=1)
print(f"✅ Updated {report_nb_path} with 5 separate chart cells")

# Update DEEPFAKE_FORENSICS_REPORT.md
md_report_path = REPO_ROOT / "DEEPFAKE_FORENSICS_REPORT.md"
with open(md_report_path, "r", encoding="utf-8") as f:
    content = f.read()

new_markdown_section = """
### 4.2.1. Phân Tích Chuyên Sâu 4 Mô Hình trên 38 Phương Pháp Tạo Ảnh Giả (5 Biểu Đồ Độc Lập)

Khảo sát đối sánh độc lập từng kiến trúc trên toàn bộ **38 phương pháp Deepfake** (300 ảnh/method):

#### 1. DINOv3 ViT-S/16+ Pretrained (Linear Probe)
- **Độ chính xác:** $89.5\%$ | **Mean Detection Rate:** $88.7\%$
- Nhận diện hoàn hảo các mô hình toàn cảnh và GAN, nhưng bỏ sót các can thiệp ghép mặt cục bộ và talking-head thô sơ.

![ViT Probe Per-Method](./experiments/plots/chart1_vit_probe_per_method.png)
*Hình 4.2b: Tỷ lệ phát hiện của ViT-S/16+ Pretrained (Linear Probe) trên 38 phương pháp tạo ảnh giả.*

#### 2. DINOv3 ConvNeXt-Tiny Pretrained (Linear Probe)
- **Độ chính xác:** $87.8\%$ | **Mean Detection Rate:** $84.3\%$
- Bắt tốt Midjourney ($99.4\%$) và Styleclip ($86.3\%$) nhờ inductive bias không gian, nhưng suy giảm ở các dạng video reenactment khẩu hình.

![ConvNeXt Probe Per-Method](./experiments/plots/chart2_convnext_probe_per_method.png)
*Hình 4.2c: Tỷ lệ phát hiện của ConvNeXt Pretrained (Linear Probe) trên 38 phương pháp tạo ảnh giả.*

#### 3. DINOv3 ViT-Plus Finetune A1 (`plus_v3_s1_best.pt`)
- **Độ chính xác:** $98.5\%$ | **Mean Detection Rate:** $99.1\%$
- Sau tinh chỉnh tập trung với Sampler A1, toàn bộ 8 phương pháp yếu đều vượt ngưỡng an toàn ($>96\%-99\%$).

![ViT-Plus Finetune A1 Per-Method](./experiments/plots/chart3_vit_plus_finetune_per_method.png)
*Hình 4.2d: Tỷ lệ phát hiện vượt trội của ViT-Plus Finetune A1 sau khi khắc phục các phương pháp yếu.*

#### 4. DINOv3 ConvNeXt-Tiny Finetuned (`convnext_weakfix_v3.pt`)
- **Độ chính xác:** $99.2\%$ | **Mean Detection Rate:** $98.7\%$
- Đạt độ ổn định đồng đều tuyệt đối trên mọi chủng loại dữ liệu, tỷ lệ báo động giả thấp nhất.

![ConvNeXt Finetuned Per-Method](./experiments/plots/chart4_convnext_finetuned_per_method.png)
*Hình 4.2e: Tỷ lệ phát hiện của ConvNeXt Finetuned trên 38 phương pháp tạo ảnh giả.*

#### 5. Đồ Thị Phân Tán Đối Sánh Tổng Hợp 4 Mô Hình
- Thể hiện sự dịch chuyển rõ rệt của cụm điểm từ dải phân tán rộng ($45\%-100\%$) ở giai đoạn Pretrained co cụm chặt chẽ về sát mốc $100\%$ sau Finetuning.

![4-Model Scatter Comparison](./experiments/plots/chart5_all4_models_scatter_comparison.png)
*Hình 4.2f: Đồ thị phân tán đối sánh cả 4 mô hình trên 38 phương pháp (đường nét đứt thể hiện mean det).*
"""

# Replace old 4.2.1 section if exists, or insert before 4.3
if "### 4.2.1." in content:
    # replace existing 4.2.1
    parts = content.split("### 4.2.1.")
    before = parts[0]
    after = parts[1].split("## 4.3. Đánh giá Mức độ Cải thiện trên Các Phương pháp Yếu")[1]
    content = before + new_markdown_section + "\n## 4.3. Đánh giá Mức độ Cải thiện trên Các Phương pháp Yếu" + after
else:
    content = content.replace("## 4.3. Đánh giá Mức độ Cải thiện trên Các Phương pháp Yếu", new_markdown_section + "\n## 4.3. Đánh giá Mức độ Cải thiện trên Các Phương pháp Yếu")

with open(md_report_path, "w", encoding="utf-8") as f:
    f.write(content)
print(f"✅ Updated {md_report_path} with separate figures")

print("✨ Successfully separated all 5 charts across notebooks and markdown report!")
