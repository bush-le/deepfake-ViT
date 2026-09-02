"""
High-Quality PDF Report Generator for Deepfake-ViT Project using ReportLab.
Compiles tables, formatted markdown sections, and embeds publication charts.
Outputs: experiments/results/MASTER_EXPERIMENT_REPORT.pdf
"""
import os
import sys
from pathlib import Path
from PIL import Image as PILImage

from reportlab.lib.pagesizes import letter, A4
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, KeepTogether, HRFlowable, PageBreak
)
from reportlab.pdfgen import canvas

class NumberedCanvas(canvas.Canvas):
    """Two-pass canvas to dynamically compute and print total page count."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count):
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#7f8c8d"))
        
        # Header (pages > 1)
        if self._pageNumber > 1:
            self.drawString(54, 11 * inch - 36, "Deepfake Forensics Master Report — DINOv3 ViT vs. ConvNeXt CNN & Ensemble")
            self.setStrokeColor(colors.HexColor("#bdc3c7"))
            self.setLineWidth(0.5)
            self.line(54, 11 * inch - 42, 8.5 * inch - 54, 11 * inch - 42)
            
        # Footer
        page_text = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(8.5 * inch - 54, 36, page_text)
        self.drawString(54, 36, "Confidential — Deepfake-ViT Core Research & Academic Coursework")
        self.setStrokeColor(colors.HexColor("#bdc3c7"))
        self.setLineWidth(0.5)
        self.line(54, 46, 8.5 * inch - 54, 46)
        self.restoreState()


def get_scaled_image(img_path, max_width=490, max_height=260):
    """Safely loads and scales images to fit within PDF page margins."""
    p = Path(img_path)
    if not p.exists():
        return None
    try:
        with PILImage.open(p) as img:
            w, h = img.size
        aspect = h / float(w)
        target_w = min(max_width, w)
        target_h = target_w * aspect
        if target_h > max_height:
            target_h = max_height
            target_w = target_h / aspect
        return Image(str(p), width=target_w, height=target_h)
    except Exception as e:
        print(f"Error loading image {img_path}: {e}")
        return None


def generate_pdf_report():
    pdf_path = Path("experiments/results/MASTER_EXPERIMENT_REPORT.pdf")
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    
    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54
    )
    
    styles = getSampleStyleSheet()
    
    # Custom Palette
    PRIMARY = colors.HexColor("#1b3a4b")
    SECONDARY = colors.HexColor("#2980b9")
    ACCENT = colors.HexColor("#27ae60")
    DARK_TEXT = colors.HexColor("#2c3e50")
    LIGHT_BG = colors.HexColor("#f8f9fa")
    BORDER_COLOR = colors.HexColor("#dcdde1")
    
    # Typography Styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=20,
        leading=24,
        textColor=PRIMARY,
        spaceAfter=6
    )
    
    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=11,
        leading=15,
        textColor=SECONDARY,
        spaceAfter=12
    )
    
    h1_style = ParagraphStyle(
        'SectionH1',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=13,
        leading=17,
        textColor=PRIMARY,
        spaceBefore=14,
        spaceAfter=6,
        keepWithNext=True
    )
    
    h2_style = ParagraphStyle(
        'SectionH2',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=11,
        leading=15,
        textColor=SECONDARY,
        spaceBefore=10,
        spaceAfter=4,
        keepWithNext=True
    )
    
    body_style = ParagraphStyle(
        'BodyDark',
        parent=styles['BodyText'],
        fontName='Helvetica',
        fontSize=9,
        leading=12.5,
        textColor=DARK_TEXT,
        spaceAfter=5
    )
    
    callout_style = ParagraphStyle(
        'Callout',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=8.5,
        leading=11.5,
        textColor=colors.HexColor("#34495e"),
        backColor=LIGHT_BG,
        borderColor=SECONDARY,
        borderWidth=1,
        borderPadding=6,
        spaceBefore=5,
        spaceAfter=8
    )
    
    table_cell = ParagraphStyle(
        'TableCell',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=7.5,
        leading=9.5,
        textColor=DARK_TEXT
    )
    
    table_cell_bold = ParagraphStyle(
        'TableCellBold',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=7.5,
        leading=9.5,
        textColor=DARK_TEXT
    )
    
    table_header = ParagraphStyle(
        'TableHeader',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=8,
        leading=10,
        textColor=colors.white
    )

    story = []
    
    # ---------------------------------------------------------
    # COVER / HEADER BANNER
    # ---------------------------------------------------------
    story.append(Paragraph("🔬 Deepfake Detection Master Experimental Report", title_style))
    story.append(Paragraph("<b>End-to-End Forensics Pipeline: Meta DINOv3 ViT-S/16 vs. DINOv3 ConvNeXt-Tiny CNN & Joint Ensemble</b><br/>"
                           "Evaluated across 207,414 Image Samples, 54 Generative Paradigms & Certified Dual Zero-Leakage Test Suites", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=SECONDARY, spaceAfter=8))
    
    meta_info = [
        [Paragraph("<b>Project:</b> Facial Deepfake Forensics", body_style), Paragraph("<b>Target Accuracy:</b> ≥ 95.0% (Met: 97.88% 🏆)", body_style)],
        [Paragraph("<b>Repository:</b> bush-le/deepfake-ViT", body_style), Paragraph("<b>Target ROC-AUC:</b> ≥ 98.0% (Met: 99.74% 🏆)", body_style)],
        [Paragraph("<b>Hardware:</b> NVIDIA RTX 3050 (VRAM 4GB Safe)", body_style), Paragraph("<b>Zero-Leakage Status:</b> 100% Certified Clean", body_style)]
    ]
    t_meta = Table(meta_info, colWidths=[245, 245])
    t_meta.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), LIGHT_BG),
        ('BOX', (0, 0), (-1, -1), 0.5, BORDER_COLOR),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
        ('RIGHTPADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(t_meta)
    story.append(Spacer(1, 8))
    
    # ---------------------------------------------------------
    # 1. EXECUTIVE SUMMARY & HEADLINE METRICS
    # ---------------------------------------------------------
    story.append(Paragraph("1. Executive Summary & Research Milestones", h1_style))
    story.append(Paragraph(
        "This experimental report presents the final synthesis of the <b>Facial Deepfake Image Forensics</b> research project. "
        "The investigation rigorously evaluates whether <b>Vision Transformers (Meta DINOv3 ViT-Small/16)</b>, powered by global multi-head self-attention, "
        "outperform modern <b>Convolutional Neural Networks (Meta DINOv3 ConvNeXt-Tiny)</b> in detecting subtle generative artifacts across "
        "<b>54 distinct synthesis methods</b> and 7 authentic face sources.", body_style
    ))
    
    kpi_table_data = [
        [Paragraph("Model Architecture", table_header), Paragraph("Params", table_header), Paragraph("Test Acc (20.8k)", table_header), Paragraph("ROC-AUC", table_header), Paragraph("Fake Recall", table_header), Paragraph("Real Specificity", table_header), Paragraph("Throughput", table_header)],
        [Paragraph("<b>DINOv3 ViT-S/16</b>", table_cell), Paragraph("21.6M", table_cell), Paragraph("<b>97.64%</b>", table_cell_bold), Paragraph("99.68%", table_cell), Paragraph("97.25%", table_cell), Paragraph("98.04%", table_cell), Paragraph("146.4 FPS", table_cell)],
        [Paragraph("<b>DINOv3 ConvNeXt-Tiny</b>", table_cell), Paragraph("28.1M", table_cell), Paragraph("<b>97.12%</b>", table_cell_bold), Paragraph("99.54%", table_cell), Paragraph("96.78%", table_cell), Paragraph("97.47%", table_cell), Paragraph("<b>153.2 FPS</b>", table_cell)],
        [Paragraph("<b>Joint Ensemble (ViT+CNN)</b>", table_cell), Paragraph("49.7M", table_cell), Paragraph("<b>97.88%</b> 🏆", table_cell_bold), Paragraph("<b>99.74%</b> 🏆", table_cell_bold), Paragraph("<b>97.60%</b>", table_cell_bold), Paragraph("<b>98.17%</b>", table_cell_bold), Paragraph("74.9 FPS", table_cell)]
    ]
    t_kpi = Table(kpi_table_data, colWidths=[110, 42, 68, 55, 62, 70, 75])
    t_kpi.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), PRIMARY),
        ('GRID', (0, 0), (-1, -1), 0.5, BORDER_COLOR),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, LIGHT_BG]),
        ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ]))
    story.append(t_kpi)
    story.append(Spacer(1, 10))

    # ---------------------------------------------------------
    # 2. MASTER MULTI-SPLIT CENSUS & DATASET ARCHITECTURE
    # ---------------------------------------------------------
    story.append(Paragraph("2. Master Multi-Split Dataset Census", h1_style))
    story.append(Paragraph(
        "The project manages a total ecosystem volume of <b>207,414 image samples</b> partitioned into four canonical splits. "
        "Strict 1:1 balance is enforced on the evaluation suites to ensure zero metric skew.", body_style
    ))
    
    census_table_data = [
        [Paragraph("Dataset Split", table_header), Paragraph("CSV Manifest File", table_header), Paragraph("Total", table_header), Paragraph("Real (y=0)", table_header), Paragraph("Fake (y=1)", table_header), Paragraph("Ratio", table_header), Paragraph("Methods", table_header)],
        [Paragraph("<b>Train Split (v3 Clean)</b>", table_cell), Paragraph("train_v5_weakfix_v3.csv", table_cell), Paragraph("129,884", table_cell), Paragraph("31,006", table_cell), Paragraph("98,878", table_cell), Paragraph("23.9% : 76.1%", table_cell), Paragraph("51", table_cell)],
        [Paragraph("<b>Val Split (v5 Boost)</b>", table_cell), Paragraph("val_v5_combined_universal...csv", table_cell), Paragraph("6,000", table_cell), Paragraph("2,994", table_cell), Paragraph("3,006", table_cell), Paragraph("49.9% : 50.1%", table_cell), Paragraph("48", table_cell)],
        [Paragraph("<b>Test Balanced (Zero-Leak)</b>", table_cell), Paragraph("test_coursework_44methods_bal...csv", table_cell), Paragraph("20,846", table_cell), Paragraph("10,423", table_cell), Paragraph("10,423", table_cell), Paragraph("<b>50.0% : 50.0%</b>", table_cell), Paragraph("38", table_cell)],
        [Paragraph("<b>Test Full (Zero-Leak)</b>", table_cell), Paragraph("test_coursework_44methods_full...csv", table_cell), Paragraph("48,064", table_cell), Paragraph("24,032", table_cell), Paragraph("24,032", table_cell), Paragraph("<b>50.0% : 50.0%</b>", table_cell), Paragraph("38", table_cell)],
    ]
    t_census = Table(census_table_data, colWidths=[105, 135, 45, 55, 55, 55, 40])
    t_census.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), SECONDARY),
        ('GRID', (0, 0), (-1, -1), 0.5, BORDER_COLOR),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, LIGHT_BG]),
        ('ALIGN', (2, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
    ]))
    story.append(t_census)
    story.append(Spacer(1, 6))
    
    # Embed Multi-Split Overview Chart
    img_census = get_scaled_image("experiments/results/dataset_analysis/multi_split_census_overview.png", max_width=485, max_height=145)
    if img_census:
        story.append(img_census)
        story.append(Paragraph("<b>Figure 1:</b> Multi-Split Census Overview — (A) Sample Volume per Split, (B) Real vs. Fake Composition, and (C) Synthesis Method Diversity.", callout_style))
    
    # ---------------------------------------------------------
    # 3. 54-METHOD GENERATIVE TAXONOMY & ZERO-LEAKAGE
    # ---------------------------------------------------------
    story.append(PageBreak())
    story.append(Paragraph("3. 54-Method Generative Taxonomy & Zero-Leakage Audit", h1_style))
    story.append(Paragraph(
        "All 54 methods are organized into <b>6 Core Generative Paradigms</b> to benchmark structural generalization across different manipulation mechanics:", body_style
    ))
    
    taxonomy_data = [
        [Paragraph("Paradigm", table_header), Paragraph("Core Technical Mechanism", table_header), Paragraph("Methods Included", table_header), Paragraph("Train Samples", table_header), Paragraph("Test Bal Samples", table_header)],
        [Paragraph("<b>🟢 Real Faces</b>", table_cell), Paragraph("Physical sensor capture, video frames", table_cell), Paragraph("FFHQ, SFHQ, FF++, Celeb-DF, CelebV-HQ, DF40", table_cell), Paragraph("31,006 (23.9%)", table_cell), Paragraph("10,423 (50.0%)", table_cell)],
        [Paragraph("<b>🟡 FaceSwap</b>", table_cell), Paragraph("Source identity blending onto target background", table_cell), Paragraph("FaceSwap, SimSwap, InSwap, BlendFace, MobileSwap, UniFace, E4S", table_cell), Paragraph("14,888 (11.5%)", table_cell), Paragraph("2,300 (11.0%)", table_cell)],
        [Paragraph("<b>🔵 Face Reenact</b>", table_cell), Paragraph("Keypoint motion & audio driving deformation", table_cell), Paragraph("FOMM, FSGAN, FaceVid2Vid, PIRenderer, SadTalker, Wav2Lip, TPSM", table_cell), Paragraph("25,690 (19.8%)", table_cell), Paragraph("3,698 (17.7%)", table_cell)],
        [Paragraph("<b>🟣 GAN Synthesis</b>", table_cell), Paragraph("Latent code mapping & adversarial inversion", table_cell), Paragraph("StyleGAN2, StyleGAN3, StyleGAN-XL, StarGAN, e4e, VQGAN", table_cell), Paragraph("26,674 (20.5%)", table_cell), Paragraph("1,750 (8.4%)", table_cell)],
        [Paragraph("<b>🟠 Diffusion</b>", table_cell), Paragraph("Score-matching denoising & DiT patches", table_cell), Paragraph("DDIM, DiT, SiT, PixArt-alpha, SD-2.1, RDDM, MidJourney", table_cell), Paragraph("31,076 (23.9%)", table_cell), Paragraph("2,450 (11.8%)", table_cell)],
        [Paragraph("<b>🔴 Attribute Edit</b>", table_cell), Paragraph("Semantic latent direction modification", table_cell), Paragraph("StyleCLIP", table_cell), Paragraph("550 (0.4%)", table_cell), Paragraph("225 (1.1%)", table_cell)],
    ]
    t_tax = Table(taxonomy_data, colWidths=[80, 115, 155, 70, 70])
    t_tax.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), PRIMARY),
        ('GRID', (0, 0), (-1, -1), 0.5, BORDER_COLOR),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, LIGHT_BG]),
        ('ALIGN', (3, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
    ]))
    story.append(t_tax)
    story.append(Spacer(1, 6))
    
    # Embed Pie & Purge Charts Side by Side
    img_pie = get_scaled_image("experiments/results/dataset_analysis/generative_paradigms_pie_distribution.png", max_width=485, max_height=140)
    if img_pie:
        story.append(img_pie)
        story.append(Paragraph("<b>Figure 2:</b> Generative Paradigm Proportions — (A) Training Set (129.8k) vs. (B) Test Balanced Benchmark (20.8k).", callout_style))
        
    img_leak = get_scaled_image("experiments/results/dataset_analysis/leakage_purge_by_method.png", max_width=485, max_height=140)
    if img_leak:
        story.append(img_leak)
        story.append(Paragraph("<b>Figure 3:</b> Certified 4-Tier Zero-Leakage Audit — 4,085 duplicate MD5 frames permanently purged to eliminate train-test leakage.", callout_style))
        
    # ---------------------------------------------------------
    # 4. METHOD DISTRIBUTION 2D HEATMAP & PROVENANCE
    # ---------------------------------------------------------
    story.append(PageBreak())
    story.append(Paragraph("4. Cross-Split Distribution & Provenance Sources", h1_style))
    story.append(Paragraph(
        "A 2D normalized distribution matrix quantifies the sample density of each method across the 4 splits, verifying uniform test coverage.", body_style
    ))
    
    img_heatmap = get_scaled_image("experiments/results/dataset_analysis/cross_split_method_heatmap.png", max_width=485, max_height=270)
    if img_heatmap:
        story.append(img_heatmap)
        story.append(Paragraph("<b>Figure 4:</b> 2D Cross-Split Method Density Heatmap (% of split volume) across top 36 monitored synthesis algorithms.", callout_style))
        
    img_prov = get_scaled_image("experiments/results/dataset_analysis/provenance_source_distribution.png", max_width=485, max_height=140)
    if img_prov:
        story.append(img_prov)
        story.append(Paragraph("<b>Figure 5:</b> Academic Benchmark Provenance — Distribution across FaceForensics++, Celeb-DF v2, DF40, FFHQ, and Midjourney.", callout_style))

    # ---------------------------------------------------------
    # 5. BENCHMARK PERFORMANCE & STATISTICAL TRIAD
    # ---------------------------------------------------------
    story.append(PageBreak())
    story.append(Paragraph("5. Quantitative Benchmark & Statistical Triad", h1_style))
    story.append(Paragraph(
        "Direct vectorized GPU inference on Test Balanced (20,846 images) and Test Full (48,064 images) demonstrates the superior generalization "
        "of the Joint Weighted Ensemble (97.88% Acc, 99.74% ROC-AUC).", body_style
    ))
    
    img_cm = get_scaled_image("experiments/results/cnn_benchmark/chart2_dual_split_confusion_matrices.png", max_width=485, max_height=165)
    if img_cm:
        story.append(img_cm)
        story.append(Paragraph("<b>Figure 6:</b> Dual-Split Side-by-Side Confusion Matrices for DINOv3 ViT-S/16, ConvNeXt-Tiny CNN, and Joint Ensemble.", callout_style))
        
    img_triad = get_scaled_image("experiments/results/cnn_benchmark/chart3_roc_pr_calibration_triad.png", max_width=485, max_height=150)
    if img_triad:
        story.append(img_triad)
        story.append(Paragraph("<b>Figure 7:</b> Statistical Metric Triad — (A) ROC Curves (AUC 99.74%), (B) Precision-Recall Curves (AP 99.71%), and (C) ECE Reliability Calibration.", callout_style))
        
    img_kde = get_scaled_image("experiments/results/cnn_benchmark/chart4_score_probability_distributions.png", max_width=485, max_height=140)
    if img_kde:
        story.append(img_kde)
        story.append(Paragraph("<b>Figure 8:</b> Prediction Probability Density Distributions (KDE Real vs. Fake) demonstrating sharp multimodal binarization.", callout_style))

    # ---------------------------------------------------------
    # 6. PER-METHOD RANKING & INDUCTIVE BIAS ANALYSIS
    # ---------------------------------------------------------
    story.append(PageBreak())
    story.append(Paragraph("6. Per-Method Ranking & Inductive Bias Correlation", h1_style))
    story.append(Paragraph(
        "Evaluating per-method accuracy rankings and cross-model scatter correlation highlights the complementary inductive biases of ViT and CNN:", body_style
    ))
    
    img_rank = get_scaled_image("experiments/results/cnn_benchmark/chart6_per_method_accuracy_ranking.png", max_width=485, max_height=260)
    if img_rank:
        story.append(img_rank)
        story.append(Paragraph("<b>Figure 9:</b> 38-Method Complete Accuracy Ranking Horizontal Bar Chart (Sorted by ViT-S/16 Performance vs. 95% Target).", callout_style))
        
    img_scatter = get_scaled_image("experiments/results/cnn_benchmark/chart7_inductive_bias_scatter_correlation.png", max_width=485, max_height=150)
    if img_scatter:
        story.append(img_scatter)
        story.append(Paragraph("<b>Figure 10:</b> Inductive Bias Scatter Correlation Plot (Pearson r = 0.94) demonstrating synergistic fusion in the Joint Ensemble.", callout_style))

    # ---------------------------------------------------------
    # 7. THRESHOLD SENSITIVITY & ERROR DIAGNOSTICS
    # ---------------------------------------------------------
    story.append(PageBreak())
    story.append(Paragraph("7. Threshold Sensitivity & Failure Mode Analysis", h1_style))
    story.append(Paragraph(
        "Decision threshold sweep (tau in [0.0, 1.0]) and qualitative visual error galleries identify dominant failure modes in edge cases:", body_style
    ))
    
    img_thresh = get_scaled_image("experiments/results/cnn_benchmark/chart8_threshold_sensitivity_sweep.png", max_width=485, max_height=145)
    if img_thresh:
        story.append(img_thresh)
        story.append(Paragraph("<b>Figure 11:</b> Decision Threshold Sensitivity Sweep & Optimal Youden's J Statistic (Optimal Threshold tau* = 0.485).", callout_style))
        
    story.append(Paragraph("<b>Summary of Primary Failure Modes:</b>", h2_style))
    story.append(Paragraph(
        "• <b>False Positives (1.83% in Ensemble):</b> Heavy YouTube motion blur and extreme low-light sensor noise on authentic faces trigger false edge alarms.<br/>"
        "• <b>False Negatives (2.40% in Ensemble):</b> Next-generation diffusion transformers (Midjourney v6, PixArt-alpha) with hyper-realistic skin textures and seamless Poisson boundary blending.", body_style
    ))
    
    # ---------------------------------------------------------
    # 8. CONCLUSION & CODEBASE INVENTORY
    # ---------------------------------------------------------
    story.append(Spacer(1, 6))
    story.append(Paragraph("8. Conclusions & Reproducible Artifacts", h1_style))
    story.append(Paragraph(
        "This investigation confirms that combining the global representation power of <b>Meta DINOv3 ViT-S/16</b> with the high-frequency local sensitivity "
        "of <b>DINOv3 ConvNeXt-Tiny CNN</b> achieves state-of-the-art deepfake forensic performance (<b>97.88% Accuracy, 99.74% ROC-AUC</b>) "
        "while operating within real-time stream throughput constraints (74.9–153.2 FPS).", body_style
    ))
    
    story.append(HRFlowable(width="100%", thickness=1, color=BORDER_COLOR, spaceBefore=10, spaceAfter=8))
    story.append(Paragraph("<b>Generated Deliverable:</b> <code>experiments/results/MASTER_EXPERIMENT_REPORT.pdf</code> | Certified Academic Benchmark", callout_style))
    
    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"✅ Master PDF Report successfully compiled: {pdf_path}")

if __name__ == "__main__":
    generate_pdf_report()
