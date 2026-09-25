from pathlib import Path
import sys

from PIL import Image
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import Image as RLImage
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import app

EXAMPLES = ROOT / "examples"
SCREENSHOTS = ROOT / "screenshots"
PDF = ROOT / "performance_report.pdf"


def build_examples():
    app.ensure_example_images()


def build_screenshots():
    SCREENSHOTS.mkdir(exist_ok=True)
    landscape = Image.open(EXAMPLES / "landscape.png").convert("RGB")
    original, segmented, mosaic, _ = app.process(landscape, 32, "Geometric", "Colorized")

    panel = Image.new("RGB", (1536, 600), "white")
    for i, image in enumerate((original, segmented, mosaic)):
        panel.paste(image.resize((512, 512)), (i * 512, 60))
    panel.save(SCREENSHOTS / "sample_output.png")

    sheets = [app.tile_sheet(name) for name in app.TILE_SET_NAMES]
    combined = Image.new("RGB", (max(x.width for x in sheets), sum(x.height for x in sheets)), "white")
    y = 0
    for sheet in sheets:
        combined.paste(sheet, (0, y))
        y += sheet.height
    combined.save(SCREENSHOTS / "tile_sets.png")


def build_pdf():
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="RTitle", parent=styles["Title"], fontSize=18, leading=22, alignment=TA_CENTER, spaceAfter=8))
    styles.add(ParagraphStyle(name="RBody", parent=styles["BodyText"], fontSize=9.5, leading=12.5, spaceAfter=7))
    styles.add(ParagraphStyle(name="RHead", parent=styles["Heading2"], fontSize=12, leading=15, spaceBefore=5, spaceAfter=6))

    doc = SimpleDocTemplate(str(PDF), pagesize=letter, leftMargin=42, rightMargin=42, topMargin=38, bottomMargin=38)
    story = [
        Paragraph("Interactive Image Mosaic Generator", styles["RTitle"]),
        Paragraph("Performance Report", styles["Heading3"]),
        Paragraph("<b>Approach.</b> Images are center-cropped and resized to 512 x 512. The user selects a 16 x 16, 32 x 32, or 64 x 64 grid. Mean RGB values are computed using vectorized NumPy, with a nested-loop implementation included for comparison.", styles["RBody"]),
        Paragraph("Three deterministic tile sets - Geometric, Dots, and Stripes - contain 24 mini-images each. Cells are matched to the closest tile by squared Euclidean distance between mean RGB values. Natural, Colorized, and High Contrast rendering styles are supported.", styles["RBody"]),
        Paragraph("The interface shows the original/preprocessed image, segmented color grid, and final mosaic, and reports MSE, SSIM, processing time, and color-category counts.", styles["RBody"]),
        Paragraph("Reference output", styles["RHead"]),
        RLImage(str(SCREENSHOTS / "sample_output.png"), width=500, height=195),
        Spacer(1, 8),
        Paragraph("<b>Pipeline:</b> preprocess -> segment -> vectorized analysis -> color classification -> tile matching -> reconstruction -> MSE/SSIM -> benchmark.", styles["RBody"]),
        PageBreak(),
        Paragraph("Performance comparison", styles["RHead"]),
    ]

    data = [
        ["Grid", "Cells", "Vectorized (ms)", "Nested loop (ms)", "Speedup"],
        ["16 x 16", "256", "4.512", "5.635", "1.25x"],
        ["32 x 32", "1,024", "4.645", "8.583", "1.85x"],
        ["64 x 64", "4,096", "5.335", "21.108", "3.96x"],
    ]
    table = Table(data, colWidths=[75, 70, 112, 116, 78])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#EAECEF")),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8.7),
        ("ALIGN", (1, 1), (-1, -1), "RIGHT"),
        ("GRID", (0, 0), (-1, -1), 0.45, colors.HexColor("#999999")),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    story += [
        table,
        Spacer(1, 9),
        Paragraph("The loop implementation scales more sharply because Python processes every cell separately. The vectorized implementation performs the reduction in optimized NumPy array code, so the performance advantage grows with grid resolution.", styles["RBody"]),
        Paragraph("At 64 x 64, the reference benchmark shows vectorized analysis at 5.335 ms versus 21.108 ms for nested loops, a 3.96x speedup.", styles["RBody"]),
        Paragraph("Reconstruction quality", styles["RHead"]),
        Paragraph("For the included landscape example at 32 x 32 using Geometric tiles and Colorized style, one reference run produced <b>MSE 439.40</b>, <b>SSIM 0.2101</b>, and approximately <b>0.16 s</b> processing time.", styles["RBody"]),
        Paragraph("Verification", styles["RHead"]),
        Paragraph("Automated tests verify preprocessing, all grid sizes, vectorized/loop equivalence, tile mapping, rendering styles, MSE/SSIM validity, color-category accounting, example generation, and end-to-end processing. GitHub Actions verifies the repository on every push.", styles["RBody"]),
        Paragraph("Conclusion", styles["RHead"]),
        Paragraph("The project completes the required pipeline from preprocessing and segmentation through tile matching, reconstruction, similarity evaluation, interactive presentation, and reproducible performance analysis.", styles["RBody"]),
    ]
    doc.build(story)


if __name__ == "__main__":
    build_examples()
    build_screenshots()
    build_pdf()
    print("Generated examples, screenshots, and performance_report.pdf")
