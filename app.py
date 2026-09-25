import time
from dataclasses import dataclass
from typing import Dict, Tuple

import gradio as gr
import numpy as np
from PIL import Image, ImageDraw
from skimage.metrics import structural_similarity as ssim

CANVAS = 512
TILE_PX = 24
GRID_OPTIONS = [16, 32, 64]
TILE_SET_NAMES = ["Geometric", "Dots", "Stripes"]
STYLE_OPTIONS = ["Natural", "Colorized", "High Contrast"]


@dataclass(frozen=True)
class TileSet:
    images: np.ndarray
    means: np.ndarray


PALETTE = [
    (35, 45, 75), (55, 90, 140), (70, 135, 180), (80, 175, 200),
    (45, 105, 75), (70, 150, 95), (125, 175, 85), (185, 195, 95),
    (145, 85, 55), (195, 120, 65), (225, 155, 75), (235, 195, 110),
    (120, 55, 85), (175, 70, 115), (215, 105, 145), (235, 150, 175),
    (75, 65, 120), (115, 90, 170), (155, 120, 205), (195, 165, 225),
    (65, 65, 65), (105, 105, 105), (165, 165, 165), (225, 225, 225),
]


def _draw_pattern(base: Tuple[int, int, int], pattern: str, index: int) -> np.ndarray:
    """Create one small patterned RGB image tile."""
    img = Image.new("RGB", (TILE_PX, TILE_PX), base)
    draw = ImageDraw.Draw(img)
    accent = tuple(min(255, c + 45) for c in base)
    dark = tuple(max(0, c - 35) for c in base)

    if pattern == "Geometric":
        mode = index % 4
        if mode == 0:
            draw.rectangle((3, 3, TILE_PX - 4, TILE_PX - 4), outline=accent, width=2)
            draw.line((0, TILE_PX // 2, TILE_PX, TILE_PX // 2), fill=dark, width=2)
            draw.line((TILE_PX // 2, 0, TILE_PX // 2, TILE_PX), fill=dark, width=2)
        elif mode == 1:
            draw.polygon([(TILE_PX // 2, 2), (TILE_PX - 3, TILE_PX - 3), (3, TILE_PX - 3)], outline=accent)
            draw.line((2, 2, TILE_PX - 3, TILE_PX - 3), fill=dark, width=2)
        elif mode == 2:
            draw.ellipse((3, 3, TILE_PX - 4, TILE_PX - 4), outline=accent, width=2)
            draw.rectangle((8, 8, TILE_PX - 9, TILE_PX - 9), fill=dark)
        else:
            for x in range(-TILE_PX, TILE_PX * 2, 7):
                draw.line((x, 0, x + TILE_PX, TILE_PX), fill=accent, width=2)
    elif pattern == "Dots":
        for y in range(4, TILE_PX, 7):
            for x in range(4, TILE_PX, 7):
                radius = 2 if (x + y + index) % 2 == 0 else 1
                fill = accent if (x + y) % 3 else dark
                draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=fill)
    elif pattern == "Stripes":
        orientation = index % 3
        if orientation == 0:
            for y in range(2, TILE_PX, 5):
                draw.line((0, y, TILE_PX, y), fill=accent, width=2)
        elif orientation == 1:
            for x in range(2, TILE_PX, 5):
                draw.line((x, 0, x, TILE_PX), fill=dark, width=2)
        else:
            for x in range(-TILE_PX, TILE_PX * 2, 6):
                draw.line((x, 0, x + TILE_PX, TILE_PX), fill=accent, width=2)
    else:
        raise ValueError(f"Unknown tile pattern: {pattern}")

    return np.asarray(img, dtype=np.uint8)


def build_tile_sets() -> Dict[str, TileSet]:
    """Build three deterministic predefined tile collections."""
    result: Dict[str, TileSet] = {}
    for pattern in TILE_SET_NAMES:
        images = np.stack([_draw_pattern(color, pattern, i) for i, color in enumerate(PALETTE)])
        means = images.reshape(len(PALETTE), -1, 3).mean(axis=1).astype(np.float32)
        result[pattern] = TileSet(images=images, means=means)
    return result


TILE_SETS = build_tile_sets()


def ensure_example_images():
    """Generate deterministic example images locally so the app is self-contained."""
    from pathlib import Path

    example_dir = Path(__file__).parent / "examples"
    example_dir.mkdir(exist_ok=True)

    size = 512

    gradient_path = example_dir / "gradient.png"
    if not gradient_path.exists():
        y, x = np.mgrid[0:size, 0:size]
        gradient = np.stack([
            x / (size - 1) * 255,
            y / (size - 1) * 255,
            (1 - x / (size - 1)) * 190 + 35,
        ], axis=2).astype(np.uint8)
        Image.fromarray(gradient).save(gradient_path)

    landscape_path = example_dir / "landscape.png"
    if not landscape_path.exists():
        img = Image.new("RGB", (size, size), (125, 195, 235))
        draw = ImageDraw.Draw(img)
        draw.rectangle((0, 300, size, size), fill=(66, 132, 76))
        draw.ellipse((380, 45, 455, 120), fill=(250, 215, 90))
        draw.polygon([(0, 330), (145, 150), (270, 330)], fill=(72, 105, 128))
        draw.polygon([(170, 330), (335, 125), (512, 330)], fill=(58, 88, 116))
        draw.polygon([(110, 195), (145, 150), (183, 199)], fill=(238, 241, 245))
        draw.polygon([(292, 178), (335, 125), (379, 181)], fill=(238, 241, 245))
        draw.rectangle((0, 390, size, 512), fill=(47, 113, 148))
        landscape_path.parent.mkdir(exist_ok=True)
        img.save(landscape_path)

    portrait_path = example_dir / "portrait.png"
    if not portrait_path.exists():
        img = Image.new("RGB", (size, size), (225, 205, 185))
        draw = ImageDraw.Draw(img)
        draw.ellipse((138, 72, 374, 330), fill=(192, 145, 112))
        draw.ellipse((186, 150, 218, 176), fill=(40, 45, 50))
        draw.ellipse((294, 150, 326, 176), fill=(40, 45, 50))
        draw.arc((215, 175, 300, 270), 25, 155, fill=(115, 65, 60), width=5)
        draw.rectangle((135, 320, 377, 512), fill=(65, 85, 125))
        img.save(portrait_path)


ensure_example_images()


def center_crop_resize(image: Image.Image, size: int = CANVAS) -> Image.Image:
    """Center-crop an image to square and resize to a fixed resolution."""
    im = image.convert("RGB")
    w, h = im.size
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    return im.crop((left, top, left + side, top + side)).resize(
        (size, size), Image.Resampling.LANCZOS
    )


def vectorized_cell_means(arr: np.ndarray, grid: int) -> np.ndarray:
    """Compute mean RGB for every grid cell using vectorized NumPy operations."""
    h, w, _ = arr.shape
    ch, cw = h // grid, w // grid
    cropped = arr[: grid * ch, : grid * cw]
    return cropped.reshape(grid, ch, grid, cw, 3).mean(axis=(1, 3))


def loop_cell_means(arr: np.ndarray, grid: int) -> np.ndarray:
    """Reference nested-loop implementation used for the performance comparison."""
    h, w, _ = arr.shape
    ch, cw = h // grid, w // grid
    out = np.zeros((grid, grid, 3), dtype=np.float32)
    for r in range(grid):
        for c in range(grid):
            cell = arr[r * ch : (r + 1) * ch, c * cw : (c + 1) * cw]
            out[r, c] = cell.mean(axis=(0, 1))
    return out


def classify_color_categories(cell_means: np.ndarray) -> Dict[str, int]:
    """Classify cells into readable color categories for assignment analysis."""
    rgb = cell_means.reshape(-1, 3).astype(np.float32)
    maxc = rgb.max(axis=1)
    minc = rgb.min(axis=1)
    delta = maxc - minc

    counts = {name: 0 for name in ["Red", "Orange", "Yellow", "Green", "Cyan", "Blue", "Purple", "Neutral"]}

    for (r, g, b), mx, d in zip(rgb, maxc, delta):
        if mx < 55 or d < 18:
            counts["Neutral"] += 1
            continue

        if mx == r:
            hue = (60 * ((g - b) / d) + 360) % 360
        elif mx == g:
            hue = 60 * ((b - r) / d + 2)
        else:
            hue = 60 * ((r - g) / d + 4)

        if hue < 20 or hue >= 345:
            counts["Red"] += 1
        elif hue < 45:
            counts["Orange"] += 1
        elif hue < 70:
            counts["Yellow"] += 1
        elif hue < 165:
            counts["Green"] += 1
        elif hue < 200:
            counts["Cyan"] += 1
        elif hue < 255:
            counts["Blue"] += 1
        else:
            counts["Purple"] += 1

    return counts

def nearest_tile_indices(cell_means: np.ndarray, tile_set_name: str) -> np.ndarray:
    """Map every cell to the tile whose mean RGB is closest."""
    tile_set = TILE_SETS[tile_set_name]
    flat = cell_means.reshape(-1, 3).astype(np.float32)
    distances = ((flat[, None, :] - tile_set.means[None, :, :]) ** 2).sum(axis=2)
    return distances.argmin(axis=1).reshape(cell_means.shape[:2])

def segmented_preview(cell_means: np.ndarray, out_size: int = CANVAS) -> Image.Image:
    """Visualize the segmented image as a flat-color grid."""
    grid = cell_means.shape[0]
    cell = out_size // grid
    quant = np.clip(cell_means, 0, 255).astype(np.uint8)
    preview = np.repeat(np.repeat(quant, cell, axis=0), cell, axis=1)
    return Image.fromarray(preview[:out_size, :out_size])


def render_mosaic(
    indices: np.ndarray,
    cell_means: np.ndarray,
    tile_set_name: str,
    style: str = "Natural",
) -> Image.Image:
    """Replace every grid cell with the selected representative mini-image tile."""
    tile_set = TILE_SETS[tile_set_name]
    grid = indices.shape[0]
    cell = CANVAS // grid
    out = np.empty((CANVAS, CANVAS, 3), dtype=np.uint8)

    for r in range(grid):
        for c in range(grid):
            tile = Image.fromarray(tile_set.images[indices[r, c]]).resize(
                (cell, cell), Image.Resampling.BILINEAR
            )
            ta = np.asarray(tile, dtype=np.float32)

            if style == "Colorized":
                target = cell_means[r, c]
                source = ta.reshape(-1, 3).mean(axis=0)
                ta = np.clip(ta * (0.35 + 0.65 * (target + 1) / (source + 1)), 0, 255)
            elif style == "High Contrast":
                ta = np.clip((ta - 128) * 1.25 + 128, 0, 255)

            out[r * cell : (r + 1) * cell, c * cell : (c + 1) * cell] = ta.astype(np.uint8)

    return Image.fromarray(out)


def similarity_metrics(original: Image.Image, mosaic: Image.Image) -> Tuple[float, float]:
    """Return MSE and SSIM between the preprocessed image and final mosaic."""
    a = np.asarray(original, dtype=np.uint8)
    b = np.asarray(mosaic.resize(original.size), dtype=np.uint8)
    mse = float(np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2))
    score = float(ssim(a, b, channel_axis=2, data_range=255))
    return mse, score


def process(image: Image.Image, grid: int, tile_set_name: str, style: str):
    """Main Gradio processing pipeline."""
    if image is None:
        raise gr.Error("Please upload an image first.")

    grid = int(grid)
    start = time.perf_counter()

    original = center_crop_resize(image)
    arr = np.asarray(original)
    means = vectorized_cell_means(arr, grid)
    categories = classify_color_categories(means)
    segmented = segmented_preview(means)
    indices = nearest_tile_indices(means, tile_set_name)
    mosaic = render_mosaic(indices, means, tile_set_name, style)
    mse, score = similarity_metrics(original, mosaic)
    elapsed = time.perf_counter() - start

    category_text = ", ".join(f"{k}: {v}" for k, v in categories.items() if v)
    report = (
        f"### Results\n"
        f"**Grid:** {grid} × {grid} ({grid * grid:,} cells)  \n"
        f"**Tile set:** {tile_set_name}  \n"
        f"**Style:** {style}  \n"
        f"**MSE:** {mse:,.2f} *(lower is better)*  \n"
        f"**SSIM:** {score:.4f} *(higher is better; max 1.0)*  \n"
        f"**Processing time:** {elapsed:.4f} s  \n"
        f"**Color categories:** {category_text}"
    )
    return original, segmented, mosaic, report


def benchmark(image: Image.Image):
    """Compare vectorized and nested-loop cell analysis for 16/32/64 grids."""
    if image is None:
        raise gr.Error("Upload an image before benchmarking.")

    arr = np.asarray(center_crop_resize(image))
    rows = []
    speedups = []

    for grid in GRID_OPTIONS:
        vectorized_times = []
        loop_times = []
        for _ in range(20):
            start = time.perf_counter()
            vectorized_cell_means(arr, grid)
            vectorized_times.append(time.perf_counter() - start)

            start = time.perf_counter()
            loop_cell_means(arr, grid)
            loop_times.append(time.perf_counter() - start)

        v = float(np.median(vectorized_times))
        l = float(np.median(loop_times))
        speedup = l / v if v else float("inf")
        speedups.append(speedup)
        rows.append([
            f"{grid}×{grid}",
            grid * grid,
            round(v * 1000, 3),
            round(l * 1000, 3),
            round(speedup, 2),
        ])

    analysis = (
        "### Scaling analysis\n"
        f"The vectorized method is **{speedups[-1]:.2f}× faster** than the nested-loop "
        "implementation at 64×64 on this run. The loop cost rises more sharply as the number "
        "of cells increases because Python executes work cell-by-cell, while NumPy performs the "
        "vectorized reduction in optimized compiled code."
    )
    return rows, analysis


def tile_sheet(tile_set_name: str) -> Image.Image:
    """Display all 24 tiles for the chosen tile set."""
    tile_set = TILE_SETS[tile_set_name]
    scale = 3
    cols = 8
    rows = 3
    sheet = Image.new("RGB", (cols * TILE_PX * scale, rows * TILE_PX * scale), "white")
    for i, arr in enumerate(tile_set.images):
        tile = Image.fromarray(arr).resize((TILE_PX * scale, TILE_PX * scale), Image.Resampling.NEAREST)
        x = (i % cols) * TILE_PX * scale
        y = (i // cols) * TILE_PX * scale
        sheet.paste(tile, (x, y))
    return sheet


CSS = """
#title {text-align:center; margin-bottom:0}
#subtitle {text-align:center; opacity:.76; margin-bottom:10px}
"""

with gr.Blocks(title="Interactive Image Mosaic Generator") as demo:
    gr.Markdown("# 🧩 Interactive Image Mosaic Generator", elem_id="title")
    gr.Markdown(
        "Upload an image, segment it into a grid, and reconstruct it from representative mini-image tiles.",
        elem_id="subtitle",
    )

    with gr.Row():
        image_in = gr.Image(type="pil", label="1. Upload Image")
        with gr.Column():
            grid = gr.Radio(GRID_OPTIONS, value=32, label="2. Grid Size")
            tile_set_name = gr.Dropdown(TILE_SET_NAMES, value="Geometric", label="3. Tile Set")
            style = gr.Dropdown(STYLE_OPTIONS, value="Colorized", label="4. Rendering Style")
            run = gr.Button("Generate Mosaic", variant="primary")
            bench = gr.Button("Run Performance Benchmark")

    tile_preview = gr.Image(value=tile_sheet("Geometric"), label="Current Tile Set", interactive=False)
    tile_set_name.change(tile_sheet, tile_set_name, tile_preview)

    with gr.Row():
        original_out = gr.Image(label="Original / Preprocessed")
        segmented_out = gr.Image(label="Segmented Color Grid")
        mosaic_out = gr.Image(label="Final Tile Mosaic")

    result_md = gr.Markdown()

    gr.Markdown("## Performance: Vectorized NumPy vs. Nested Loops")
    table = gr.Dataframe(
        headers=["Grid", "Cells", "Vectorized (ms)", "Loop (ms)", "Speedup"],
        datatype=["str", "number", "number", "number", "number"],
        interactive=False,
    )
    analysis_md = gr.Markdown()

    gr.Markdown("## Try an included example")
    gr.Examples(
        examples=[
            ["examples/gradient.png", 32, "Geometric", "Colorized"],
            ["examples/landscape.png", 32, "Dots", "Natural"],
            ["examples/portrait.png", 64, "Stripes", "High Contrast"],
        ],
        inputs=[image_in, grid, tile_set_name, style],
    )

    run.click(
        process,
        [image_in, grid, tile_set_name, style],
        [original_out, segmented_out, mosaic_out, result_md],
    )
    bench.click(benchmark, image_in, [table, analysis_md])


if __name__ == "__main__":
    demo.launch(css=CSS)
