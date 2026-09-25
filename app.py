import time

import gradio as gr
import numpy as np
from PIL import Image, ImageDraw
from skimage.metrics import structural_similarity as ssim

CANVAS = 512
TILE_PX = 24


def build_tile_set():
    """Create a deterministic predefined mini-image tile set in memory."""
    palette = [
        (35, 45, 75), (55, 90, 140), (70, 135, 180), (80, 175, 200),
        (45, 105, 75), (70, 150, 95), (125, 175, 85), (185, 195, 95),
        (145, 85, 55), (195, 120, 65), (225, 155, 75), (235, 195, 110),
        (120, 55, 85), (175, 70, 115), (215, 105, 145), (235, 150, 175),
        (75, 65, 120), (115, 90, 170), (155, 120, 205), (195, 165, 225),
        (65, 65, 65), (105, 105, 105), (165, 165, 165), (225, 225, 225),
    ]

    tiles, means = [], []
    for i, base in enumerate(palette):
        img = Image.new("RGB", (TILE_PX, TILE_PX), base)
        draw = ImageDraw.Draw(img)
        accent = tuple(min(255, c + 45) for c in base)
        dark = tuple(max(0, c - 35) for c in base)

        pattern = i % 4
        if pattern == 0:
            for x in range(-TILE_PX, TILE_PX * 2, 6):
                draw.line((x, 0, x + TILE_PX, TILE_PX), fill=accent, width=2)
        elif pattern == 1:
            for y in range(3, TILE_PX, 6):
                draw.line((0, y, TILE_PX, y), fill=dark, width=2)
        elif pattern == 2:
            for y in range(3, TILE_PX, 8):
                for x in range(3, TILE_PX, 8):
                    draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=accent)
        else:
            draw.rectangle((3, 3, TILE_PX - 4, TILE_PX - 4), outline=accent, width=2)
            draw.line((0, TILE_PX // 2, TILE_PX, TILE_PX // 2), fill=dark, width=2)
            draw.line((TILE_PX // 2, 0, TILE_PX // 2, TILE_PX), fill=dark, width=2)

        arr = np.asarray(img, dtype=np.uint8)
        tiles.append(arr)
        means.append(arr.reshape(-1, 3).mean(axis=0))

    return np.stack(tiles), np.asarray(means, dtype=np.float32)


TILES, TILE_MEANS = build_tile_set()


def center_crop_resize(image, size=CANVAS):
    im = image.convert("RGB")
    w, h = im.size
    side = min(w, h)
    left, top = (w - side) // 2, (h - side) // 2
    return im.crop((left, top, left + side, top + side)).resize(
        (size, size), Image.Resampling.LANCZOS
    )


def vectorized_cell_means(arr, grid):
    h, w, _ = arr.shape
    ch, cw = h // grid, w // grid
    cropped = arr[: grid * ch, : grid * cw]
    return cropped.reshape(grid, ch, grid, cw, 3).mean(axis=(1, 3))


def loop_cell_means(arr, grid):
    h, w, _ = arr.shape
    ch, cw = h // grid, w // grid
    out = np.zeros((grid, grid, 3), dtype=np.float32)
    for r in range(grid):
        for c in range(grid):
            cell = arr[r * ch : (r + 1) * ch, c * cw : (c + 1) * cw]
            out[r, c] = cell.mean(axis=(0, 1))
    return out


def nearest_tile_indices(cell_means):
    flat = cell_means.reshape(-1, 3).astype(np.float32)
    distances = ((flat[:, None, :] - TILE_MEANS[None, :, :]) ** 2).sum(axis=2)
    return distances.argmin(axis=1).reshape(cell_means.shape[:2])


def segmented_preview(cell_means, out_size=CANVAS):
    grid = cell_means.shape[0]
    cell = out_size // grid
    quant = np.clip(cell_means, 0, 255).astype(np.uint8)
    preview = np.repeat(np.repeat(quant, cell, axis=0), cell, axis=1)
    return Image.fromarray(preview[:out_size, :out_size])


def render_mosaic(indices, cell_means, style="Natural"):
    grid = indices.shape[0]
    cell = CANVAS // grid
    out = np.empty((CANVAS, CANVAS, 3), dtype=np.uint8)

    for r in range(grid):
        for c in range(grid):
            tile = Image.fromarray(TILES[indices[r, c]]).resize(
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


def metrics(original, mosaic):
    a = np.asarray(original, dtype=np.uint8)
    b = np.asarray(mosaic.resize(original.size), dtype=np.uint8)
    mse = float(np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2))
    score = float(ssim(a, b, channel_axis=2, data_range=255))
    return mse, score


def process(image, grid, style):
    if image is None:
        raise gr.Error("Please upload an image first.")

    grid = int(grid)
    start = time.perf_counter()
    original = center_crop_resize(image)
    arr = np.asarray(original)
    means = vectorized_cell_means(arr, grid)
    segmented = segmented_preview(means)
    ids = nearest_tile_indices(means)
    mosaic = render_mosaic(ids, means, style)
    mse, score = metrics(original, mosaic)
    elapsed = time.perf_counter() - start

    report = (
        f"### Results\n"
        f"**Grid:** {grid} × {grid} ({grid * grid:,} cells)  \n"
        f"**MSE:** {mse:,.2f} *(lower is better)*  \n"
        f"**SSIM:** {score:.4f} *(higher is better; max 1.0)*  \n"
        f"**Processing time:** {elapsed:.4f} s"
    )
    return original, segmented, mosaic, report


def benchmark(image):
    if image is None:
        raise gr.Error("Upload an image before benchmarking.")

    arr = np.asarray(center_crop_resize(image))
    rows = []
    for grid in (16, 32, 64):
        vt, lt = [], []
        for _ in range(5):
            start = time.perf_counter()
            vectorized_cell_means(arr, grid)
            vt.append(time.perf_counter() - start)

            start = time.perf_counter()
            loop_cell_means(arr, grid)
            lt.append(time.perf_counter() - start)

        v, l = np.median(vt), np.median(lt)
        rows.append([
            f"{grid}×{grid}",
            grid * grid,
            round(v * 1000, 3),
            round(l * 1000, 3),
            round(l / v, 2) if v else None,
        ])
    return rows


CSS = """
#title {text-align:center; margin-bottom:0}
#subtitle {text-align:center; opacity:.75}
"""

with gr.Blocks(title="Interactive Image Mosaic Generator") as demo:
    gr.Markdown("# 🧩 Interactive Image Mosaic Generator", elem_id="title")
    gr.Markdown(
        "Upload a photo, analyze it as a color grid, and reconstruct it from mini-image tiles.",
        elem_id="subtitle",
    )

    with gr.Row():
        image_in = gr.Image(type="pil", label="1. Upload Image")
        with gr.Column():
            grid = gr.Radio([16, 32, 64], value=32, label="Grid Size")
            style = gr.Dropdown(
                ["Natural", "Colorized", "High Contrast"],
                value="Colorized",
                label="Tile Style",
            )
            run = gr.Button("Generate Mosaic", variant="primary")
            bench = gr.Button("Run Performance Benchmark")

    with gr.Row():
        original_out = gr.Image(label="Original / Preprocessed")
        segmented_out = gr.Image(label="Segmented Color Grid")
        mosaic_out = gr.Image(label="Final Tile Mosaic")

    result_md = gr.Markdown()
    gr.Markdown("### Vectorized vs. Loop Benchmark")
    table = gr.Dataframe(
        headers=["Grid", "Cells", "Vectorized (ms)", "Loop (ms)", "Speedup"],
        datatype=["str", "number", "number", "number", "number"],
        interactive=False,
    )

    run.click(
        process,
        [image_in, grid, style],
        [original_out, segmented_out, mosaic_out, result_md],
    )
    bench.click(benchmark, image_in, table)


if __name__ == "__main__":
    demo.launch(css=CSS)
