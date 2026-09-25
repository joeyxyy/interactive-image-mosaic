import time
from pathlib import Path
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
PALETTE = [
    (35,45,75),(55,90,140),(70,135,180),(80,175,200),
    (45,105,75),(70,150,95),(125,175,85),(185,195,95),
    (145,85,55),(195,120,65),(225,155,75),(235,195,110),
    (120,55,85),(175,70,115),(215,105,145),(235,150,175),
    (75,65,120),(115,90,170),(155,120,205),(195,165,225),
    (65,65,65),(105,105,105),(165,165,165),(225,225,225),
]


def _tile(base, pattern, i):
    img = Image.new("RGB", (TILE_PX, TILE_PX), base)
    d = ImageDraw.Draw(img)
    hi = tuple(min(255, c + 45) for c in base)
    lo = tuple(max(0, c - 35) for c in base)
    if pattern == "Geometric":
        m = i % 4
        if m == 0:
            d.rectangle((3,3,20,20), outline=hi, width=2)
            d.line((0,12,24,12), fill=lo, width=2); d.line((12,0,12,24), fill=lo, width=2)
        elif m == 1:
            d.polygon([(12,2),(21,21),(3,21)], outline=hi); d.line((2,2,21,21), fill=lo, width=2)
        elif m == 2:
            d.ellipse((3,3,20,20), outline=hi, width=2); d.rectangle((8,8,15,15), fill=lo)
        else:
            for x in range(-24, 48, 7): d.line((x,0,x+24,24), fill=hi, width=2)
    elif pattern == "Dots":
        for y in range(4, 24, 7):
            for x in range(4, 24, 7):
                r = 2 if (x + y + i) % 2 == 0 else 1
                d.ellipse((x-r,y-r,x+r,y+r), fill=hi if (x+y)%3 else lo)
    else:
        m = i % 3
        if m == 0:
            for y in range(2, 24, 5): d.line((0,y,24,y), fill=hi, width=2)
        elif m == 1:
            for x in range(2, 24, 5): d.line((x,0,x,24), fill=lo, width=2)
        else:
            for x in range(-24, 48, 6): d.line((x,0,x+24,24), fill=hi, width=2)
    return np.asarray(img, dtype=np.uint8)


def build_tile_sets():
    sets = {}
    for name in TILE_SET_NAMES:
        images = np.stack([_tile(c, name, i) for i, c in enumerate(PALETTE)])
        means = images.reshape(len(PALETTE), -1, 3).mean(axis=1).astype(np.float32)
        sets[name] = (images, means)
    return sets


TILE_SETS = build_tile_sets()


def ensure_example_images():
    out = Path(__file__).parent / "examples"
    out.mkdir(exist_ok=True)
    n = 512
    if not (out / "gradient.png").exists():
        y, x = np.mgrid[0:n, 0:n]
        arr = np.stack([x/(n-1)*255, y/(n-1)*255, (1-x/(n-1))*190+35], axis=2).astype(np.uint8)
        Image.fromarray(arr).save(out / "gradient.png")
    if not (out / "landscape.png").exists():
        img = Image.new("RGB", (n,n), (125,195,235)); d = ImageDraw.Draw(img)
        d.rectangle((0,300,n,n), fill=(66,132,76)); d.ellipse((380,45,455,120), fill=(250,215,90))
        d.polygon([(0,330),(145,150),(270,330)], fill=(72,105,128))
        d.polygon([(170,330),(335,125),(512,330)], fill=(58,88,116))
        d.polygon([(110,195),(145,150),(183,199)], fill=(238,241,245))
        d.polygon([(292,178),(335,125),(379,181)], fill=(238,241,245))
        d.rectangle((0,390,n,n), fill=(47,113,148)); img.save(out / "landscape.png")
    if not (out / "portrait.png").exists():
        img = Image.new("RGB", (n,n), (225,205,185)); d = ImageDraw.Draw(img)
        d.ellipse((138,72,374,330), fill=(192,145,112)); d.ellipse((186,150,218,176), fill=(40,45,50))
        d.ellipse((294,150,326,176), fill=(40,45,50)); d.arc((215,175,300,270),25,155,fill=(115,65,60),width=5)
        d.rectangle((135,320,377,512), fill=(65,85,125)); img.save(out / "portrait.png")


ensure_example_images()


def center_crop_resize(image: Image.Image, size: int = CANVAS) -> Image.Image:
    im = image.convert("RGB")
    w, h = im.size; side = min(w, h)
    left, top = (w-side)//2, (h-side)//2
    return im.crop((left, top, left+side, top+side)).resize((size,size), Image.Resampling.LANCZOS)


def vectorized_cell_means(arr: np.ndarray, grid: int) -> np.ndarray:
    h, w, _ = arr.shape; ch, cw = h//grid, w//grid
    a = arr[:grid*ch, :grid*cw]
    return a.reshape(grid, ch, grid, cw, 3).mean(axis=(1,3))


def loop_cell_means(arr: np.ndarray, grid: int) -> np.ndarray:
    h, w, _ = arr.shape; ch, cw = h//grid, w//grid
    out = np.zeros((grid,grid,3), dtype=np.float32)
    for r in range(grid):
        for c in range(grid):
            out[r,c] = arr[r*ch:(r+1)*ch, c*cw:(c+1)*cw].mean(axis=(0,1))
    return out


def classify_color_categories(means: np.ndarray) -> Dict[str, int]:
    names = ["Red","Orange","Yellow","Green","Cyan","Blue","Purple","Neutral"]
    counts = {k: 0 for k in names}
    for r,g,b in means.reshape(-1,3).astype(np.float32):
        mx, mn = max(r,g,b), min(r,g,b); d = mx-mn
        if mx < 55 or d < 18: counts["Neutral"] += 1; continue
        if mx == r: hue = (60*((g-b)/d)+360)%360
        elif mx == g: hue = 60*((b-r)/d+2)
        else: hue = 60*((r-g)/d+4)
        key = "Red" if hue < 20 or hue >= 345 else "Orange" if hue < 45 else "Yellow" if hue < 70 else "Green" if hue < 165 else "Cyan" if hue < 200 else "Blue" if hue < 255 else "Purple"
        counts[key] += 1
    return counts


def nearest_tile_indices(means: np.ndarray, tile_set_name: str) -> np.ndarray:
    _, tile_means = TILE_SETS[tile_set_name]
    flat = means.reshape(-1,3).astype(np.float32)
    dist = ((flat[:, None, :] - tile_means[None, :, :]) ** 2).sum(axis=2)
    return dist.argmin(axis=1).reshape(means.shape[:2])


def segmented_preview(means: np.ndarray) -> Image.Image:
    grid = means.shape[0]; cell = CANVAS // grid
    q = np.clip(means,0,255).astype(np.uint8)
    return Image.fromarray(np.repeat(np.repeat(q, cell, 0), cell, 1)[:CANVAS,:CANVAS])


def render_mosaic(indices: np.ndarray, means: np.ndarray, tile_set_name: str, style="Natural") -> Image.Image:
    images, _ = TILE_SETS[tile_set_name]
    grid = indices.shape[0]; cell = CANVAS // grid
    out = np.empty((CANVAS,CANVAS,3), dtype=np.uint8)
    for r in range(grid):
        for c in range(grid):
            tile = Image.fromarray(images[indices[r,c]]).resize((cell,cell), Image.Resampling.BILINEAR)
            a = np.asarray(tile, dtype=np.float32)
            if style == "Colorized":
                target = means[r,c]; source = a.reshape(-1,3).mean(axis=0)
                a = np.clip(a * (0.35 + 0.65*(target+1)/(source+1)), 0, 255)
            elif style == "High Contrast":
                a = np.clip((a-128)*1.25+128, 0, 255)
            out[r*cell:(r+1)*cell, c*cell:(c+1)*cell] = a.astype(np.uint8)
    return Image.fromarray(out)


def similarity_metrics(original: Image.Image, mosaic: Image.Image) -> Tuple[float, float]:
    a = np.asarray(original, dtype=np.uint8); b = np.asarray(mosaic, dtype=np.uint8)
    mse = float(np.mean((a.astype(np.float32)-b.astype(np.float32))**2))
    return mse, float(ssim(a,b,channel_axis=2,data_range=255))


def process(image, grid, tile_set_name, style):
    if image is None: raise gr.Error("Please upload an image first.")
    grid = int(grid); start = time.perf_counter()
    original = center_crop_resize(image); arr = np.asarray(original)
    means = vectorized_cell_means(arr, grid); segmented = segmented_preview(means)
    ids = nearest_tile_indices(means, tile_set_name); mosaic = render_mosaic(ids, means, tile_set_name, style)
    mse, score = similarity_metrics(original, mosaic); elapsed = time.perf_counter()-start
    cats = classify_color_categories(means); cat_text = ", ".join(f"{k}: {v}" for k,v in cats.items() if v)
    report = (f"### Results\n**Grid:** {grid} × {grid} ({grid*grid:,} cells)  \n**Tile set:** {tile_set_name}  \n"
              f"**Style:** {style}  \n**MSE:** {mse:,.2f} *(lower is better)*  \n**SSIM:** {score:.4f} *(higher is better)*  \n"
              f"**Processing time:** {elapsed:.4f} s  \n**Color categories:** {cat_text}")
    return original, segmented, mosaic, report


def benchmark(image):
    if image is None: raise gr.Error("Upload an image before benchmarking.")
    arr = np.asarray(center_crop_resize(image)); rows=[]; speedups=[]
    for grid in GRID_OPTIONS:
        vt=[]; lt=[]
        for _ in range(20):
            t=time.perf_counter(); vectorized_cell_means(arr,grid); vt.append(time.perf_counter()-t)
            t=time.perf_counter(); loop_cell_means(arr,grid); lt.append(time.perf_counter()-t)
        v,l=float(np.median(vt)),float(np.median(lt)); s=l/v; speedups.append(s)
        rows.append([f"{grid}×{grid}",grid*grid,round(v*1000,3),round(l*1000,3),round(s,2)])
    analysis = f"### Scaling analysis\nAt 64×64, vectorized NumPy is **{speedups[-1]:.2f}× faster** on this run because the reduction stays in optimized array code instead of Python cell-by-cell loops."
    return rows, analysis


def tile_sheet(name):
    images,_ = TILE_SETS[name]; scale=3
    sheet = Image.new("RGB", (8*TILE_PX*scale,3*TILE_PX*scale), "white")
    for i,a in enumerate(images):
        tile = Image.fromarray(a).resize((TILE_PX*scale,TILE_PX*scale), Image.Resampling.NEAREST)
        sheet.paste(tile, ((i%8)*TILE_PX*scale,(i//8)*TILE_PX*scale))
    return sheet


with gr.Blocks(title="Interactive Image Mosaic Generator") as demo:
    gr.Markdown("# 🧩 Interactive Image Mosaic Generator")
    gr.Markdown("Upload an image, segment it into a grid, and reconstruct it from representative mini-image tiles.")
    with gr.Row():
        image_in = gr.Image(type="pil", label="1. Upload Image")
        with gr.Column():
            grid = gr.Radio(GRID_OPTIONS, value=32, label="2. Grid Size")
            tile_set_name = gr.Dropdown(TILE_SET_NAMES, value="Geometric", label="3. Tile Set")
            style = gr.Dropdown(STYLE_OPTIONS, value="Colorized", label="4. Rendering Style")
            run = gr.Button("Generate Mosaic", variant="primary"); bench = gr.Button("Run Performance Benchmark")
    tile_preview = gr.Image(value=tile_sheet("Geometric"), label="Current Tile Set", interactive=False)
    tile_set_name.change(tile_sheet, tile_set_name, tile_preview)
    with gr.Row():
        original_out = gr.Image(label="Original / Preprocessed")
        segmented_out = gr.Image(label="Segmented Color Grid")
        mosaic_out = gr.Image(label="Final Tile Mosaic")
    result_md = gr.Markdown()
    gr.Markdown("## Performance: Vectorized NumPy vs. Nested Loops")
    table = gr.Dataframe(headers=["Grid","Cells","Vectorized (ms)","Loop (ms)","Speedup"], interactive=False)
    analysis_md = gr.Markdown()
    gr.Markdown("## Try an included example")
    gr.Examples(
        examples=[["examples/gradient.png",32,"Geometric","Colorized"],["examples/landscape.png",32,"Dots","Natural"],["examples/portrait.png",64,"Stripes","High Contrast"]],
        inputs=[image_in,grid,tile_set_name,style],
    )
    run.click(process,[image_in,grid,tile_set_name,style],[original_out,segmented_out,mosaic_out,result_md])
    bench.click(benchmark,image_in,[table,analysis_md])


if __name__ == "__main__":
    demo.launch()
