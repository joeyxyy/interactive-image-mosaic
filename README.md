---
title: Interactive Image Mosaic Generator
emoji: 🧩
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 6.28.0
app_file: app.py
pinned: false
license: mit
short_description: Interactive image mosaics with NumPy and Gradio.
---

# Interactive Image Mosaic Generator

An interactive image-processing project that segments an uploaded image into a grid and reconstructs it using representative mini-image tiles. The implementation emphasizes vectorized NumPy operations, measurable reconstruction quality, performance analysis, and an easy-to-use Gradio demo.

**Live Gradio demo:** https://huggingface.co/spaces/Joeyxyy/interactive-image-mosaic  
**GitHub repository:** https://github.com/joeyxyy/interactive-image-mosaic  
**Submission links/index:** [SUBMISSION_DELIVERABLES.md](SUBMISSION_DELIVERABLES.md)  
**Copy/paste links file:** [SUBMISSION_LINKS.txt](SUBMISSION_LINKS.txt)  
**Performance report:** [performance_report.pdf](performance_report.pdf)

![Sample output](screenshots/sample_output.png)

## Assignment Step 1 - Image Selection and Preprocessing

- Includes three test images in `examples/`: `gradient.png`, `landscape.png`, and `portrait.png`.
- Center-crops uploaded images to a square.
- Resizes every input to a fixed **512×512** working resolution for consistent grid processing.

## Assignment Step 2 - Image Grid and Thresholding

- Supports **16×16, 32×32, and 64×64** grid resolutions.
- Computes the mean RGB value of every cell.
- Uses a **vectorized NumPy reshape + reduction** for the main implementation.
- Includes a nested-loop implementation only for the required performance comparison.
- Classifies cells into **Red, Orange, Yellow, Green, Cyan, Blue, Purple, or Neutral** categories.

## Assignment Step 3 - Tile Mapping

- Provides **three predefined tile sets**: Geometric, Dots, and Stripes.
- Each tile set contains **24 mini-image tiles** spanning a broad color palette.
- Matches each cell to the closest tile by squared Euclidean distance between average RGB values.
- Supports **Natural, Colorized, and High Contrast** rendering styles.

![Tile sets](screenshots/tile_sets.png)

## Assignment Step 4 - Gradio Interface

The live Gradio interface lets users:

- Upload an image.
- Select grid size.
- Select tile set.
- Select rendering style.
- Generate the mosaic in real time.
- View the **original/preprocessed image**, **segmented color grid**, and **final mosaic** side by side.
- Run the performance benchmark directly from the interface.

**Live demo:** https://huggingface.co/spaces/Joeyxyy/interactive-image-mosaic

## Assignment Step 5 - Performance Metric

The app compares the preprocessed original image with the reconstructed mosaic using:

- **Mean Squared Error (MSE)** - lower is better.
- **Structural Similarity Index (SSIM)** - higher is better.

For the included landscape example at 32×32 using Geometric + Colorized, one reference run produced:

- **MSE:** 439.40
- **SSIM:** 0.2101
- **Processing time:** approximately 0.14 s

## Assignment Step 6 - Computational Performance

Reference benchmark on `examples/landscape.png`, using the median of 20 repetitions per method:

| Grid | Cells | Vectorized | Nested loop | Speedup |
| --- | ---: | ---: | ---: | ---: |
| 16×16 | 256 | 4.512 ms | 5.635 ms | 1.25× |
| 32×32 | 1,024 | 4.645 ms | 8.583 ms | 1.85× |
| 64×64 | 4,096 | 5.335 ms | 21.108 ms | 3.96× |

The nested-loop implementation scales more sharply because Python performs repeated slicing and averaging for every cell. The vectorized method keeps the reduction inside optimized NumPy operations, so the performance advantage increases as grid resolution grows.

The complete 1–2 page report is available as [performance_report.pdf](performance_report.pdf), with the Markdown source in [PERFORMANCE_REPORT.md](PERFORMANCE_REPORT.md).

## Guidelines, Creativity, and Presentation

The project goes beyond the minimum requirements with:

- Three distinct tile-set designs.
- Three rendering styles.
- Multiple selectable grid sizes.
- Side-by-side original, segmented, and final images.
- Built-in examples.
- A tile-set preview.
- Live MSE, SSIM, processing-time, and benchmark output.

## Submission Deliverables

| Deliverable | Location |
| --- | --- |
| Code submission | https://github.com/joeyxyy/interactive-image-mosaic |
| Live Gradio demo | https://huggingface.co/spaces/Joeyxyy/interactive-image-mosaic |
| 1–2 page performance report | [performance_report.pdf](performance_report.pdf) |
| Detailed report source | [PERFORMANCE_REPORT.md](PERFORMANCE_REPORT.md) |
| Submission link/index file | [SUBMISSION_DELIVERABLES.md](SUBMISSION_DELIVERABLES.md) |
| Copy/paste links file | [SUBMISSION_LINKS.txt](SUBMISSION_LINKS.txt) |
| Main application | [app.py](app.py) |
| Test images | [examples/](examples/) |
| Automated tests | [tests/](tests/) |

## Run Locally

The assignment recommends using Gradio's development command while editing:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
gradio app.py
```

The application can also be started normally with:

```bash
python app.py
```

## Run the Tests

```bash
pip install -r requirements-dev.txt
pytest -q
```

The final test suite contains **9 automated tests** covering preprocessing, all three grid sizes, vectorized/loop equivalence, tile mapping, tile sets and styles, MSE/SSIM validity, color-category accounting, benchmark structure, tile-sheet generation, example generation, and end-to-end processing.

## Project Structure

```text
interactive-image-mosaic/
├── app.py
├── README.md
├── SUBMISSION_DELIVERABLES.md
├── SUBMISSION_LINKS.txt
├── PERFORMANCE_REPORT.md
├── performance_report.pdf
├── requirements.txt
├── requirements-dev.txt
├── LICENSE
├── examples/
│   ├── gradient.png
│   ├── landscape.png
│   └── portrait.png
├── screenshots/
│   ├── sample_output.png
│   └── tile_sets.png
├── scripts/
│   └── generate_assets.py
├── tests/
│   ├── conftest.py
│   └── test_app.py
└── .github/workflows/
    ├── generate-assets.yml
    ├── sync-to-huggingface.yml
    └── tests.yml
```

## Requirement Coverage Checklist

| Assignment requirement | Implementation |
| --- | --- |
| Test images and preprocessing | Three examples; center-crop + 512×512 resize |
| Fixed-grid segmentation | 16×16, 32×32, and 64×64 |
| Vectorized grid operations | NumPy reshape + mean across all cells |
| Color classification | Eight readable color categories |
| Predefined tile set | Three deterministic 24-tile sets |
| Cell-to-tile mapping | Nearest average RGB |
| Gradio interface | Upload + grid + tile-set + style controls |
| Original / segmented / mosaic views | All three shown side by side |
| Similarity metric | MSE and SSIM |
| Processing-time analysis | 16×16, 32×32, and 64×64 |
| Scaling discussion | Included in report |
| Vectorized vs. loop comparison | Both implementations benchmarked |
| Creativity | Three tile sets + three rendering styles |
| Live demo | Permanent Hugging Face Space |
| Report | PDF + Markdown |
| Reproducibility | Pinned dependencies + tests + CI + deployment workflow |

## Assignment-Provided Resources

- Gradio: https://gradio.app/
- Gradio sharing guide: https://www.gradio.app/guides/sharing-your-app
- Hugging Face Gradio Spaces guide: https://huggingface.co/blog/gradio-spaces
- Color quantization reference: https://en.wikipedia.org/wiki/Color_quantization

