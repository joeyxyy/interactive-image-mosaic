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
short_description: Reconstruct images as interactive tile mosaics with NumPy and Gradio.
---

# Interactive Image Mosaic Generator

An interactive image-processing project that segments an uploaded image into a grid and reconstructs it using representative mini-image tiles. The implementation emphasizes vectorized NumPy operations, measurable reconstruction quality, performance analysis, and an easy-to-use Gradio demo.


## Features

- Upload any image and center-crop/resize it to a fixed **512×512** working resolution.
- Choose **16×16, 32×32, or 64×64** grid resolution.
- Choose among **three predefined tile sets**: Geometric, Dots, and Stripes.
- Choose **Natural, Colorized, or High Contrast** rendering styles.
- Compute cell-average RGB values with a **vectorized NumPy** implementation.
- Classify cells into readable color categories: Red, Orange, Yellow, Green, Cyan, Blue, Purple, and Neutral.
- Map each grid cell to its closest tile using squared Euclidean distance in RGB space.
- Display the **original/preprocessed image**, **segmented color grid**, and **final mosaic** side by side.
- Evaluate reconstruction quality with **MSE** and **SSIM**.
- Compare **vectorized NumPy vs. nested-loop** implementations at 16×16, 32×32, and 64×64.
- Generates three deterministic example images automatically and includes automated tests.

## Tile sets

Each tile set contains 24 mini-images spanning a broad color palette. The tile art is generated deterministically, so the application is fully self-contained and does not depend on external assets or network calls.

## How it works

1. **Preprocess** - center-crop the uploaded image to square and resize it to 512×512.
2. **Segment** - divide the image into a fixed grid of 16×16, 32×32, or 64×64 cells.
3. **Analyze** - compute the mean RGB value of every cell using a single vectorized reshape/reduction.
4. **Classify** - assign each cell to a descriptive color category for analysis.
5. **Match** - compare every cell mean to the precomputed mean color of every tile and choose the closest match.
6. **Reconstruct** - resize and place the selected mini-image into each corresponding grid cell.
7. **Evaluate** - compute MSE and SSIM between the preprocessed input and reconstructed mosaic.
8. **Benchmark** - time both the vectorized and nested-loop cell-analysis implementations and report the scaling difference.

## Performance

Reference benchmark on the included `examples/landscape.png` image:

| Grid | Cells | Vectorized | Nested loop | Speedup |
| --- | ---: | ---: | ---: | ---: |
| 16×16 | 256 | 4.344 ms | 5.342 ms | 1.23× |
| 32×32 | 1,024 | 4.562 ms | 8.442 ms | 1.85× |
| 64×64 | 4,096 | 5.314 ms | 20.346 ms | 3.83× |

The loop implementation scales more sharply because Python processes every cell separately. The vectorized method keeps the reduction inside optimized NumPy operations, so the performance advantage becomes larger as grid resolution increases.

For the 32×32 Geometric/Colorized example, one reference run produced **MSE 439.40** and **SSIM 0.2101**. Exact timings and quality values vary by image and machine, so the live app calculates them on demand.

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
python app.py
```

Then open the local Gradio URL printed in the terminal.

## Run the tests

```bash
pip install -r requirements-dev.txt
pytest -q
```

The test suite checks preprocessing, all three grid sizes, vectorized/loop equivalence, all tile sets and styles, valid MSE/SSIM output, color-category accounting, and complete end-to-end processing.

## Project structure

```text
interactive-image-mosaic/
├── app.py
├── README.md
├── PERFORMANCE_REPORT.md
├── requirements.txt
├── requirements-dev.txt
├── LICENSE
├── examples/                 # generated automatically at first launch
├── tests/
│   └── test_app.py
└── .github/workflows/
    ├── sync-to-huggingface.yml
    └── tests.yml
```

## Assignment coverage

This project includes image preprocessing, fixed-grid segmentation, vectorized cell analysis, color categorization, a predefined tile set, cell-to-tile matching, Gradio controls for grid size and tile set, original/segmented/final presentation, MSE and SSIM evaluation, performance timing for 16×16/32×32/64×64, and a direct vectorized-vs-loop comparison.

A formatted 1–2 page performance report is provided with the submission, and the same analysis is documented in `PERFORMANCE_REPORT.md`.
