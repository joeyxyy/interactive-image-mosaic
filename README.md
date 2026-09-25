---
title: Interactive Image Mosaic Generator
emoji: 🧩
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 6.28.0
app_file: app.py
pinned: false
---

# Interactive Image Mosaic Generator

A complete Gradio lab project that reconstructs an uploaded image using a predefined set of mini-image tiles.

## Features

- 16×16, 32×32, and 64×64 grid sizes
- Vectorized NumPy cell-color analysis
- Deterministic 24-tile mini-image set generated in memory
- Nearest-tile matching with squared RGB distance
- Natural, Colorized, and High Contrast mosaic styles
- Original, segmented-grid, and final-mosaic views
- MSE and SSIM similarity metrics
- Built-in vectorized vs. nested-loop performance benchmark

## Run locally

```bash
pip install -r requirements.txt
python app.py
```

## Project structure

- `app.py` — complete mosaic algorithm and Gradio interface
- `requirements.txt` — pinned Python dependencies
- `README.md` — Hugging Face Space metadata and documentation

The tile set is generated deterministically when the app starts, so the Space does not depend on external asset files.
