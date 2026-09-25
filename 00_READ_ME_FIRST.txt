INTERACTIVE IMAGE MOSAIC GENERATOR
SUBMISSION - READ THIS FIRST

REQUIRED DELIVERABLES

1. CODE SUBMISSION
GitHub Repository:
https://github.com/joeyxyy/interactive-image-mosaic

2. LIVE GRADIO DEMO
Hugging Face Space:
https://huggingface.co/spaces/Joeyxyy/interactive-image-mosaic

3. PERFORMANCE REPORT
File:
performance_report.pdf

GitHub copy:
https://github.com/joeyxyy/interactive-image-mosaic/blob/main/performance_report.pdf

ASSIGNMENT STEP MAP

STEP 1 - Image Selection and Preprocessing
Files: examples/ and app.py
Includes test images, center-crop, and fixed 512x512 resize.

STEP 2 - Image Grid and Thresholding
File: app.py
Includes 16x16, 32x32, 64x64 grids, vectorized NumPy analysis,
nested-loop comparison, and color-category classification.

STEP 3 - Tile Mapping
File: app.py
Includes three predefined tile sets, 24 tiles per set, nearest-color
matching, and tile-based reconstruction.

STEP 4 - Gradio Interface
File: app.py
Live demo: https://huggingface.co/spaces/Joeyxyy/interactive-image-mosaic
Includes image upload, grid-size control, tile-set control, style control,
and original/segmented/final outputs.

STEP 5 - Performance Metric
Files: app.py, PERFORMANCE_REPORT.md, performance_report.pdf
Includes MSE and SSIM.

STEP 6 - Computational Performance
Files: app.py, PERFORMANCE_REPORT.md, performance_report.pdf
Includes 16x16/32x32/64x64 timing, scaling analysis, and vectorized-vs-loop comparison.

OTHER USEFUL FILES

README.md                 Full project explanation and rubric coverage
SUBMISSION_DELIVERABLES.md Detailed deliverable and step index
SUBMISSION_LINKS.txt       Copy/paste-ready links
screenshots/               Sample output and tile-set preview
tests/                     Automated verification
requirements.txt           Pinned runtime dependencies
.github/workflows/         CI and Hugging Face deployment

CANVAS SUBMISSION

Paste:
- GitHub repository link
- Live Hugging Face / Gradio demo link

Upload:
- performance_report.pdf

Upload the ZIP only if Canvas also asks for a code/file upload.

No recorded video is listed as a required submission deliverable in the assignment instructions.
