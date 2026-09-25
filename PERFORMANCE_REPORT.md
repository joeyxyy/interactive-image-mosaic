# Interactive Image Mosaic Generator - Performance Report

## 1. Approach

The application first center-crops the uploaded image and resizes it to 512×512 pixels. The user selects a 16×16, 32×32, or 64×64 grid. Mean RGB values for all cells are computed using a vectorized NumPy reshape-and-reduction. For comparison, a nested-loop implementation performs the same calculation one cell at a time.

Three deterministic tile sets are provided: Geometric, Dots, and Stripes. Each contains 24 mini-images covering a broad RGB palette. A cell is matched to the tile whose precomputed mean RGB value has the smallest squared Euclidean distance from the cell mean. The final mosaic is then rendered in Natural, Colorized, or High Contrast style.

The interface presents the original/preprocessed image, segmented color grid, and final mosaic. Reconstruction quality is measured with Mean Squared Error (MSE) and Structural Similarity Index (SSIM). Cells are also classified into Red, Orange, Yellow, Green, Cyan, Blue, Purple, or Neutral categories.

## 2. Performance results

Reference benchmark on `examples/landscape.png`, using the median of 20 repetitions per method:

| Grid | Cells | Vectorized (ms) | Nested loop (ms) | Speedup |
| --- | ---: | ---: | ---: | ---: |
| 16×16 | 256 | 4.512 | 5.635 | 1.25× |
| 32×32 | 1,024 | 4.645 | 8.583 | 1.85× |
| 64×64 | 4,096 | 5.335 | 21.108 | 3.96× |

The nested-loop implementation becomes progressively slower as the number of cells grows because Python performs repeated slicing and averaging for each cell individually. The vectorized implementation reshapes the image into a structured 5-D view and calculates all cell means in one NumPy reduction. Most work therefore remains inside compiled array operations instead of the Python interpreter.

The scaling difference is clearest at 64×64. This grid contains 4,096 cells - sixteen times as many as 16×16 - but the vectorized analysis increases only slightly, from about 4.51 ms to 5.34 ms in this run. The loop method rises from about 5.64 ms to 21.11 ms, producing a measured 3.96× speedup for vectorization.

## 3. Reconstruction quality

For the included landscape example at a 32×32 grid using the Geometric tile set and Colorized style, one reference run produced:

- **MSE:** 439.40
- **SSIM:** 0.2101
- **Processing time:** approximately 0.14 s

MSE measures average squared pixel error, so lower values indicate closer pixel-level reconstruction. SSIM measures structural similarity and is bounded near 1.0 for highly similar images. A tile mosaic intentionally replaces local image detail with mini-image texture, so the metrics are useful mainly for comparing settings rather than expecting near-perfect pixel identity.

## 4. Verification

The final automated suite contains **9 tests**. It verifies preprocessing, all three grid sizes, vectorized/loop equivalence, tile mapping, all tile sets and rendering styles, valid MSE/SSIM output, color-category accounting, benchmark structure, tile-sheet generation, example generation, and complete end-to-end processing. GitHub Actions independently runs the tests on every push, and the deployment workflow publishes the same source to the Hugging Face Space.

## 5. Conclusion

Vectorized NumPy is the better approach for cell analysis, especially as grid size increases. The project demonstrates the full pipeline from image preprocessing and segmentation through tile matching, reconstruction, similarity evaluation, interactive presentation, reproducible benchmarking, automated testing, and live deployment. The built-in benchmark allows users to reproduce the timing comparison on their own machine or Hugging Face Space environment.
