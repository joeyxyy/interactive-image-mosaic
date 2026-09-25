# Performance Report — Interactive Image Mosaic Generator

## Method

The app center-crops each uploaded image and resizes it to 512×512 pixels. It supports 16×16, 32×32, and 64×64 grids. Cell mean RGB values are computed using a vectorized NumPy reshape-and-mean operation. Each cell is matched to the closest predefined mini-image tile using squared Euclidean distance in RGB space.

The interface provides Natural, Colorized, and High Contrast styles and displays the original/preprocessed image, the segmented color grid, and the final mosaic. Similarity is measured with Mean Squared Error (MSE) and Structural Similarity Index (SSIM).

## Performance comparison

A repeated median benchmark on the project test image produced:

| Grid | Cells | Vectorized | Nested loop | Speedup |
| --- | ---: | ---: | ---: | ---: |
| 16×16 | 256 | 4.665 ms | 5.445 ms | 1.17× |
| 32×32 | 1,024 | 4.826 ms | 8.537 ms | 1.77× |
| 64×64 | 4,096 | 5.058 ms | 20.016 ms | 3.96× |

The loop implementation becomes progressively slower as the number of cells grows because Python executes work separately for each cell. The vectorized implementation keeps most of the work inside optimized NumPy operations, so its runtime grows much more slowly.

## Conclusion

Vectorization is especially useful at higher grid resolutions. The 64×64 test contains sixteen times as many cells as the 16×16 test, yet the vectorized cell-analysis time increased only slightly while the nested-loop implementation became several times slower. Exact timings vary by machine, so the live app includes a benchmark button that measures the current environment directly.
