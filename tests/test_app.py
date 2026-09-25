import numpy as np
from PIL import Image

import app


def synthetic_image(size=512):
    y, x = np.mgrid[0:size, 0:size]
    arr = np.stack([
        (x / (size - 1) * 255),
        (y / (size - 1) * 255),
        ((x + y) / (2 * (size - 1)) * 255),
    ], axis=2).astype(np.uint8)
    return Image.fromarray(arr)


def test_preprocess_output_shape():
    out = app.center_crop_resize(synthetic_image(300))
    assert out.size == (512, 512)


def test_vectorized_matches_loop_all_grids():
    arr = np.asarray(synthetic_image())
    for grid in app.GRID_OPTIONS:
        vectorized = app.vectorized_cell_means(arr, grid)
        loop = app.loop_cell_means(arr, grid)
        assert vectorized.shape == (grid, grid, 3)
        assert np.allclose(vectorized, loop, atol=1e-5)


def test_tile_mapping_and_render_all_sets():
    image = synthetic_image()
    arr = np.asarray(image)
    means = app.vectorized_cell_means(arr, 32)
    for tile_set_name in app.TILE_SET_NAMES:
        ids = app.nearest_tile_indices(means, tile_set_name)
        assert ids.shape == (32, 32)
        assert ids.min() >= 0
        assert ids.max() < len(app.PALETTE)
        for style in app.STYLE_OPTIONS:
            mosaic = app.render_mosaic(ids, means, tile_set_name, style)
            assert mosaic.size == (512, 512)


def test_metrics_are_valid():
    image = synthetic_image()
    means = app.vectorized_cell_means(np.asarray(image), 32)
    ids = app.nearest_tile_indices(means, "Geometric")
    mosaic = app.render_mosaic(ids, means, "Geometric", "Colorized")
    mse, score = app.similarity_metrics(image, mosaic)
    assert mse >= 0
    assert -1 <= score <= 1


def test_color_categories_sum_to_cells():
    arr = np.asarray(synthetic_image())
    means = app.vectorized_cell_means(arr, 16)
    counts = app.classify_color_categories(means)
    assert sum(counts.values()) == 16 * 16


def test_process_all_grid_sizes():
    image = synthetic_image()
    for grid in app.GRID_OPTIONS:
        original, segmented, mosaic, report = app.process(image, grid, "Dots", "Natural")
        assert original.size == (512, 512)
        assert segmented.size == (512, 512)
        assert mosaic.size == (512, 512)
        assert "MSE" in report and "SSIM" in report


def test_examples_are_generated():
    app.ensure_example_images()
    from pathlib import Path
    root = Path(app.__file__).parent / "examples"
    for name in ["gradient.png", "landscape.png", "portrait.png"]:
        assert (root / name).exists()
