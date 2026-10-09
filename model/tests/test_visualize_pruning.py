import numpy as np

from model.visualize_pruning import IMAGE_SIZE, SCALE, render


def test_render_produces_correctly_sized_image():
    image = np.random.rand(IMAGE_SIZE, IMAGE_SIZE).astype(np.float32)
    kept_mask = np.zeros(100, dtype=bool)
    kept_mask[:50] = True
    out = render(image, kept_mask, "test_label")
    assert out.size == (IMAGE_SIZE * SCALE, IMAGE_SIZE * SCALE)
