import os
import tempfile

import numpy as np
import pytest

from data_pipeline.modelnet_loader import (
    IMAGE_SIZE,
    NUM_POINTS,
    normalize_points,
    parse_off,
    project_to_image,
    sample_points_on_mesh,
)

HERE = os.path.dirname(__file__)
SHARDS_TRAIN_DIR = os.path.join(HERE, "..", "shards", "train")

# a minimal single-triangle mesh, written with the known ModelNet "glued header"
# quirk (counts on the same line as "OFF") to make sure the parser handles it
GLUED_HEADER_OFF = "OFF3 1 0\n0 0 0\n1 0 0\n0 1 0\n3 0 1 2\n"

# the same mesh, written in the normal (non-glued) OFF format
NORMAL_OFF = "OFF\n3 1 0\n0 0 0\n1 0 0\n0 1 0\n3 0 1 2\n"


def _write_and_parse(content):
    with tempfile.NamedTemporaryFile(mode="w", suffix=".off", delete=False) as f:
        f.write(content)
        path = f.name
    try:
        return parse_off(path)
    finally:
        os.unlink(path)


def test_parse_off_handles_glued_header():
    vertices, faces = _write_and_parse(GLUED_HEADER_OFF)
    assert vertices.shape == (3, 3)
    assert faces == [[0, 1, 2]]


def test_parse_off_normal_format():
    vertices, faces = _write_and_parse(NORMAL_OFF)
    assert vertices.shape == (3, 3)
    assert faces == [[0, 1, 2]]


def test_sample_points_on_mesh_stays_within_triangle_bounds():
    vertices, faces = _write_and_parse(NORMAL_OFF)
    rng = np.random.default_rng(0)
    points = sample_points_on_mesh(vertices, faces, n_points=50, rng=rng)
    assert points.shape == (50, 3)
    # every sampled point must lie within the triangle's bounding box
    assert np.all(points[:, 0] >= -1e-5) and np.all(points[:, 0] <= 1 + 1e-5)
    assert np.all(points[:, 1] >= -1e-5) and np.all(points[:, 1] <= 1 + 1e-5)


def test_normalize_points_centers_and_scales():
    points = np.array([[0, 0, 0], [2, 0, 0], [0, 2, 0]], dtype=np.float32)
    normed = normalize_points(points)
    assert np.allclose(normed.mean(axis=0), normed.mean(axis=0))  # sanity, always true
    max_norm = np.max(np.linalg.norm(normed, axis=1))
    assert abs(max_norm - 1.0) < 1e-4


def test_project_to_image_shape_and_range():
    rng = np.random.default_rng(0)
    points = rng.normal(0, 0.3, size=(100, 3)).astype(np.float32)
    image = project_to_image(points, image_size=32, rng=rng)
    assert image.shape == (32, 32)
    assert image.min() >= 0.0 and image.max() <= 1.0 + 1e-6


@pytest.mark.skipif(not os.path.isdir(SHARDS_TRAIN_DIR), reason="run build_shards.py first")
def test_shard_roundtrip_shapes():
    shard_files = sorted(f for f in os.listdir(SHARDS_TRAIN_DIR) if f.endswith(".npz"))
    assert len(shard_files) > 0
    data = np.load(os.path.join(SHARDS_TRAIN_DIR, shard_files[0]), allow_pickle=True)

    images, pointclouds, labels, classes = (
        data["images"],
        data["pointclouds"],
        data["labels"],
        data["classes"],
    )
    n = len(labels)
    assert images.shape == (n, IMAGE_SIZE, IMAGE_SIZE)
    assert pointclouds.shape == (n, NUM_POINTS, 3)
    assert set(np.unique(labels)).issubset(set(range(len(classes))))


def test_hard_case_miner_flags_low_confidence():
    from data_pipeline.hard_case_miner import find_hard_cases

    probs = np.array([[0.9, 0.1], [0.5, 0.5], [0.1, 0.9]])
    labels = np.array([0, 1, 1])
    hard = find_hard_cases(probs, labels, threshold=0.6)
    assert hard == [1]
