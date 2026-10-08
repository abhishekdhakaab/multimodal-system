"""
Generates a small synthetic multimodal dataset.

For each example we create two views of the same shape:
  - a 2D image (the "camera" modality) with the shape drawn at a random
    position/size/rotation, plus a bit of pixel noise
  - a 3D point cloud (the "lidar" modality) sampled from that same shape's
    outline, with random rotation and a bit of point jitter

The label is the shape class. The task is intentionally simple: a model
has to combine both modalities to classify the shape reliably, which is
the whole point (either modality alone is still mostly informative here,
on purpose — this is a learning toy, not a benchmark).
"""

import math
import random

import numpy as np
from PIL import Image, ImageDraw

CLASSES = ["circle", "square", "triangle", "star"]
IMAGE_SIZE = 32
NUM_POINTS = 64


def _draw_shape(cls, cx, cy, size, rotation):
    img = Image.new("L", (IMAGE_SIZE, IMAGE_SIZE), color=0)
    draw = ImageDraw.Draw(img)

    if cls == "circle":
        draw.ellipse([cx - size, cy - size, cx + size, cy + size], fill=255)
    elif cls == "square":
        pts = _polygon_points(cx, cy, size, 4, rotation + math.pi / 4)
        draw.polygon(pts, fill=255)
    elif cls == "triangle":
        pts = _polygon_points(cx, cy, size, 3, rotation)
        draw.polygon(pts, fill=255)
    elif cls == "star":
        pts = _star_points(cx, cy, size, rotation)
        draw.polygon(pts, fill=255)
    else:
        raise ValueError(f"unknown class {cls}")

    return np.array(img, dtype=np.float32) / 255.0


def _polygon_points(cx, cy, size, n_sides, rotation):
    pts = []
    for i in range(n_sides):
        angle = rotation + 2 * math.pi * i / n_sides
        pts.append((cx + size * math.cos(angle), cy + size * math.sin(angle)))
    return pts


def _star_points(cx, cy, size, rotation):
    pts = []
    for i in range(10):
        angle = rotation + 2 * math.pi * i / 10
        r = size if i % 2 == 0 else size * 0.45
        pts.append((cx + r * math.cos(angle), cy + r * math.sin(angle)))
    return pts


def _shape_outline_2d(cls, size, rotation, n_points):
    """Points on the shape's outline in a local 2D frame, used to build the 3D point cloud."""
    if cls == "circle":
        angles = np.linspace(0, 2 * math.pi, n_points, endpoint=False)
        xs = size * np.cos(angles)
        ys = size * np.sin(angles)
    elif cls == "square":
        corners = np.array(_polygon_points(0, 0, size, 4, rotation + math.pi / 4))
        xs, ys = _sample_along_polygon(corners, n_points)
    elif cls == "triangle":
        corners = np.array(_polygon_points(0, 0, size, 3, rotation))
        xs, ys = _sample_along_polygon(corners, n_points)
    elif cls == "star":
        corners = np.array(_star_points(0, 0, size, rotation))
        xs, ys = _sample_along_polygon(corners, n_points)
    else:
        raise ValueError(f"unknown class {cls}")
    return xs, ys


def _sample_along_polygon(corners, n_points):
    n = len(corners)
    seg_lengths = [np.linalg.norm(corners[(i + 1) % n] - corners[i]) for i in range(n)]
    total = sum(seg_lengths)
    samples = np.linspace(0, total, n_points, endpoint=False)
    xs, ys = [], []
    for s in samples:
        acc = 0.0
        for i in range(n):
            if acc + seg_lengths[i] >= s:
                t = (s - acc) / seg_lengths[i] if seg_lengths[i] > 0 else 0.0
                p = corners[i] + t * (corners[(i + 1) % n] - corners[i])
                xs.append(p[0])
                ys.append(p[1])
                break
            acc += seg_lengths[i]
    return np.array(xs), np.array(ys)


def generate_example(cls, rng):
    """Returns (image [32,32] float32, pointcloud [NUM_POINTS,3] float32, label str)."""
    size = rng.uniform(8, 13)
    rotation = rng.uniform(0, 2 * math.pi)
    cx = rng.uniform(size + 2, IMAGE_SIZE - size - 2)
    cy = rng.uniform(size + 2, IMAGE_SIZE - size - 2)

    image = _draw_shape(cls, cx, cy, size, rotation)
    noise = rng.normal(0, 0.03, image.shape).astype(np.float32)
    image = np.clip(image + noise, 0.0, 1.0)

    xs, ys = _shape_outline_2d(cls, size, rotation, NUM_POINTS)
    zs = rng.normal(0, 0.5, size=xs.shape)  # small random height, point cloud is 3D
    points = np.stack([xs, ys, zs], axis=1).astype(np.float32)
    points += rng.normal(0, 0.15, points.shape).astype(np.float32)  # lidar-style jitter

    # random rigid rotation around z-axis so the point cloud isn't trivially aligned with the image
    theta = rng.uniform(0, 2 * math.pi)
    rot = np.array(
        [[math.cos(theta), -math.sin(theta), 0], [math.sin(theta), math.cos(theta), 0], [0, 0, 1]],
        dtype=np.float32,
    )
    points = points @ rot.T

    return image, points, cls


def generate_dataset(n_per_class=750, seed=0):
    rng = np.random.default_rng(seed)
    images, pointclouds, labels = [], [], []
    for cls in CLASSES:
        for _ in range(n_per_class):
            img, pts, _ = generate_example(cls, rng)
            images.append(img)
            pointclouds.append(pts)
            labels.append(CLASSES.index(cls))
    images = np.stack(images)
    pointclouds = np.stack(pointclouds)
    labels = np.array(labels, dtype=np.int64)

    # shuffle so classes are interleaved, not blocked
    perm = rng.permutation(len(labels))
    return images[perm], pointclouds[perm], labels[perm]


if __name__ == "__main__":
    import os

    out_dir = os.path.join(os.path.dirname(__file__), "generated")
    os.makedirs(out_dir, exist_ok=True)

    images, pointclouds, labels = generate_dataset()
    np.savez(
        os.path.join(out_dir, "dataset.npz"),
        images=images,
        pointclouds=pointclouds,
        labels=labels,
        classes=np.array(CLASSES),
    )
    print(f"generated {len(labels)} examples -> {out_dir}/dataset.npz")
    print(f"images shape: {images.shape}, pointclouds shape: {pointclouds.shape}")
