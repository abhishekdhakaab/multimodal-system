"""
Loads ModelNet10 (.off mesh files), and derives both modalities for a given
mesh from the same real 3D geometry:
  - "lidar": points sampled directly from the mesh surface
  - "camera": a 2D projection of that same geometry from a random viewpoint,
    rasterized into a small grayscale image

Both views come from one real object, not two unrelated generators.
"""

import math
import os

import numpy as np

CLASSES = [
    "bathtub",
    "bed",
    "chair",
    "desk",
    "dresser",
    "monitor",
    "night_stand",
    "sofa",
    "table",
    "toilet",
]

RAW_DIR = os.path.join(os.path.dirname(__file__), "raw", "ModelNet10")
IMAGE_SIZE = 40  # must stay divisible by vision_encoder.PATCH_SIZE
NUM_POINTS = 256  # sparse, for the "lidar" point cloud -- real lidar is much lower-resolution than a camera
NUM_RENDER_POINTS = 6000  # dense, used only to render the "camera" image, not exposed to the model as points


def parse_off(path):
    """Parses a .off mesh file into (vertices [V,3], faces: list of int lists).

    Handles the well-known ModelNet .off quirk where some files write the
    header and counts on the same line (e.g. "OFF3451 2958 0" instead of
    "OFF\\n3451 2958 0"), which trips up a naive parser.
    """
    # a handful of ModelNet10 files have stray non-UTF-8 bytes; replace them
    # rather than failing, since they never land inside the numeric data we need
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        text = f.read()

    lines = [line.strip() for line in text.splitlines() if line.strip() and not line.startswith("#")]

    first = lines[0]
    if first.startswith("OFF") and first != "OFF":
        # counts are glued onto the OFF header line itself
        counts_line = first[3:]
        rest_lines = lines[1:]
    else:
        counts_line = lines[1]
        rest_lines = lines[2:]

    n_verts, n_faces, _ = (int(x) for x in counts_line.split()[:3])

    vertices = np.array(
        [[float(x) for x in rest_lines[i].split()[:3]] for i in range(n_verts)], dtype=np.float32
    )

    faces = []
    for i in range(n_faces):
        parts = rest_lines[n_verts + i].split()
        n_in_face = int(parts[0])
        faces.append([int(x) for x in parts[1 : 1 + n_in_face]])

    return vertices, faces


def _triangulate(faces):
    """Fan-triangulates any non-triangular faces (OFF allows arbitrary polygons)."""
    tris = []
    for face in faces:
        for i in range(1, len(face) - 1):
            tris.append((face[0], face[i], face[i + 1]))
    return np.array(tris, dtype=np.int64)


def sample_points_on_mesh(vertices, faces, n_points, rng):
    """Area-weighted sampling of points on the mesh surface (standard point-cloud
    sampling technique, the same approach PointNet's own preprocessing uses)."""
    tris = _triangulate(faces)
    v0, v1, v2 = vertices[tris[:, 0]], vertices[tris[:, 1]], vertices[tris[:, 2]]

    cross = np.cross(v1 - v0, v2 - v0)
    areas = 0.5 * np.linalg.norm(cross, axis=1)
    areas = np.clip(areas, 1e-12, None)
    probs = areas / areas.sum()

    chosen = rng.choice(len(tris), size=n_points, p=probs)
    r1 = rng.random(n_points)
    r2 = rng.random(n_points)
    sqrt_r1 = np.sqrt(r1)

    # standard uniform-on-triangle sampling formula
    a = 1 - sqrt_r1
    b = sqrt_r1 * (1 - r2)
    c = sqrt_r1 * r2

    points = (
        a[:, None] * v0[chosen] + b[:, None] * v1[chosen] + c[:, None] * v2[chosen]
    )
    return points.astype(np.float32)


def normalize_points(points):
    """Centers on the origin and scales to fit in a unit sphere."""
    centroid = points.mean(axis=0)
    points = points - centroid
    scale = np.max(np.linalg.norm(points, axis=1))
    return points / (scale + 1e-8)


def project_to_image(points, image_size, rng):
    """Random-viewpoint orthographic projection of a point cloud, rasterized
    into a grayscale image. This is the "camera" modality — a real 2D view
    of the same real 3D geometry as the point cloud, not a separate drawing."""
    theta = rng.uniform(0, 2 * math.pi)  # azimuth
    phi = rng.uniform(math.pi / 6, math.pi - math.pi / 6)  # elevation, avoid degenerate top/bottom views

    # viewing direction and an orthonormal basis for the image plane
    view_dir = np.array(
        [math.sin(phi) * math.cos(theta), math.sin(phi) * math.sin(theta), math.cos(phi)]
    )
    up = np.array([0.0, 0.0, 1.0])
    right = np.cross(up, view_dir)
    right /= np.linalg.norm(right) + 1e-8
    cam_up = np.cross(view_dir, right)

    u = points @ right
    v = points @ cam_up

    # map [-1, 1] range to pixel coords, with a small margin
    margin = 0.9
    px = ((u / margin + 1) / 2 * (image_size - 1)).astype(np.int32)
    py = ((v / margin + 1) / 2 * (image_size - 1)).astype(np.int32)
    px = np.clip(px, 0, image_size - 1)
    py = np.clip(py, 0, image_size - 1)

    image = np.zeros((image_size, image_size), dtype=np.float32)
    # splat each point onto a small 3x3 blob, not a single pixel, so a dense
    # point sample reads as a continuous silhouette instead of sparse noise
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            yy = np.clip(py + dy, 0, image_size - 1)
            xx = np.clip(px + dx, 0, image_size - 1)
            np.add.at(image, (yy, xx), 1.0)

    if image.max() > 0:
        image = image / image.max()
    return image


def list_examples(split):
    """split: 'train' or 'test'. Returns list of (off_path, class_name)."""
    examples = []
    for cls in CLASSES:
        split_dir = os.path.join(RAW_DIR, cls, split)
        for fname in sorted(os.listdir(split_dir)):
            if fname.endswith(".off"):
                examples.append((os.path.join(split_dir, fname), cls))
    return examples


def build_example(off_path, cls, rng):
    """Returns (image [IMAGE_SIZE,IMAGE_SIZE] float32, points [NUM_POINTS,3] float32, label int).

    Samples one dense point set off the real mesh surface, normalizes it once,
    renders the "camera" image from the full dense set (so it reads as a
    continuous silhouette, like a real camera sees a continuous surface), and
    uses a random sparse subset of that same normalized set as the "lidar"
    point cloud (real lidar is much lower-resolution than a camera) -- both
    modalities come from the same real geometry in the same reference frame.
    """
    vertices, faces = parse_off(off_path)
    dense_points = sample_points_on_mesh(vertices, faces, NUM_RENDER_POINTS, rng)
    dense_points = normalize_points(dense_points)

    image = project_to_image(dense_points, IMAGE_SIZE, rng)

    sparse_idx = rng.choice(NUM_RENDER_POINTS, size=NUM_POINTS, replace=False)
    lidar_points = dense_points[sparse_idx] + rng.normal(0, 0.01, (NUM_POINTS, 3)).astype(np.float32)

    label = CLASSES.index(cls)
    return image, lidar_points, label
