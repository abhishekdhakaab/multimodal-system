"""
Renders a real validation image with its kept/pruned patches overlaid,
so the token-pruning result (docs/token_pruning_notes.md) is a picture,
not just a table of numbers. Kept patches (the top-k by content) are
left visible; pruned patches are dimmed, with a grid line drawn between
every patch so the boundary is legible.

Usage: python -m model.visualize_pruning
"""

import glob
import os

import numpy as np
from PIL import Image, ImageDraw

from model.vision_encoder import IMAGE_SIZE, PATCH_SIZE, NUM_PATCHES, _select_top_k_patches, _patchify
import jax.numpy as jnp

HERE = os.path.dirname(__file__)
SHARDS_DIR = os.path.join(HERE, "..", "data_pipeline", "shards")
OUT_DIR = os.path.join(HERE, "..", "docs", "images")
SCALE = 8  # upscale factor so the output is actually legible


def load_examples(n=6):
    paths = sorted(glob.glob(os.path.join(SHARDS_DIR, "val", "shard_*.npz")))
    d = np.load(paths[0], allow_pickle=True)
    return d["images"][:n], d["labels"][:n], d["classes"]


def render(image, kept_mask, label_name):
    """image: [IMAGE_SIZE, IMAGE_SIZE], kept_mask: [NUM_PATCHES] bool -> PIL.Image"""
    img = (image * 255).astype(np.uint8)
    base = Image.fromarray(img).convert("RGB").resize(
        (IMAGE_SIZE * SCALE, IMAGE_SIZE * SCALE), Image.NEAREST
    )
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    n_per_side = IMAGE_SIZE // PATCH_SIZE
    for idx in range(NUM_PATCHES):
        row, col = idx // n_per_side, idx % n_per_side
        x0, y0 = col * PATCH_SIZE * SCALE, row * PATCH_SIZE * SCALE
        x1, y1 = x0 + PATCH_SIZE * SCALE, y0 + PATCH_SIZE * SCALE
        if not kept_mask[idx]:
            draw.rectangle([x0, y0, x1, y1], fill=(255, 0, 0, 120))  # dim pruned patches red
        draw.rectangle([x0, y0, x1, y1], outline=(80, 80, 80, 180))

    composed = Image.alpha_composite(base.convert("RGBA"), overlay).convert("RGB")
    draw2 = ImageDraw.Draw(composed)
    draw2.text((4, 4), label_name, fill=(0, 255, 0))
    return composed


def main(prune_k=50):
    images, labels, classes = load_examples(n=6)
    os.makedirs(OUT_DIR, exist_ok=True)

    patches = _patchify(jnp.array(images))
    _, top_idx = _select_top_k_patches(patches, prune_k)
    top_idx = np.array(top_idx)

    for i in range(len(images)):
        kept_mask = np.zeros(NUM_PATCHES, dtype=bool)
        kept_mask[top_idx[i]] = True
        label_name = str(classes[labels[i]])
        out = render(images[i], kept_mask, label_name)
        out_path = os.path.join(OUT_DIR, f"pruning_example_{i}_{label_name}.png")
        out.save(out_path)
        print(f"saved {out_path} ({kept_mask.sum()}/{NUM_PATCHES} patches kept)")


if __name__ == "__main__":
    main()
