"""
Builds and caches a [NUM_CLASSES, 50] matrix of real GloVe word embeddings
for the ModelNet10 class names, used by the zero-shot classification head
(model/zero_shot_head.py). Run once, offline -- the cached .npy is what
training/eval/serving actually load, so gensim (and the 66MB GloVe
download) is a one-time dev-time dependency, not something baked into the
serving image.

Multi-word class names (night_stand) are embedded as the mean of their
constituent words' vectors -- a standard, simple composition for word
embeddings when there's no multi-word entry.

Usage: python -m model.build_class_embeddings
"""

import os

import numpy as np

from data_pipeline.modelnet_loader import CLASSES

OUT_PATH = os.path.join(os.path.dirname(__file__), "class_embeddings.npy")


def build():
    import gensim.downloader as api

    print("loading glove-wiki-gigaword-50 (real pretrained word embeddings)...")
    glove = api.load("glove-wiki-gigaword-50")

    embeddings = []
    for cls in CLASSES:
        words = cls.split("_")
        missing = [w for w in words if w not in glove]
        if missing:
            raise ValueError(f"class '{cls}' has words not in GloVe vocab: {missing}")
        vec = np.mean([glove[w] for w in words], axis=0)
        embeddings.append(vec)
        print(f"  {cls}: embedded from {words}")

    embeddings = np.stack(embeddings).astype(np.float32)
    np.save(OUT_PATH, embeddings)
    print(f"saved [{embeddings.shape[0]}, {embeddings.shape[1]}] class embedding matrix to {OUT_PATH}")


if __name__ == "__main__":
    build()
