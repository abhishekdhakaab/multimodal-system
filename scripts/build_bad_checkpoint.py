"""
Generates a deliberately broken checkpoint (randomly initialized, never
trained) to use as the "bad deployment" in the Phase 6 canary/rollback
demo -- standing in for something like a training job that silently
failed to converge or a corrupted checkpoint upload. Its accuracy will be
near the 10% random-chance baseline, far enough below the real model's
~83-86% to trip the rollback threshold.
"""

import os
import pickle
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from jax import random

from model.full_model import init_params

HERE = os.path.dirname(__file__)
OUT_PATH = os.path.join(HERE, "..", "model", "checkpoints", "model_bad.pkl")

if __name__ == "__main__":
    import jax

    params = init_params(random.PRNGKey(999))
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "wb") as f:
        pickle.dump(jax.tree_util.tree_map(np.array, params), f)
    print(f"wrote untrained (deliberately bad) checkpoint to {OUT_PATH}")
