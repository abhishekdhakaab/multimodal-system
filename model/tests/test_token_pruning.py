import jax.numpy as jnp
from jax import random

from model.vision_encoder import (
    EMBED_DIM,
    NUM_PATCHES,
    _select_top_k_patches,
    _sink_patch_indices,
    forward,
    init_params,
)


def test_select_top_k_keeps_highest_intensity_patches():
    patches = jnp.zeros((2, 5, 4))
    patches = patches.at[0, 2, :].set(10.0)  # patch index 2 is clearly the most "informative"
    patches = patches.at[0, 4, :].set(5.0)
    selected, idx = _select_top_k_patches(patches, k=2)
    assert selected.shape == (2, 2, 4)
    assert set(idx[0].tolist()) == {2, 4}


def test_pruned_forward_runs_and_differs_from_full():
    key = random.PRNGKey(0)
    params = init_params(key)
    images = random.uniform(random.PRNGKey(1), (3, 40, 40))

    full = forward(params, images)
    pruned = forward(params, images, prune_k=50)

    assert full.shape == (3, EMBED_DIM)
    assert pruned.shape == (3, EMBED_DIM)
    assert not jnp.allclose(full, pruned)


def test_sink_tokens_always_included_regardless_of_content():
    patches = jnp.ones((1, NUM_PATCHES, 4))  # uniform content -- sinks have no content advantage
    sink_idx = _sink_patch_indices()
    # give one non-sink patch a huge score so it would normally dominate the pool
    patches = patches.at[0, 50, :].set(1000.0)
    _, idx = _select_top_k_patches(patches, k=6, use_sink_tokens=True)
    selected = set(idx[0].tolist())
    assert set(sink_idx.tolist()).issubset(selected)
    assert 50 in selected  # the high-content patch still gets one of the remaining slots


def test_pruned_forward_with_all_patches_kept_is_close_to_full():
    """Sanity check: pruning down to NUM_PATCHES (keeping everything) should
    recover the same set of tokens as the unpruned path, just reordered --
    so the two forward passes should match once position embeddings are
    correctly gathered."""
    key = random.PRNGKey(2)
    params = init_params(key)
    images = random.uniform(random.PRNGKey(3), (2, 40, 40))

    full = forward(params, images)
    kept_all = forward(params, images, prune_k=NUM_PATCHES)
    assert jnp.allclose(full, kept_all, atol=1e-4)
