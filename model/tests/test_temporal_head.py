import jax.numpy as jnp
from jax import random

from model.temporal_head import FUSION_DIM, bucket_sequence, forward, init_params


def test_bucket_sequence_shape():
    frame_embeddings = jnp.zeros((3, 8, FUSION_DIM))
    bucketed = bucket_sequence(frame_embeddings)
    assert bucketed.shape == (3, 4, FUSION_DIM)  # 2 grouped + 2 recent = 4


def test_bucket_sequence_averages_correctly():
    frame_embeddings = jnp.arange(8 * FUSION_DIM, dtype=jnp.float32).reshape(1, 8, FUSION_DIM)
    bucketed = bucket_sequence(frame_embeddings)
    # first group = mean of frames 0,1,2; recent frames (6,7) kept verbatim
    expected_group0 = frame_embeddings[0, 0:3].mean(axis=0)
    assert jnp.allclose(bucketed[0, 0], expected_group0)
    assert jnp.allclose(bucketed[0, 2], frame_embeddings[0, 6])
    assert jnp.allclose(bucketed[0, 3], frame_embeddings[0, 7])


def test_forward_shapes():
    key = random.PRNGKey(0)
    params = init_params(key, num_classes=10)
    latest = jnp.zeros((4, FUSION_DIM))
    memory = jnp.zeros((4, 5, FUSION_DIM))
    logits = forward(params, latest, memory)
    assert logits.shape == (4, 10)
