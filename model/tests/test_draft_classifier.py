import jax.numpy as jnp
from jax import random

from model.draft_classifier import NUM_CLASSES, NUM_FEATURES, extract_features, forward, init_params


def test_extract_features_shape():
    images = jnp.zeros((5, 40, 40))
    points = jnp.zeros((5, 256, 3))
    features = extract_features(images, points)
    assert features.shape == (5, NUM_FEATURES)


def test_forward_shape():
    key = random.PRNGKey(0)
    params = init_params(key)
    images = jnp.zeros((5, 40, 40))
    points = jnp.zeros((5, 256, 3))
    logits = forward(params, images, points)
    assert logits.shape == (5, NUM_CLASSES)


def test_different_inputs_give_different_features():
    images_a = jnp.zeros((1, 40, 40))
    images_b = jnp.ones((1, 40, 40))
    points = jnp.zeros((1, 256, 3))
    fa = extract_features(images_a, points)
    fb = extract_features(images_b, points)
    assert not jnp.allclose(fa, fb)
