import jax.numpy as jnp
from jax import random

from model.full_model import forward, init_params
from model.vision_encoder import EMBED_DIM as VISION_DIM
from model.lidar_encoder import forward_per_point, init_params as lidar_init
from model.vision_encoder import forward as vision_forward, init_params as vision_init


def test_forward_shapes():
    key = random.PRNGKey(0)
    params = init_params(key)
    images = jnp.zeros((5, 32, 32))
    points = jnp.zeros((5, 64, 3))
    logits = forward(params, images, points)
    assert logits.shape == (5, 4)


def test_vision_encoder_output_dim():
    key = random.PRNGKey(1)
    params = vision_init(key)
    images = jnp.zeros((3, 32, 32))
    embed = vision_forward(params, images)
    assert embed.shape == (3, VISION_DIM)


def test_lidar_encoder_per_point_shape():
    key = random.PRNGKey(2)
    params = lidar_init(key)
    points = jnp.zeros((3, 64, 3))
    features = forward_per_point(params, points)
    assert features.shape == (3, 64, VISION_DIM)


def test_different_inputs_give_different_logits():
    key = random.PRNGKey(3)
    params = init_params(key)
    images_a = jnp.zeros((1, 32, 32))
    images_b = jnp.ones((1, 32, 32))
    points = jnp.zeros((1, 64, 3))
    logits_a = forward(params, images_a, points)
    logits_b = forward(params, images_b, points)
    assert not jnp.allclose(logits_a, logits_b)
