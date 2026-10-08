import jax.numpy as jnp
from jax import random

from model.full_model import forward, init_params
from model.vision_encoder import EMBED_DIM as VISION_DIM, IMAGE_SIZE
from model.lidar_encoder import forward_per_point, init_params as lidar_init
from model.vision_encoder import forward as vision_forward, init_params as vision_init
from model.fusion import NUM_CLASSES
from data_pipeline.modelnet_loader import NUM_POINTS


def test_forward_shapes():
    key = random.PRNGKey(0)
    params = init_params(key)
    images = jnp.zeros((5, IMAGE_SIZE, IMAGE_SIZE))
    points = jnp.zeros((5, NUM_POINTS, 3))
    logits = forward(params, images, points)
    assert logits.shape == (5, NUM_CLASSES)


def test_vision_encoder_output_dim():
    key = random.PRNGKey(1)
    params = vision_init(key)
    images = jnp.zeros((3, IMAGE_SIZE, IMAGE_SIZE))
    embed = vision_forward(params, images)
    assert embed.shape == (3, VISION_DIM)


def test_lidar_encoder_per_point_shape():
    key = random.PRNGKey(2)
    params = lidar_init(key)
    points = jnp.zeros((3, NUM_POINTS, 3))
    features = forward_per_point(params, points)
    assert features.shape == (3, NUM_POINTS, VISION_DIM)


def test_different_inputs_give_different_logits():
    key = random.PRNGKey(3)
    params = init_params(key)
    images_a = jnp.zeros((1, IMAGE_SIZE, IMAGE_SIZE))
    images_b = jnp.ones((1, IMAGE_SIZE, IMAGE_SIZE))
    points = jnp.zeros((1, NUM_POINTS, 3))
    logits_a = forward(params, images_a, points)
    logits_b = forward(params, images_b, points)
    assert not jnp.allclose(logits_a, logits_b)


def test_fusion_num_classes_matches_real_dataset():
    """Guards against the exact bug that was found: fusion.py's NUM_CLASSES
    silently drifting out of sync with the real dataset's class count."""
    from data_pipeline.modelnet_loader import CLASSES

    assert NUM_CLASSES == len(CLASSES)
