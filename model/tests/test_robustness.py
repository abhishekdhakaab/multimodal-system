import jax.numpy as jnp
from jax import random

from model.eval_robustness import accuracy
from model.full_model import init_params


def test_accuracy_runs_with_zeroed_modality():
    key = random.PRNGKey(0)
    params = init_params(key)
    images = jnp.zeros((5, 40, 40))
    points = jnp.zeros((5, 256, 3))
    labels = jnp.array([0, 1, 2, 3, 4])
    acc = accuracy(params, images, points, labels)
    assert 0.0 <= acc <= 1.0


def test_accuracy_runs_with_noisy_modality():
    key = random.PRNGKey(1)
    params = init_params(key)
    images = random.normal(random.PRNGKey(2), (5, 40, 40))
    points = random.normal(random.PRNGKey(3), (5, 256, 3))
    labels = jnp.array([0, 1, 2, 3, 4])
    acc = accuracy(params, images, points, labels)
    assert 0.0 <= acc <= 1.0
