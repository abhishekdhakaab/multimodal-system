import jax.numpy as jnp
from jax import random

from model.moe_head import NUM_EXPERTS, FUSION_DIM, forward_hard, forward_soft, init_params, load_balance_loss


def test_forward_soft_shapes():
    key = random.PRNGKey(0)
    params = init_params(key, num_classes=10)
    combined = jnp.zeros((4, FUSION_DIM))
    logits, probs = forward_soft(params, combined)
    assert logits.shape == (4, 10)
    assert probs.shape == (4, NUM_EXPERTS)
    assert jnp.allclose(probs.sum(axis=-1), 1.0, atol=1e-5)


def test_forward_hard_selects_one_expert():
    key = random.PRNGKey(1)
    params = init_params(key, num_classes=10)
    combined = random.normal(random.PRNGKey(2), (5, FUSION_DIM))
    logits, chosen = forward_hard(params, combined)
    assert logits.shape == (5, 10)
    assert chosen.shape == (5,)
    assert jnp.all((chosen >= 0) & (chosen < NUM_EXPERTS))


def test_load_balance_loss_zero_when_uniform():
    uniform_probs = jnp.full((8, NUM_EXPERTS), 1.0 / NUM_EXPERTS)
    assert float(load_balance_loss(uniform_probs)) < 1e-6


def test_load_balance_loss_positive_when_collapsed():
    collapsed_probs = jnp.zeros((8, NUM_EXPERTS)).at[:, 0].set(1.0)
    assert float(load_balance_loss(collapsed_probs)) > 0.01
