import os

import jax.numpy as jnp
import numpy as np
import pytest
from jax import random

from model.full_model_zero_shot import init_params, similarity_logits
from model.zero_shot_head import FUSION_DIM, load_class_embeddings
from data_pipeline.modelnet_loader import CLASSES, IMAGE_SIZE, NUM_POINTS

CLASS_EMBEDDINGS_EXISTS = os.path.exists(
    os.path.join(os.path.dirname(os.path.dirname(__file__)), "class_embeddings.npy")
)


@pytest.mark.skipif(not CLASS_EMBEDDINGS_EXISTS, reason="run model/build_class_embeddings.py first")
def test_class_embeddings_shape_and_normalization():
    embeddings = load_class_embeddings()
    assert embeddings.shape == (len(CLASSES), 50)
    norms = jnp.linalg.norm(embeddings, axis=1)
    assert jnp.allclose(norms, 1.0, atol=1e-5)


@pytest.mark.skipif(not CLASS_EMBEDDINGS_EXISTS, reason="run model/build_class_embeddings.py first")
def test_similarity_logits_shape():
    key = random.PRNGKey(0)
    params = init_params(key)
    class_embeddings = load_class_embeddings()

    images = jnp.zeros((4, IMAGE_SIZE, IMAGE_SIZE))
    points = jnp.zeros((4, NUM_POINTS, 3))

    logits = similarity_logits(params, images, points, class_embeddings)
    assert logits.shape == (4, len(CLASSES))


@pytest.mark.skipif(not CLASS_EMBEDDINGS_EXISTS, reason="run model/build_class_embeddings.py first")
def test_similarity_logits_bounded_by_temperature():
    """Cosine similarity is in [-1, 1], so logits must stay within [-TEMPERATURE, TEMPERATURE]."""
    from model.zero_shot_head import TEMPERATURE

    key = random.PRNGKey(1)
    params = init_params(key)
    class_embeddings = load_class_embeddings()

    images = random.normal(random.PRNGKey(2), (4, IMAGE_SIZE, IMAGE_SIZE))
    points = random.normal(random.PRNGKey(3), (4, NUM_POINTS, 3))

    logits = similarity_logits(params, images, points, class_embeddings)
    assert float(jnp.max(jnp.abs(logits))) <= TEMPERATURE + 1e-4
