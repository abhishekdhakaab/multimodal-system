"""
Trains the temporal bucket-attention head (model/temporal_head.py) on
precomputed frame-sequence embeddings (model/generate_temporal_sequences.py),
and compares the three real conditions:

  (a) no memory    -- classify the latest frame alone, using the
                       EXISTING already-trained single-frame model's own
                       head (model.pkl's fusion head_w/head_b) -- not a
                       new model, the real original baseline
  (b) bucketed      -- T=8 frames compressed to M=4 tokens (2 recent +
                       2 averaged groups of 3 older frames)
  (c) full memory   -- all T=8 frames kept as individual tokens, no
                       compression -- the expensive upper bound

(b) and (c) share one trained temporal-attention module; the only
difference at evaluation time is which memory tensor is passed in --
this isolates "does bucketing lose much vs. unbounded memory" from
"does temporal attention help at all", which are two different
questions and shouldn't be conflated into one number.

Usage: python -m model.train_temporal
"""

import os
import pickle

import jax
import jax.numpy as jnp
import numpy as np
from jax import random

import model.temporal_head as th
from data_pipeline.modelnet_loader import CLASSES

HERE = os.path.dirname(__file__)
TEMPORAL_DIR = os.path.join(HERE, "..", "data_pipeline", "temporal")
FULL_MODEL_CHECKPOINT = os.path.join(HERE, "checkpoints", "model.pkl")
TEMPORAL_CHECKPOINT = os.path.join(HERE, "checkpoints", "temporal_head.pkl")

NUM_CLASSES = len(CLASSES)
LEARNING_RATE = 0.005
NUM_EPOCHS = 60
BATCH_SIZE = 32
ADAM_B1, ADAM_B2, ADAM_EPS = 0.9, 0.999, 1e-8


def load_sequences(split):
    d = np.load(os.path.join(TEMPORAL_DIR, f"{split}.npz"))
    return jnp.array(d["embeddings"]), jnp.array(d["labels"])  # [N, T, 96], [N]


def loss_fn(params, latest, memory, labels):
    logits = th.forward(params, latest, memory)
    log_probs = jax.nn.log_softmax(logits)
    one_hot = jax.nn.one_hot(labels, NUM_CLASSES)
    return -jnp.mean(jnp.sum(one_hot * log_probs, axis=-1))


@jax.jit
def train_step(params, adam_state, latest, memory, labels):
    loss, grads = jax.value_and_grad(loss_fn)(params, latest, memory, labels)
    t = adam_state["t"] + 1
    m = jax.tree_util.tree_map(lambda m, g: ADAM_B1 * m + (1 - ADAM_B1) * g, adam_state["m"], grads)
    v = jax.tree_util.tree_map(lambda v, g: ADAM_B2 * v + (1 - ADAM_B2) * (g**2), adam_state["v"], grads)
    m_hat = jax.tree_util.tree_map(lambda m: m / (1 - ADAM_B1**t), m)
    v_hat = jax.tree_util.tree_map(lambda v: v / (1 - ADAM_B2**t), v)
    updates = jax.tree_util.tree_map(
        lambda mh, vh: LEARNING_RATE * mh / (jnp.sqrt(vh) + ADAM_EPS), m_hat, v_hat
    )
    params = jax.tree_util.tree_map(lambda p, u: p - u, params, updates)
    return params, {"m": m, "v": v, "t": t}, loss


def accuracy(params, latest, memory, labels):
    logits = th.forward(params, latest, memory)
    preds = jnp.argmax(logits, axis=-1)
    return float(jnp.mean(preds == labels))


def baseline_no_memory_accuracy(embeddings, labels):
    """(a): classify the latest frame alone using the EXISTING trained
    single-frame model's own head, not a newly trained one."""
    with open(FULL_MODEL_CHECKPOINT, "rb") as f:
        full_params = pickle.load(f)
    head_w, head_b = full_params["fusion"]["head_w"], full_params["fusion"]["head_b"]
    latest = embeddings[:, -1, :]  # [N, 96], the most recent frame
    logits = latest @ head_w + head_b
    preds = jnp.argmax(logits, axis=-1)
    return float(jnp.mean(preds == labels))


def train_one(train_latest, train_memory, train_labels, seed=0):
    """Trains one temporal-attention head matched to ONE memory regime
    (either always-bucketed or always-full) -- trained and evaluated on
    the SAME memory type, so a model trained on bucketed memory is never
    evaluated on full memory out-of-distribution, or vice versa. That
    conflation is a real methodological trap this experiment must not
    make: without it, "bucketing beat full memory" could just mean "full
    memory was out-of-distribution for this model," not that bucketing
    is actually better."""
    key = random.PRNGKey(seed)
    params = th.init_params(key, NUM_CLASSES)
    adam_state = {
        "m": jax.tree_util.tree_map(jnp.zeros_like, params),
        "v": jax.tree_util.tree_map(jnp.zeros_like, params),
        "t": 0,
    }
    rng = np.random.default_rng(seed)

    for epoch in range(NUM_EPOCHS):
        perm = rng.permutation(train_latest.shape[0])
        for start in range(0, train_latest.shape[0], BATCH_SIZE):
            idx = jnp.array(perm[start : start + BATCH_SIZE])
            params, adam_state, loss = train_step(
                params, adam_state, train_latest[idx], train_memory[idx], train_labels[idx]
            )
    return params


def main():
    train_embeddings, train_labels = load_sequences("train")
    val_embeddings, val_labels = load_sequences("val")
    print(f"train sequences: {train_embeddings.shape}, val sequences: {val_embeddings.shape}")

    # condition (a): no memory, existing model, no training needed
    baseline_acc = baseline_no_memory_accuracy(val_embeddings, val_labels)
    print(f"\n(a) no memory (existing single-frame model, latest frame only): {baseline_acc:.4f}")

    train_latest = train_embeddings[:, -1, :]
    val_latest = val_embeddings[:, -1, :]

    # condition (b): a model trained AND evaluated on bucketed memory
    print("\ntraining model (b): bucketed memory (T=8 -> M=4 tokens)...")
    train_bucketed = th.bucket_sequence(train_embeddings)
    val_bucketed = th.bucket_sequence(val_embeddings)
    bucketed_params = train_one(train_latest, train_bucketed, train_labels, seed=0)
    bucketed_acc = accuracy(bucketed_params, val_latest, val_bucketed, val_labels)
    print(f"(b) bucketed memory accuracy: {bucketed_acc:.4f}")

    # condition (c): a SEPARATE model trained AND evaluated on full, uncompressed memory
    print("\ntraining model (c): full memory (all T=8 tokens, no compression)...")
    full_params = train_one(train_latest, train_embeddings, train_labels, seed=0)
    full_acc = accuracy(full_params, val_latest, val_embeddings, val_labels)
    print(f"(c) full memory accuracy: {full_acc:.4f}")

    print(f"\nsummary: no-memory={baseline_acc:.4f}  bucketed={bucketed_acc:.4f}  full={full_acc:.4f}")
    if full_acc > baseline_acc:
        print(f"bucketing recovers {(bucketed_acc - baseline_acc) / (full_acc - baseline_acc) * 100:.1f}% "
              f"of the gain full memory gets over no-memory (fair comparison: each model trained on its own regime)")

    with open(TEMPORAL_CHECKPOINT, "wb") as f:
        pickle.dump(jax.tree_util.tree_map(np.array, bucketed_params), f)
    print(f"saved bucketed model to {TEMPORAL_CHECKPOINT}")

    return baseline_acc, bucketed_acc, full_acc


if __name__ == "__main__":
    main()
