"""
A small Mixture-of-Experts classification head: instead of one fixed
linear head over the fused embedding, route through one of NUM_EXPERTS
parallel linear heads, chosen by a learned gate. The real question this
tests (not assumed either way): does the gate learn a meaningful
specialization across the data, or does it collapse onto one expert
(a well-known real MoE failure mode)? See model/train_moe.py and
docs/moe_notes.md for the measured answer.

Standard MoE training trick used here: train with SOFT routing (a
differentiable weighted mixture of all experts' outputs, so gradients
flow to every expert and the gate), but evaluate with HARD routing
(argmax over the gate, only one expert's output used) -- this is what
would give real sparse-compute savings at inference in a larger MoE,
and is the actual thing worth measuring: does hard routing hold up
versus the soft-routed training objective?

Also includes a standard load-balancing auxiliary loss (discourages the
gate from collapsing all examples onto one expert), trained as part of
the total loss.
"""

import jax
import jax.numpy as jnp
from jax import random

NUM_EXPERTS = 3
FUSION_DIM = 96  # matches model/fusion.py's combined (vision_embed + attended lidar) dim


def init_params(key, num_classes):
    keys = random.split(key, NUM_EXPERTS + 1)
    experts = []
    for i in range(NUM_EXPERTS):
        ek = random.split(keys[i], 2)
        experts.append(
            {
                "w": random.normal(ek[0], (FUSION_DIM, num_classes)) * 0.1,
                "b": jnp.zeros(num_classes),
            }
        )
    return {
        "experts": experts,
        "gate_w": random.normal(keys[-1], (FUSION_DIM, NUM_EXPERTS)) * 0.1,
        "gate_b": jnp.zeros(NUM_EXPERTS),
    }


def _expert_logits(params, combined):
    """combined: [B, FUSION_DIM] -> [B, NUM_EXPERTS, num_classes]"""
    return jnp.stack([combined @ e["w"] + e["b"] for e in params["experts"]], axis=1)


def gate_probs(params, combined):
    gate_logits = combined @ params["gate_w"] + params["gate_b"]  # [B, NUM_EXPERTS]
    return jax.nn.softmax(gate_logits, axis=-1)


def forward_soft(params, combined):
    """Differentiable soft mixture -- used during training so gradients
    reach every expert and the gate. Returns (logits, gate_probs)."""
    probs = gate_probs(params, combined)  # [B, E]
    expert_logits = _expert_logits(params, combined)  # [B, E, C]
    logits = jnp.sum(probs[:, :, None] * expert_logits, axis=1)  # [B, C]
    return logits, probs


def forward_hard(params, combined):
    """Only the top-1 expert's output is used per example -- what a real
    sparse MoE would actually run at inference. Returns (logits, chosen_expert)."""
    probs = gate_probs(params, combined)  # [B, E]
    expert_logits = _expert_logits(params, combined)  # [B, E, C]
    chosen = jnp.argmax(probs, axis=-1)  # [B]
    logits = jnp.take_along_axis(expert_logits, chosen[:, None, None], axis=1)[:, 0, :]  # [B, C]
    return logits, chosen


def load_balance_loss(probs):
    """Penalizes the gate for deviating from uniform expert usage across
    the batch -- a standard MoE training trick to discourage collapse
    onto a single expert. probs: [B, NUM_EXPERTS]."""
    mean_usage = jnp.mean(probs, axis=0)  # [NUM_EXPERTS]
    target = 1.0 / NUM_EXPERTS
    return jnp.sum((mean_usage - target) ** 2)
