"""
Profiles one forward pass of the trained model to find the most expensive
op/op-group. The finding here determines what Phase 3's custom CUDA kernel
fuses — write it down honestly, don't pick a target in advance.
"""

import time

import jax
import jax.numpy as jnp
from jax import random

from model import vision_encoder, lidar_encoder, fusion
from model.full_model import forward, init_params

BATCH_SIZE = 64
N_TIMING_RUNS = 50


def time_fn(fn, *args, n=N_TIMING_RUNS):
    # warm up (first call includes JIT trace/compile time)
    out = fn(*args)
    jax.block_until_ready(out)

    start = time.perf_counter()
    for _ in range(n):
        out = fn(*args)
    jax.block_until_ready(out)
    elapsed = time.perf_counter() - start
    return elapsed / n


def main():
    key = random.PRNGKey(0)
    params = init_params(key)
    images = jnp.zeros((BATCH_SIZE, 32, 32))
    points = jnp.zeros((BATCH_SIZE, 64, 3))

    full_fn = jax.jit(forward)
    full_time = time_fn(full_fn, params, images, points)
    print(f"full forward pass: {full_time*1000:.3f} ms/batch (batch={BATCH_SIZE})")

    vision_fn = jax.jit(vision_encoder.forward)
    vision_time = time_fn(vision_fn, params["vision"], images)
    print(f"  vision encoder only:  {vision_time*1000:.3f} ms  ({vision_time/full_time*100:.1f}% of total)")

    lidar_fn = jax.jit(lidar_encoder.forward_per_point)
    lidar_time = time_fn(lidar_fn, params["lidar"], points)
    print(f"  lidar encoder only:   {lidar_time*1000:.3f} ms  ({lidar_time/full_time*100:.1f}% of total)")

    vision_embed = vision_encoder.forward(params["vision"], images)
    lidar_points = lidar_encoder.forward_per_point(params["lidar"], points)
    fusion_fn = jax.jit(fusion.forward)
    fusion_time = time_fn(fusion_fn, params["fusion"], vision_embed, lidar_points)
    print(f"  fusion (cross-attn) only: {fusion_time*1000:.3f} ms  ({fusion_time/full_time*100:.1f}% of total)")

    print()
    print("breakdown (note: components sum to more than 100% because the full")
    print("forward pass is JIT-fused end to end; component timings below are")
    print("each individually JIT-compiled and so include their own XLA fusion")
    print("boundaries that don't exist in the full fused graph)")


if __name__ == "__main__":
    main()
