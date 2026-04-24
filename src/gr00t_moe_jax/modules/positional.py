"""Sinusoidal positional encoding, ported from
`gr00t/model/modules/embodiment_conditioned_mlp.py::SinusoidalPositionalEncoding`.

Produces a (B, T, w) encoding for a (B, T) grid of (typically) flow-matching timesteps
that have been broadcast across an action chunk. Matches the PyTorch reference exactly:

    half = w // 2
    exponent = -arange(half) * log(10000) / half
    freqs    = timesteps[..., None] * exp(exponent)
    enc      = cat([sin(freqs), cos(freqs)], dim=-1)
"""

from __future__ import annotations

import math

import jax.numpy as jnp
from flax import nnx


class SinusoidalPositionalEncoding(nnx.Module):
    def __init__(self, embedding_dim: int):
        # Odd sizes silently lose a channel (half = dim // 2, output = 2 * half).
        # Reject at construction — downstream callers like
        # MultiEmbodimentActionEncoder depend on the output width matching
        # the requested embedding_dim exactly.
        if embedding_dim % 2 != 0:
            raise ValueError(
                f"embedding_dim must be even (got {embedding_dim}); "
                "odd sizes would silently truncate the output to 2·(dim//2)"
            )
        self.embedding_dim = embedding_dim

    def __call__(self, timesteps: jnp.ndarray) -> jnp.ndarray:
        # timesteps: (B, T) — one scalar per chunk entry.
        half = self.embedding_dim // 2
        exponent = -jnp.arange(half, dtype=jnp.float32) * (math.log(10000.0) / half)
        freqs = timesteps.astype(jnp.float32)[..., None] * jnp.exp(exponent)
        return jnp.concatenate([jnp.sin(freqs), jnp.cos(freqs)], axis=-1)
