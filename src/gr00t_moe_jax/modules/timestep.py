"""Timestep embedding for flow-matching / diffusion models, matching
`diffusers.models.embeddings.{Timesteps, TimestepEmbedding}` as used by GR00T.

GR00T wires these as:

    TimestepEncoder(embedding_dim)
      = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=1)
      → Linear(256 → embedding_dim) → SiLU → Linear(embedding_dim → embedding_dim)

The `downscale_freq_shift=1` branch is what makes this *different* from the
plain `SinusoidalPositionalEncoding` used by the action encoder: the
exponent divisor is `half_dim - 1` instead of `half_dim`, and the sin/cos
halves are swapped at the end (flip_sin_to_cos=True).
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
from flax import nnx


def get_timestep_embedding(
    timesteps: jnp.ndarray,
    embedding_dim: int,
    *,
    flip_sin_to_cos: bool = True,
    downscale_freq_shift: float = 1.0,
    scale: float = 1.0,
    max_period: float = 10000.0,
) -> jnp.ndarray:
    """Matches diffusers `get_timestep_embedding`.

    Args:
        timesteps: 1-D integer/float tensor of shape (N,).
        embedding_dim: output channel count. Must be even (asserted).
    Returns:
        (N, embedding_dim) float array.
    """
    assert embedding_dim % 2 == 0, "embedding_dim must be even"
    half_dim = embedding_dim // 2
    exponent = -math.log(max_period) * jnp.arange(half_dim, dtype=jnp.float32)
    exponent = exponent / (half_dim - downscale_freq_shift)
    emb = jnp.exp(exponent)
    emb = timesteps.astype(jnp.float32)[:, None] * emb[None, :]
    emb = scale * emb
    emb = jnp.concatenate([jnp.sin(emb), jnp.cos(emb)], axis=-1)
    if flip_sin_to_cos:
        emb = jnp.concatenate([emb[:, half_dim:], emb[:, :half_dim]], axis=-1)
    return emb


class TimestepEncoder(nnx.Module):
    """Full pipeline: (N,) int/float timesteps → (N, embedding_dim) embedding."""

    def __init__(
        self,
        embedding_dim: int,
        *,
        inner_channels: int = 256,
        flip_sin_to_cos: bool = True,
        downscale_freq_shift: float = 1.0,
        rngs: nnx.Rngs,
    ):
        self.embedding_dim = embedding_dim
        self.inner_channels = inner_channels
        self.flip_sin_to_cos = flip_sin_to_cos
        self.downscale_freq_shift = downscale_freq_shift

        self.linear_1 = nnx.Linear(inner_channels, embedding_dim, rngs=rngs)
        self.linear_2 = nnx.Linear(embedding_dim, embedding_dim, rngs=rngs)

    def __call__(self, timesteps: jnp.ndarray) -> jnp.ndarray:
        emb = get_timestep_embedding(
            timesteps,
            self.inner_channels,
            flip_sin_to_cos=self.flip_sin_to_cos,
            downscale_freq_shift=self.downscale_freq_shift,
        )
        emb = self.linear_1(emb)
        emb = jax.nn.silu(emb)
        emb = self.linear_2(emb)
        return emb
