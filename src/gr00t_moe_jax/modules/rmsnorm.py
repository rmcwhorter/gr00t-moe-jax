"""Qwen3RMSNorm (= Qwen2RMSNorm = T5 LayerNorm), ported from
`transformers/models/qwen3/modeling_qwen3.py::Qwen3RMSNorm`.

Root-mean-square layer norm with a single learnable scale vector (no bias,
no mean-subtraction). The reference always upcasts to fp32 for the
variance computation and casts back to input dtype at the end; we do the
same to preserve bit-exact parity.

    variance = mean(x**2, axis=-1)
    x        = x * rsqrt(variance + eps)
    out      = weight * x
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import nnx


class RMSNorm(nnx.Module):
    def __init__(self, hidden_size: int, *, eps: float = 1e-6, rngs: nnx.Rngs):
        # `rngs` accepted for API consistency with other NNX modules even
        # though the scale is initialised deterministically to ones.
        del rngs
        self.hidden_size = hidden_size
        self.eps = eps
        self.weight = nnx.Param(jnp.ones((hidden_size,), dtype=jnp.float32))

    def __call__(self, hidden_states: jnp.ndarray) -> jnp.ndarray:
        input_dtype = hidden_states.dtype
        x = hidden_states.astype(jnp.float32)
        variance = jnp.mean(x * x, axis=-1, keepdims=True)
        x = x * jax.lax.rsqrt(variance + self.eps)
        return (self.weight[...] * x).astype(input_dtype)
