"""Qwen3MLP (SwiGLU), ported from
`transformers/models/qwen3/modeling_qwen3.py::Qwen3MLP`.

    out = down_proj(silu(gate_proj(x)) * up_proj(x))

Three dense layers, all bias-free in the default Qwen3 config. The
activation (`config.hidden_act`) is SiLU for Qwen3; we hardcode that
here since the only other realistic choice is GELU, and GR00T uses the
default Qwen3 config.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import nnx


class Qwen3MLP(nnx.Module):
    def __init__(
        self,
        *,
        hidden_size: int,
        intermediate_size: int,
        bias: bool = False,
        rngs: nnx.Rngs,
    ):
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size

        self.gate_proj = nnx.Linear(hidden_size, intermediate_size, use_bias=bias, rngs=rngs)
        self.up_proj = nnx.Linear(hidden_size, intermediate_size, use_bias=bias, rngs=rngs)
        self.down_proj = nnx.Linear(intermediate_size, hidden_size, use_bias=bias, rngs=rngs)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return self.down_proj(jax.nn.silu(self.gate_proj(x)) * self.up_proj(x))
