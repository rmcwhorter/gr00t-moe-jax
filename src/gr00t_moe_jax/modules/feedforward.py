"""FeedForward MLP with configurable activation, matching
`diffusers.models.attention.FeedForward`. GR00T's DiT uses
`activation_fn="gelu-approximate"` — a plain non-gated MLP with
tanh-approximate GELU — not GeGLU, despite some surface similarity.

Supported activation_fn values:
- "gelu":             Linear(dim → inner) → gelu(exact)        → Linear(inner → dim_out)
- "gelu-approximate": Linear(dim → inner) → gelu(tanh-approx)  → Linear(inner → dim_out)
- "geglu":            Linear(dim → 2·inner) → hidden·gelu(gate) → Linear(inner → dim_out)
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import nnx

_SUPPORTED = ("gelu", "gelu-approximate", "geglu")


class FeedForward(nnx.Module):
    def __init__(
        self,
        dim: int,
        *,
        inner_dim: int | None = None,
        mult: int = 4,
        dim_out: int | None = None,
        activation_fn: str = "gelu-approximate",
        bias: bool = True,
        rngs: nnx.Rngs,
    ):
        if activation_fn not in _SUPPORTED:
            raise ValueError(
                f"activation_fn={activation_fn!r} not in {_SUPPORTED}"
            )
        if inner_dim is None:
            inner_dim = dim * mult
        if dim_out is None:
            dim_out = dim

        self.dim = dim
        self.inner_dim = inner_dim
        self.dim_out = dim_out
        self.activation_fn = activation_fn

        proj_in_out = inner_dim * 2 if activation_fn == "geglu" else inner_dim
        self.proj_in = nnx.Linear(dim, proj_in_out, use_bias=bias, rngs=rngs)
        self.proj_out = nnx.Linear(inner_dim, dim_out, use_bias=bias, rngs=rngs)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        h = self.proj_in(x)
        if self.activation_fn == "gelu-approximate":
            h = jax.nn.gelu(h, approximate=True)
        elif self.activation_fn == "gelu":
            h = jax.nn.gelu(h, approximate=False)
        else:  # geglu
            hidden, gate = jnp.split(h, 2, axis=-1)
            h = hidden * jax.nn.gelu(gate, approximate=False)
        return self.proj_out(h)
