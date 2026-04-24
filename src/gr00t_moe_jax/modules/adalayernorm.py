"""AdaLayerNorm, ported from `gr00t/model/modules/dit.py::AdaLayerNorm`.

Layer norm whose scale/shift are produced from the timestep embedding rather than
learned directly. The workhorse mechanism that makes a DiT a DiT: conditioning
enters as per-layer modulation rather than through attention or concatenation.

    temb   →  SiLU  →  Linear(dim → 2·dim)  →  (scale, shift)
    x      →  LayerNorm(affine=False)       →  x
    out    =  x · (1 + scale) + shift

(LayerNorm here has no learned affine since the adaptive scale/shift replaces it.)
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import nnx


class AdaLayerNorm(nnx.Module):
    def __init__(
        self,
        embedding_dim: int,
        *,
        norm_eps: float = 1e-5,
        rngs: nnx.Rngs,
    ):
        self.embedding_dim = embedding_dim
        # temb path.
        self.linear = nnx.Linear(embedding_dim, 2 * embedding_dim, rngs=rngs)
        # Content path — no affine, scale/shift come from temb.
        self.norm = nnx.LayerNorm(
            embedding_dim,
            epsilon=norm_eps,
            use_bias=False,
            use_scale=False,
            rngs=rngs,
        )

    def __call__(self, x: jnp.ndarray, temb: jnp.ndarray) -> jnp.ndarray:
        # x: (B, T, D); temb: (B, D).
        temb = self.linear(jax.nn.silu(temb))  # (B, 2D)
        scale, shift = jnp.split(temb, 2, axis=-1)  # each (B, D)
        return self.norm(x) * (1.0 + scale[:, None, :]) + shift[:, None, :]
