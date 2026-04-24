"""MultiEmbodimentActionEncoder, ported from
`gr00t/model/modules/embodiment_conditioned_mlp.py::MultiEmbodimentActionEncoder`.

Encodes a noised action trajectory + flow-matching timestep into the DiT's
inner embedding dimension. Three per-category linear layers, with sinusoidal
timestep encoding concatenated in the middle:

    a = W1(action)                            # (B, T, w)
    tau = sinusoid(timestep broadcast to T)   # (B, T, w)
    x = swish(W2(cat([a, tau])))              # (B, T, w)
    x = W3(x)                                 # (B, T, w)
"""

from __future__ import annotations

import jax.numpy as jnp
from flax import nnx

from gr00t_moe_jax.modules.category_specific import CategorySpecificLinear
from gr00t_moe_jax.modules.positional import SinusoidalPositionalEncoding


def _swish(x: jnp.ndarray) -> jnp.ndarray:
    # Swish = SiLU = x * sigmoid(x).
    return x * nnx.sigmoid(x)


class MultiEmbodimentActionEncoder(nnx.Module):
    def __init__(
        self,
        action_dim: int,
        hidden_size: int,
        num_embodiments: int,
        *,
        rngs: nnx.Rngs,
    ):
        self.hidden_size = hidden_size
        self.num_embodiments = num_embodiments

        self.W1 = CategorySpecificLinear(num_embodiments, action_dim, hidden_size, rngs=rngs)
        self.W2 = CategorySpecificLinear(
            num_embodiments, 2 * hidden_size, hidden_size, rngs=rngs
        )
        self.W3 = CategorySpecificLinear(num_embodiments, hidden_size, hidden_size, rngs=rngs)
        self.pos_encoding = SinusoidalPositionalEncoding(hidden_size)

    def __call__(
        self,
        actions: jnp.ndarray,  # (B, T, action_dim)
        timesteps: jnp.ndarray,  # (B,) integer or float timesteps per batch item
        cat_ids: jnp.ndarray,  # (B,) int embodiment IDs
    ) -> jnp.ndarray:
        B, T, _ = actions.shape
        if timesteps.ndim != 1 or timesteps.shape[0] != B:
            raise ValueError(
                f"timesteps must have shape (B,), got {timesteps.shape}"
            )

        # Broadcast the per-batch scalar timestep across the chunk.
        timesteps_bt = jnp.broadcast_to(timesteps[:, None], (B, T))

        a_emb = self.W1(actions, cat_ids)  # (B, T, w)
        tau_emb = self.pos_encoding(timesteps_bt).astype(a_emb.dtype)  # (B, T, w)

        x = jnp.concatenate([a_emb, tau_emb], axis=-1)  # (B, T, 2w)
        x = _swish(self.W2(x, cat_ids))  # (B, T, w)
        x = self.W3(x, cat_ids)  # (B, T, w)
        return x
