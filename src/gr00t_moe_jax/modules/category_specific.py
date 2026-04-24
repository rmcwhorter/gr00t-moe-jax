"""Category-specific (embodiment-indexed) linear and MLP layers,
ported from `gr00t/model/modules/embodiment_conditioned_mlp.py`.

Each module stores a weight *tensor* of shape (num_categories, in_dim, out_dim)
and a bias of shape (num_categories, out_dim). At forward time, the relevant
slice is gathered by integer `cat_ids` (one category per batch item) and a
batched matmul is performed.

This is structurally a hard-routed MoE — each embodiment picks exactly one
"expert" per layer, with no learned gate. A natural starting point for the
learned-router MoE generalization in later phases.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import nnx


class CategorySpecificLinear(nnx.Module):
    """Per-category affine: y = x @ W[cat_id] + b[cat_id]."""

    def __init__(
        self,
        num_categories: int,
        in_features: int,
        out_features: int,
        *,
        rngs: nnx.Rngs,
    ):
        self.num_categories = num_categories
        self.in_features = in_features
        self.out_features = out_features

        key = rngs.params()
        w_init = 0.02 * jax.random.normal(
            key, (num_categories, in_features, out_features), dtype=jnp.float32
        )
        self.W = nnx.Param(w_init)
        self.b = nnx.Param(jnp.zeros((num_categories, out_features), dtype=jnp.float32))

    def __call__(self, x: jnp.ndarray, cat_ids: jnp.ndarray) -> jnp.ndarray:
        # x: (B, T, in_features) or (B, in_features)
        # cat_ids: (B,) int
        selected_W = self.W[...][cat_ids]  # (B, in_features, out_features)
        selected_b = self.b[...][cat_ids]  # (B, out_features)

        if x.ndim == 2:
            # (B, in) -> (B, out)
            return jnp.einsum("bi,bio->bo", x, selected_W) + selected_b
        # (B, T, in) -> (B, T, out)
        return jnp.einsum("bti,bio->bto", x, selected_W) + selected_b[:, None, :]


class CategorySpecificMLP(nnx.Module):
    """Two-layer per-category MLP: ReLU(W1 x) -> W2."""

    def __init__(
        self,
        num_categories: int,
        in_features: int,
        hidden_features: int,
        out_features: int,
        *,
        rngs: nnx.Rngs,
    ):
        self.num_categories = num_categories
        self.layer1 = CategorySpecificLinear(
            num_categories, in_features, hidden_features, rngs=rngs
        )
        self.layer2 = CategorySpecificLinear(
            num_categories, hidden_features, out_features, rngs=rngs
        )

    def __call__(self, x: jnp.ndarray, cat_ids: jnp.ndarray) -> jnp.ndarray:
        hidden = jax.nn.relu(self.layer1(x, cat_ids))
        return self.layer2(hidden, cat_ids)
