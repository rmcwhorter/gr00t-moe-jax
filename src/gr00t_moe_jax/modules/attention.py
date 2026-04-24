"""Multi-head attention, matching the subset of
`diffusers.models.attention.Attention` that GR00T uses: optional
cross-attention via encoder_hidden_states, padding mask on the key side,
scaled dot-product, linear out-projection.

GR00T passes `attention_bias=True` and `out_bias=True` (default).

The structure matches the PyTorch reference attributes (`to_q`, `to_k`,
`to_v`, and an `out_proj` that corresponds to `to_out[0]`). A weight
loader can map `to_out.0.weight` → `out_proj.weight`.
"""

from __future__ import annotations

import math

import jax.numpy as jnp
from flax import nnx
from jax import nn as jnn


class Attention(nnx.Module):
    """Standard multi-head attention. Cross-attention when `encoder_hidden_states` is given.

    If `cross_attention_dim` is None, K/V weights share `query_dim` as their
    input dim (self-attention weights shape), even if called with separate
    encoder states — diffusers' default behaviour.
    """

    def __init__(
        self,
        *,
        query_dim: int,
        heads: int = 8,
        dim_head: int = 64,
        cross_attention_dim: int | None = None,
        bias: bool = True,
        out_bias: bool = True,
        rngs: nnx.Rngs,
    ):
        self.heads = heads
        self.dim_head = dim_head
        self.inner_dim = heads * dim_head
        self.query_dim = query_dim
        self.cross_attention_dim = cross_attention_dim or query_dim
        self.scale = 1.0 / math.sqrt(dim_head)

        self.to_q = nnx.Linear(query_dim, self.inner_dim, use_bias=bias, rngs=rngs)
        self.to_k = nnx.Linear(self.cross_attention_dim, self.inner_dim, use_bias=bias, rngs=rngs)
        self.to_v = nnx.Linear(self.cross_attention_dim, self.inner_dim, use_bias=bias, rngs=rngs)
        self.out_proj = nnx.Linear(self.inner_dim, query_dim, use_bias=out_bias, rngs=rngs)

    def __call__(
        self,
        hidden_states: jnp.ndarray,  # (B, T_q, query_dim)
        encoder_hidden_states: jnp.ndarray | None = None,  # (B, T_k, kv_dim) or None
        attention_mask: jnp.ndarray | None = None,  # (B, T_k) bool, True = keep
    ) -> jnp.ndarray:
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        B, T_q, _ = hidden_states.shape
        T_k = encoder_hidden_states.shape[1]
        H, D = self.heads, self.dim_head

        q = self.to_q(hidden_states).reshape(B, T_q, H, D).transpose(0, 2, 1, 3)
        k = self.to_k(encoder_hidden_states).reshape(B, T_k, H, D).transpose(0, 2, 1, 3)
        v = self.to_v(encoder_hidden_states).reshape(B, T_k, H, D).transpose(0, 2, 1, 3)
        # Shapes: (B, H, T, D)

        scores = jnp.einsum("bhqd,bhkd->bhqk", q, k) * self.scale  # (B, H, T_q, T_k)

        if attention_mask is not None:
            if attention_mask.ndim == 2:
                attention_mask = attention_mask[:, None, None, :]  # (B, 1, 1, T_k)
            mask_value = jnp.finfo(scores.dtype).min
            scores = jnp.where(attention_mask.astype(jnp.bool_), scores, mask_value)

        # Edge case: if every key is masked for some query, softmax(all -inf)
        # returns a uniform distribution (so the output becomes the mean of V)
        # rather than zero. This matches PyTorch/diffusers behavior exactly,
        # which is why we preserve it — switching to zero-output would break
        # parity. See `tests/parity/test_all_masked_softmax.py` for the pin.
        # AlternateVLDiT can produce this shape when a batch item has no
        # image tokens (→ image_attn_mask all False) or no text tokens.
        attn = jnn.softmax(scores, axis=-1)
        out = jnp.einsum("bhqk,bhkd->bhqd", attn, v)  # (B, H, T_q, D)
        out = out.transpose(0, 2, 1, 3).reshape(B, T_q, self.inner_dim)
        return self.out_proj(out)
