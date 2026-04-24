"""Qwen3Attention, ported from
`transformers/models/qwen3/modeling_qwen3.py::Qwen3Attention`.

Characteristic features vs. a vanilla LLaMA-style attention:

1. **Per-head RMSNorm on Q and K** (applied between the projection and RoPE).
   These norms act on the `head_dim` axis of each individual head rather than
   on the full fused projection output — "only on the head dim" per the
   reference comment.

2. **Grouped-Query Attention (GQA)**: `num_key_value_heads <= num_attention_heads`.
   K and V are projected to the smaller count, then repeated to match Q heads
   before the scaled dot-product.

3. **Additive attention mask convention**: the mask is added to the logits
   (0 for attend, -inf for mask), not multiplied / boolean-selected like our
   earlier DiT attention. The mask therefore carries the causal / padding
   information baked in as a full (B, H, T_q, T_k) or broadcastable tensor.

4. **Fp32 softmax**: the softmax is computed in fp32 and cast back to the
   input dtype. This matters for bit-exact parity with the reference.

5. **Bias-free projections** (Qwen3 default: attention_bias=False).
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
from flax import nnx

from gr00t_moe_jax.modules.rmsnorm import RMSNorm
from gr00t_moe_jax.modules.rope import apply_rotary_pos_emb


def repeat_kv(hidden_states: jnp.ndarray, n_rep: int) -> jnp.ndarray:
    """Repeat KV heads to match Q heads for GQA.

    Input shape: (B, num_kv_heads, T, head_dim)
    Output shape: (B, num_kv_heads * n_rep, T, head_dim)
    """
    if n_rep == 1:
        return hidden_states
    B, num_kv_heads, T, head_dim = hidden_states.shape
    # Expand then reshape — matches the PyTorch `expand + reshape` pattern.
    expanded = jnp.broadcast_to(
        hidden_states[:, :, None, :, :],
        (B, num_kv_heads, n_rep, T, head_dim),
    )
    return expanded.reshape(B, num_kv_heads * n_rep, T, head_dim)


class Qwen3Attention(nnx.Module):
    def __init__(
        self,
        *,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        rms_norm_eps: float = 1e-6,
        attention_bias: bool = False,
        rngs: nnx.Rngs,
    ):
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.num_key_value_groups = num_attention_heads // num_key_value_heads
        self.scaling = 1.0 / math.sqrt(head_dim)

        self.q_proj = nnx.Linear(
            hidden_size, num_attention_heads * head_dim, use_bias=attention_bias, rngs=rngs
        )
        self.k_proj = nnx.Linear(
            hidden_size, num_key_value_heads * head_dim, use_bias=attention_bias, rngs=rngs
        )
        self.v_proj = nnx.Linear(
            hidden_size, num_key_value_heads * head_dim, use_bias=attention_bias, rngs=rngs
        )
        self.o_proj = nnx.Linear(
            num_attention_heads * head_dim, hidden_size, use_bias=attention_bias, rngs=rngs
        )

        # Per-head RMSNorm applied AFTER projection reshape, BEFORE RoPE.
        self.q_norm = RMSNorm(head_dim, eps=rms_norm_eps, rngs=rngs)
        self.k_norm = RMSNorm(head_dim, eps=rms_norm_eps, rngs=rngs)

    def __call__(
        self,
        hidden_states: jnp.ndarray,  # (B, T, hidden_size)
        position_embeddings: tuple[jnp.ndarray, jnp.ndarray],  # (cos, sin) each (B, T, head_dim)
        attention_mask: jnp.ndarray | None = None,  # additive, broadcastable to (B, H, T_q, T_k)
    ) -> jnp.ndarray:
        B, T, _ = hidden_states.shape
        input_dtype = hidden_states.dtype

        # Project then reshape to (B, T, num_heads, head_dim) before per-head norm.
        q = self.q_proj(hidden_states).reshape(B, T, self.num_attention_heads, self.head_dim)
        k = self.k_proj(hidden_states).reshape(B, T, self.num_key_value_heads, self.head_dim)
        v = self.v_proj(hidden_states).reshape(B, T, self.num_key_value_heads, self.head_dim)

        # Per-head RMSNorm on the head_dim axis (reference does this after reshape,
        # which is why the norm works cleanly per head without further reshaping).
        q = self.q_norm(q)
        k = self.k_norm(k)

        # Transpose to (B, num_heads, T, head_dim) for attention.
        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        # RoPE.
        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # GQA: repeat K/V heads to match Q heads.
        k = repeat_kv(k, self.num_key_value_groups)
        v = repeat_kv(v, self.num_key_value_groups)

        # Scaled dot-product attention with additive mask + fp32 softmax.
        attn_weights = jnp.einsum("bhqd,bhkd->bhqk", q, k) * self.scaling
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        attn_weights = jax.nn.softmax(attn_weights.astype(jnp.float32), axis=-1).astype(input_dtype)

        attn_output = jnp.einsum("bhqk,bhkd->bhqd", attn_weights, v)
        # (B, H, T, head_dim) → (B, T, H*head_dim)
        attn_output = attn_output.transpose(0, 2, 1, 3).reshape(B, T, -1)
        return self.o_proj(attn_output)
