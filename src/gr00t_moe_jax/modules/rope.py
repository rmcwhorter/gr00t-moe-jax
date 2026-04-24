"""Rotary Position Embedding (RoPE), ported from
`transformers/models/qwen3/modeling_qwen3.py`.

Three pieces:

1. `compute_default_inv_freq`: Precompute the inverse frequencies that
   define the rotation rates per feature pair. Only depends on head_dim
   and `rope_theta`.

2. `RotaryEmbedding`: module that holds `inv_freq` and produces the
   (cos, sin) tables for a given position_ids tensor. In HF's code,
   `inv_freq` is stored as a non-persistent buffer (recomputed from
   config, not saved in state_dict).

3. `apply_rotary_pos_emb` / `rotate_half`: apply the rotation to (q, k)
   tensors. Matches the HF implementation — the `emb = cat([freqs, freqs])`
   duplication is what makes the rotation work on paired channels
   without needing complex numbers.
"""

from __future__ import annotations

import jax.numpy as jnp
from flax import nnx


def compute_default_inv_freq(head_dim: int, rope_theta: float) -> jnp.ndarray:
    """1 / (base^(2i/dim)) for i in [0, dim/2). Shape: (head_dim/2,)."""
    indices = jnp.arange(0, head_dim, 2, dtype=jnp.float32)
    return 1.0 / (rope_theta ** (indices / head_dim))


def rotate_half(x: jnp.ndarray) -> jnp.ndarray:
    """Rotate half of the last-dim channels: cat([-x2, x1]) where x=[x1 | x2]."""
    d = x.shape[-1]
    x1, x2 = x[..., : d // 2], x[..., d // 2 :]
    return jnp.concatenate([-x2, x1], axis=-1)


def apply_rotary_pos_emb(
    q: jnp.ndarray,
    k: jnp.ndarray,
    cos: jnp.ndarray,
    sin: jnp.ndarray,
    *,
    unsqueeze_dim: int = 1,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Apply RoPE to q and k.

    Shapes:
    - q, k: (B, H, T, head_dim)   (with unsqueeze_dim=1, default)
            or (B, T, H, head_dim) (with unsqueeze_dim=2)
    - cos, sin: (B, T, head_dim) — unsqueezed along `unsqueeze_dim` for broadcasting
    """
    cos = jnp.expand_dims(cos, axis=unsqueeze_dim)
    sin = jnp.expand_dims(sin, axis=unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class RotaryEmbedding(nnx.Module):
    """Precomputes `inv_freq` at construction, returns (cos, sin) for position_ids.

    Matches HF's Qwen3RotaryEmbedding forward: takes hidden_states (for
    dtype) and position_ids (B, T) and returns cos, sin each of shape
    (B, T, head_dim).
    """

    def __init__(
        self,
        *,
        head_dim: int,
        rope_theta: float = 10000.0,
        attention_scaling: float = 1.0,
        rngs: nnx.Rngs,
    ):
        del rngs  # inv_freq is deterministic from config
        self.head_dim = head_dim
        self.rope_theta = rope_theta
        self.attention_scaling = attention_scaling
        # Non-trainable buffer (HF also registers this non-persistently).
        self.inv_freq = nnx.Variable(compute_default_inv_freq(head_dim, rope_theta))

    def __call__(
        self,
        hidden_states: jnp.ndarray,
        position_ids: jnp.ndarray,  # (B, T)
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        # Do the frequency math in fp32, cast back at the end.
        inv_freq = self.inv_freq[...]  # (head_dim/2,)
        # HF: inv_freq_expanded @ position_ids_expanded → (B, head_dim/2, T)
        # We produce (B, T, head_dim/2) directly via outer-product.
        freqs = position_ids.astype(jnp.float32)[:, :, None] * inv_freq[None, None, :]
        # Duplicate to full head_dim. HF cats (freqs, freqs), NOT interleaves.
        emb = jnp.concatenate([freqs, freqs], axis=-1)  # (B, T, head_dim)
        cos = jnp.cos(emb) * self.attention_scaling
        sin = jnp.sin(emb) * self.attention_scaling
        return cos.astype(hidden_states.dtype), sin.astype(hidden_states.dtype)
