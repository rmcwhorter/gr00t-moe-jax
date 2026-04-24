"""Parity test for RotaryEmbedding + apply_rotary_pos_emb against HF Qwen3."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
import torch
from flax import nnx

from gr00t_moe_jax.modules.rope import RotaryEmbedding, apply_rotary_pos_emb


def _get_torch_rope():
    try:
        from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
        from transformers.models.qwen3.modeling_qwen3 import (
            Qwen3RotaryEmbedding,
        )
        from transformers.models.qwen3.modeling_qwen3 import (
            apply_rotary_pos_emb as torch_apply_rotary,
        )
    except Exception as e:
        pytest.skip(f"Cannot import Qwen3 RoPE: {e}")
    return Qwen3RotaryEmbedding, Qwen3Config, torch_apply_rotary


def test_rotary_embedding_parity():
    TorchRope, Qwen3Config, _ = _get_torch_rope()

    head_dim = 64
    rope_theta = 10000.0
    cfg = Qwen3Config(
        hidden_size=head_dim * 4,
        num_attention_heads=4,
        num_key_value_heads=4,
        num_hidden_layers=2,
        head_dim=head_dim,
        rope_parameters={"rope_type": "default", "rope_theta": rope_theta},
    )

    pt_rope = TorchRope(cfg).eval()
    jax_rope = RotaryEmbedding(head_dim=head_dim, rope_theta=rope_theta, rngs=nnx.Rngs(0))

    B, T = 2, 12
    pos_np = np.tile(np.arange(T), (B, 1)).astype(np.int64)
    x_np = np.zeros((B, T, head_dim), dtype=np.float32)

    with torch.no_grad():
        pt_cos, pt_sin = pt_rope(torch.from_numpy(x_np), torch.from_numpy(pos_np))
        pt_cos = pt_cos.numpy()
        pt_sin = pt_sin.numpy()

    jax_cos, jax_sin = jax_rope(jnp.asarray(x_np), jnp.asarray(pos_np))
    np.testing.assert_allclose(np.asarray(jax_cos), pt_cos, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(np.asarray(jax_sin), pt_sin, rtol=1e-5, atol=1e-5)


def test_apply_rotary_pos_emb_parity():
    _, _, torch_apply_rotary = _get_torch_rope()

    B, H, T, D = 2, 4, 6, 16
    rng = np.random.default_rng(0)
    q_np = rng.standard_normal((B, H, T, D)).astype(np.float32)
    k_np = rng.standard_normal((B, H, T, D)).astype(np.float32)
    cos_np = rng.standard_normal((B, T, D)).astype(np.float32)
    sin_np = rng.standard_normal((B, T, D)).astype(np.float32)

    with torch.no_grad():
        pt_q, pt_k = torch_apply_rotary(
            torch.from_numpy(q_np),
            torch.from_numpy(k_np),
            torch.from_numpy(cos_np),
            torch.from_numpy(sin_np),
        )

    jax_q, jax_k = apply_rotary_pos_emb(
        jnp.asarray(q_np), jnp.asarray(k_np), jnp.asarray(cos_np), jnp.asarray(sin_np)
    )

    np.testing.assert_allclose(np.asarray(jax_q), pt_q.numpy(), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(np.asarray(jax_k), pt_k.numpy(), rtol=1e-5, atol=1e-5)
