"""Parity test for Qwen3Attention against HF Qwen3Attention."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
import torch
from flax import nnx

from gr00t_moe_jax.modules.qwen3_attention import Qwen3Attention
from gr00t_moe_jax.modules.rope import RotaryEmbedding
from gr00t_moe_jax.weights.converter import copy_linear, torch_to_jax


def _get_torch_bits():
    try:
        from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
        from transformers.models.qwen3.modeling_qwen3 import (
            Qwen3Attention as TorchAttn,
        )
        from transformers.models.qwen3.modeling_qwen3 import (
            Qwen3RotaryEmbedding,
        )
    except Exception as e:
        pytest.skip(f"Cannot import Qwen3 bits: {e}")
    return TorchAttn, Qwen3Config, Qwen3RotaryEmbedding


def _copy_attn(pt_mod, jax_mod: Qwen3Attention) -> None:
    copy_linear(pt_mod.q_proj, jax_mod.q_proj)
    copy_linear(pt_mod.k_proj, jax_mod.k_proj)
    copy_linear(pt_mod.v_proj, jax_mod.v_proj)
    copy_linear(pt_mod.o_proj, jax_mod.o_proj)
    jax_mod.q_norm.weight[...] = torch_to_jax(pt_mod.q_norm.weight)
    jax_mod.k_norm.weight[...] = torch_to_jax(pt_mod.k_norm.weight)


def test_qwen3_attention_parity_mha():
    """Full attention (num_kv_heads == num_heads)."""
    TorchAttn, Qwen3Config, TorchRope = _get_torch_bits()

    hidden_size = 64
    heads = 4
    kv_heads = 4
    head_dim = 16
    cfg = Qwen3Config(
        hidden_size=hidden_size,
        intermediate_size=128,
        num_attention_heads=heads,
        num_key_value_heads=kv_heads,
        num_hidden_layers=1,
        head_dim=head_dim,
        attention_bias=False,
        rms_norm_eps=1e-6,
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0},
    )

    torch.manual_seed(0)
    pt_attn = TorchAttn(cfg, layer_idx=0).eval()
    pt_rope = TorchRope(cfg).eval()

    jax_attn = Qwen3Attention(
        hidden_size=hidden_size,
        num_attention_heads=heads,
        num_key_value_heads=kv_heads,
        head_dim=head_dim,
        attention_bias=False,
        rngs=nnx.Rngs(0),
    )
    _copy_attn(pt_attn, jax_attn)

    jax_rope = RotaryEmbedding(head_dim=head_dim, rope_theta=10000.0, rngs=nnx.Rngs(0))

    B, T = 2, 7
    rng = np.random.default_rng(0)
    x_np = rng.standard_normal((B, T, hidden_size)).astype(np.float32)
    pos_np = np.tile(np.arange(T), (B, 1)).astype(np.int64)

    # Build a standard causal mask (additive).
    causal_np = np.triu(np.full((T, T), -np.inf, dtype=np.float32), k=1)[None, None, :, :]
    causal_np = np.broadcast_to(causal_np, (B, 1, T, T)).copy()

    with torch.no_grad():
        pt_cos, pt_sin = pt_rope(torch.from_numpy(x_np), torch.from_numpy(pos_np))
        pt_out, _ = pt_attn(
            hidden_states=torch.from_numpy(x_np),
            position_embeddings=(pt_cos, pt_sin),
            attention_mask=torch.from_numpy(causal_np),
        )
        pt_out = pt_out.numpy()

    jax_cos, jax_sin = jax_rope(jnp.asarray(x_np), jnp.asarray(pos_np))
    jax_out = np.asarray(
        jax_attn(
            jnp.asarray(x_np),
            position_embeddings=(jax_cos, jax_sin),
            attention_mask=jnp.asarray(causal_np),
        )
    )

    np.testing.assert_allclose(jax_out, pt_out, rtol=1e-5, atol=1e-5)


def test_qwen3_attention_parity_gqa():
    """Grouped-Query Attention (num_kv_heads < num_heads)."""
    TorchAttn, Qwen3Config, TorchRope = _get_torch_bits()

    hidden_size = 96
    heads = 6
    kv_heads = 2  # 3x repeat
    head_dim = 16
    cfg = Qwen3Config(
        hidden_size=hidden_size,
        intermediate_size=192,
        num_attention_heads=heads,
        num_key_value_heads=kv_heads,
        num_hidden_layers=1,
        head_dim=head_dim,
        attention_bias=False,
        rms_norm_eps=1e-6,
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0},
    )

    torch.manual_seed(1)
    pt_attn = TorchAttn(cfg, layer_idx=0).eval()
    pt_rope = TorchRope(cfg).eval()

    jax_attn = Qwen3Attention(
        hidden_size=hidden_size,
        num_attention_heads=heads,
        num_key_value_heads=kv_heads,
        head_dim=head_dim,
        attention_bias=False,
        rngs=nnx.Rngs(0),
    )
    _copy_attn(pt_attn, jax_attn)
    jax_rope = RotaryEmbedding(head_dim=head_dim, rope_theta=10000.0, rngs=nnx.Rngs(0))

    B, T = 2, 9
    rng = np.random.default_rng(3)
    x_np = rng.standard_normal((B, T, hidden_size)).astype(np.float32)
    pos_np = np.tile(np.arange(T), (B, 1)).astype(np.int64)

    causal_np = np.triu(np.full((T, T), -np.inf, dtype=np.float32), k=1)[None, None, :, :]
    causal_np = np.broadcast_to(causal_np, (B, 1, T, T)).copy()

    with torch.no_grad():
        pt_cos, pt_sin = pt_rope(torch.from_numpy(x_np), torch.from_numpy(pos_np))
        pt_out, _ = pt_attn(
            hidden_states=torch.from_numpy(x_np),
            position_embeddings=(pt_cos, pt_sin),
            attention_mask=torch.from_numpy(causal_np),
        )
        pt_out = pt_out.numpy()

    jax_cos, jax_sin = jax_rope(jnp.asarray(x_np), jnp.asarray(pos_np))
    jax_out = np.asarray(
        jax_attn(
            jnp.asarray(x_np),
            position_embeddings=(jax_cos, jax_sin),
            attention_mask=jnp.asarray(causal_np),
        )
    )

    np.testing.assert_allclose(jax_out, pt_out, rtol=1e-5, atol=1e-5)
