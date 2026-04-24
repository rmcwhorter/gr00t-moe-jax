"""Parity test for Attention against diffusers' Attention.

Diffusers stores:
    to_q.weight/bias
    to_k.weight/bias
    to_v.weight/bias
    to_out.0.weight/bias   (first element of a ModuleList [Linear, Dropout])
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
import torch
from flax import nnx

from gr00t_moe_jax.modules.attention import Attention
from gr00t_moe_jax.weights.converter import copy_linear


def _get_torch_attention():
    try:
        from diffusers.models.attention_processor import Attention as TorchAttention
    except Exception as e:
        pytest.skip(f"Cannot import diffusers Attention: {e}")
    return TorchAttention


def _copy_attention(pt_mod, jax_mod: Attention) -> None:
    copy_linear(pt_mod.to_q, jax_mod.to_q)
    copy_linear(pt_mod.to_k, jax_mod.to_k)
    copy_linear(pt_mod.to_v, jax_mod.to_v)
    copy_linear(pt_mod.to_out[0], jax_mod.out_proj)


def test_self_attention_parity():
    TorchAttn = _get_torch_attention()

    query_dim, heads, dim_head = 64, 4, 16
    torch.manual_seed(0)
    pt_mod = TorchAttn(query_dim=query_dim, heads=heads, dim_head=dim_head, bias=True, out_bias=True)
    pt_mod.eval()

    jax_mod = Attention(query_dim=query_dim, heads=heads, dim_head=dim_head, rngs=nnx.Rngs(0))
    _copy_attention(pt_mod, jax_mod)

    rng = np.random.default_rng(42)
    x_np = rng.standard_normal((2, 9, query_dim)).astype(np.float32)

    with torch.no_grad():
        pt_out = pt_mod(torch.from_numpy(x_np)).numpy()
    jax_out = np.asarray(jax_mod(jnp.asarray(x_np)))

    np.testing.assert_allclose(jax_out, pt_out, rtol=1e-5, atol=1e-5)


def test_cross_attention_parity():
    TorchAttn = _get_torch_attention()

    query_dim, kv_dim, heads, dim_head = 64, 128, 4, 16
    torch.manual_seed(1)
    pt_mod = TorchAttn(
        query_dim=query_dim,
        cross_attention_dim=kv_dim,
        heads=heads,
        dim_head=dim_head,
        bias=True,
        out_bias=True,
    )
    pt_mod.eval()

    jax_mod = Attention(
        query_dim=query_dim,
        cross_attention_dim=kv_dim,
        heads=heads,
        dim_head=dim_head,
        rngs=nnx.Rngs(0),
    )
    _copy_attention(pt_mod, jax_mod)

    rng = np.random.default_rng(7)
    q_np = rng.standard_normal((2, 9, query_dim)).astype(np.float32)
    kv_np = rng.standard_normal((2, 11, kv_dim)).astype(np.float32)

    with torch.no_grad():
        pt_out = pt_mod(
            torch.from_numpy(q_np), encoder_hidden_states=torch.from_numpy(kv_np)
        ).numpy()
    jax_out = np.asarray(
        jax_mod(jnp.asarray(q_np), encoder_hidden_states=jnp.asarray(kv_np))
    )

    np.testing.assert_allclose(jax_out, pt_out, rtol=1e-5, atol=1e-5)
