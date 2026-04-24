"""Parity test for BasicTransformerBlock — the composite of AdaLayerNorm +
Attention + LayerNorm + FeedForward. If this passes, we know the wiring
between these four modules is correct end-to-end.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
import torch
from flax import nnx

from gr00t_moe_jax.modules.transformer_block import BasicTransformerBlock
from gr00t_moe_jax.weights.converter import copy_linear, torch_to_jax


def _get_torch_block():
    from tests.parity.conftest import gr00t_src_available, load_gr00t_dit

    if not gr00t_src_available():
        pytest.skip("Isaac-GR00T source directory not found")
    try:
        mod = load_gr00t_dit()
    except Exception as e:
        pytest.skip(f"Cannot load dit.py: {e}")
    return mod.BasicTransformerBlock


def _copy_block(pt_mod, jax_mod: BasicTransformerBlock) -> None:
    # norm1 (AdaLayerNorm): has .linear, .norm. norm is non-affine in GR00T.
    copy_linear(pt_mod.norm1.linear, jax_mod.norm1.linear)
    # attn1 (Attention)
    copy_linear(pt_mod.attn1.to_q, jax_mod.attn1.to_q)
    copy_linear(pt_mod.attn1.to_k, jax_mod.attn1.to_k)
    copy_linear(pt_mod.attn1.to_v, jax_mod.attn1.to_v)
    copy_linear(pt_mod.attn1.to_out[0], jax_mod.attn1.out_proj)
    # norm3 (LayerNorm, affine=True by default)
    if pt_mod.norm3.weight is not None and jax_mod.norm3.scale is not None:
        jax_mod.norm3.scale[...] = torch_to_jax(pt_mod.norm3.weight)
    if pt_mod.norm3.bias is not None and jax_mod.norm3.bias is not None:
        jax_mod.norm3.bias[...] = torch_to_jax(pt_mod.norm3.bias)
    # ff: gelu-approximate variant
    copy_linear(pt_mod.ff.net[0].proj, jax_mod.ff.proj_in)
    copy_linear(pt_mod.ff.net[2], jax_mod.ff.proj_out)


def test_cross_attention_block_parity():
    TorchBlock = _get_torch_block()

    dim = 64
    heads = 4
    head_dim = 16
    cross_dim = 128

    torch.manual_seed(0)
    pt_mod = TorchBlock(
        dim=dim,
        num_attention_heads=heads,
        attention_head_dim=head_dim,
        norm_type="ada_norm",
        cross_attention_dim=cross_dim,
        activation_fn="gelu-approximate",
    )
    pt_mod.eval()

    jax_mod = BasicTransformerBlock(
        dim=dim,
        num_attention_heads=heads,
        attention_head_dim=head_dim,
        cross_attention_dim=cross_dim,
        activation_fn="gelu-approximate",
        rngs=nnx.Rngs(0),
    )
    _copy_block(pt_mod, jax_mod)

    rng = np.random.default_rng(42)
    x_np = rng.standard_normal((2, 9, dim)).astype(np.float32)
    enc_np = rng.standard_normal((2, 11, cross_dim)).astype(np.float32)
    temb_np = rng.standard_normal((2, dim)).astype(np.float32)

    with torch.no_grad():
        pt_out = pt_mod(
            torch.from_numpy(x_np),
            encoder_hidden_states=torch.from_numpy(enc_np),
            temb=torch.from_numpy(temb_np),
        ).numpy()
    jax_out = np.asarray(
        jax_mod(
            jnp.asarray(x_np),
            temb=jnp.asarray(temb_np),
            encoder_hidden_states=jnp.asarray(enc_np),
        )
    )

    np.testing.assert_allclose(jax_out, pt_out, rtol=1e-5, atol=1e-5)


def test_self_attention_block_parity():
    TorchBlock = _get_torch_block()

    dim = 64
    heads = 4
    head_dim = 16

    torch.manual_seed(1)
    pt_mod = TorchBlock(
        dim=dim,
        num_attention_heads=heads,
        attention_head_dim=head_dim,
        norm_type="ada_norm",
        cross_attention_dim=None,  # self-attn
        activation_fn="gelu-approximate",
    )
    pt_mod.eval()

    jax_mod = BasicTransformerBlock(
        dim=dim,
        num_attention_heads=heads,
        attention_head_dim=head_dim,
        cross_attention_dim=None,
        activation_fn="gelu-approximate",
        rngs=nnx.Rngs(0),
    )
    _copy_block(pt_mod, jax_mod)

    rng = np.random.default_rng(7)
    x_np = rng.standard_normal((2, 9, dim)).astype(np.float32)
    temb_np = rng.standard_normal((2, dim)).astype(np.float32)

    with torch.no_grad():
        pt_out = pt_mod(
            torch.from_numpy(x_np),
            encoder_hidden_states=None,
            temb=torch.from_numpy(temb_np),
        ).numpy()
    jax_out = np.asarray(jax_mod(jnp.asarray(x_np), temb=jnp.asarray(temb_np)))

    np.testing.assert_allclose(jax_out, pt_out, rtol=1e-5, atol=1e-5)
