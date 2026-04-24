"""Top-level parity test for DiT and AlternateVLDiT.

If this passes, the full DiT stack (timestep encoder + 16 alternating
cross/self-attention blocks + final AdaLN output head) matches the
PyTorch reference bit-exactly (within fp32 tolerance).
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
import torch
from flax import nnx

from gr00t_moe_jax.models.dit import AlternateVLDiT, DiT
from gr00t_moe_jax.weights.converter import copy_linear, torch_to_jax


def _get_torch_dit():
    from tests.parity.conftest import gr00t_src_available, load_gr00t_dit

    if not gr00t_src_available():
        pytest.skip("Isaac-GR00T source directory not found")
    try:
        mod = load_gr00t_dit()
    except Exception as e:
        pytest.skip(f"Cannot load dit.py: {e}")
    return mod


def _copy_block(pt_block, jax_block) -> None:
    """Block weight copy (identical to the transformer-block parity test)."""
    copy_linear(pt_block.norm1.linear, jax_block.norm1.linear)
    copy_linear(pt_block.attn1.to_q, jax_block.attn1.to_q)
    copy_linear(pt_block.attn1.to_k, jax_block.attn1.to_k)
    copy_linear(pt_block.attn1.to_v, jax_block.attn1.to_v)
    copy_linear(pt_block.attn1.to_out[0], jax_block.attn1.out_proj)
    if pt_block.norm3.weight is not None and jax_block.norm3.scale is not None:
        jax_block.norm3.scale[...] = torch_to_jax(pt_block.norm3.weight)
    if pt_block.norm3.bias is not None and jax_block.norm3.bias is not None:
        jax_block.norm3.bias[...] = torch_to_jax(pt_block.norm3.bias)
    copy_linear(pt_block.ff.net[0].proj, jax_block.ff.proj_in)
    copy_linear(pt_block.ff.net[2], jax_block.ff.proj_out)


def _copy_dit(pt_mod, jax_mod: DiT) -> None:
    # Timestep encoder: pt.timestep_encoder.timestep_embedder.{linear_1, linear_2}
    copy_linear(pt_mod.timestep_encoder.timestep_embedder.linear_1, jax_mod.timestep_encoder.linear_1)
    copy_linear(pt_mod.timestep_encoder.timestep_embedder.linear_2, jax_mod.timestep_encoder.linear_2)
    # Each transformer block
    for pt_block, jax_block in zip(pt_mod.transformer_blocks, jax_mod.transformer_blocks):
        _copy_block(pt_block, jax_block)
    # norm_out: elementwise_affine=False → no params to copy
    # proj_out_1, proj_out_2
    copy_linear(pt_mod.proj_out_1, jax_mod.proj_out_1)
    copy_linear(pt_mod.proj_out_2, jax_mod.proj_out_2)


def test_alternate_vl_dit_parity():
    """Small but realistic AlternateVLDiT: 4 layers of alternating cross/self
    attention, with image/text mask routing — the exact config pattern
    GR00T N1.7 uses, just scaled down."""
    gr00t_dit = _get_torch_dit()
    TorchAlternateVLDiT = gr00t_dit.AlternateVLDiT

    # Shapes.
    heads, head_dim = 4, 16
    inner_dim = heads * head_dim  # 64
    output_dim = 32
    num_layers = 4
    cross_dim = 128
    B, T, S = 2, 8, 12

    torch.manual_seed(0)
    pt_mod = TorchAlternateVLDiT(
        num_attention_heads=heads,
        attention_head_dim=head_dim,
        output_dim=output_dim,
        num_layers=num_layers,
        cross_attention_dim=cross_dim,
        activation_fn="gelu-approximate",
        norm_type="ada_norm",
        interleave_self_attention=True,
        positional_embeddings=None,
        attend_text_every_n_blocks=2,
    )
    pt_mod.eval()

    jax_mod = AlternateVLDiT(
        num_attention_heads=heads,
        attention_head_dim=head_dim,
        output_dim=output_dim,
        num_layers=num_layers,
        cross_attention_dim=cross_dim,
        activation_fn="gelu-approximate",
        norm_type="ada_norm",
        interleave_self_attention=True,
        attend_text_every_n_blocks=2,
        rngs=nnx.Rngs(0),
    )
    _copy_dit(pt_mod, jax_mod)

    # Inputs.
    rng = np.random.default_rng(7)
    x_np = rng.standard_normal((B, T, inner_dim)).astype(np.float32)
    enc_np = rng.standard_normal((B, S, cross_dim)).astype(np.float32)
    ts_np = np.array([100, 500], dtype=np.int64)
    image_mask_np = np.array([[True] * (S // 2) + [False] * (S // 2)] * B, dtype=bool)
    backbone_mask_np = np.ones((B, S), dtype=bool)

    with torch.no_grad():
        pt_out = pt_mod(
            torch.from_numpy(x_np),
            torch.from_numpy(enc_np),
            torch.from_numpy(ts_np),
            image_mask=torch.from_numpy(image_mask_np),
            backbone_attention_mask=torch.from_numpy(backbone_mask_np),
        ).numpy()

    jax_out = np.asarray(
        jax_mod(
            jnp.asarray(x_np),
            jnp.asarray(enc_np),
            jnp.asarray(ts_np),
            image_mask=jnp.asarray(image_mask_np),
            backbone_attention_mask=jnp.asarray(backbone_mask_np),
        )
    )

    np.testing.assert_allclose(jax_out, pt_out, rtol=1e-5, atol=1e-5)


def test_dit_parity():
    """Plain DiT (no image/text mask alternation)."""
    gr00t_dit = _get_torch_dit()
    TorchDiT = gr00t_dit.DiT

    heads, head_dim = 4, 16
    inner_dim = heads * head_dim
    output_dim = 32
    num_layers = 4
    cross_dim = 128
    B, T, S = 2, 8, 12

    torch.manual_seed(1)
    pt_mod = TorchDiT(
        num_attention_heads=heads,
        attention_head_dim=head_dim,
        output_dim=output_dim,
        num_layers=num_layers,
        cross_attention_dim=cross_dim,
        activation_fn="gelu-approximate",
        norm_type="ada_norm",
        interleave_self_attention=True,
        positional_embeddings=None,
    )
    pt_mod.eval()

    jax_mod = DiT(
        num_attention_heads=heads,
        attention_head_dim=head_dim,
        output_dim=output_dim,
        num_layers=num_layers,
        cross_attention_dim=cross_dim,
        activation_fn="gelu-approximate",
        norm_type="ada_norm",
        interleave_self_attention=True,
        rngs=nnx.Rngs(0),
    )
    _copy_dit(pt_mod, jax_mod)

    rng = np.random.default_rng(3)
    x_np = rng.standard_normal((B, T, inner_dim)).astype(np.float32)
    enc_np = rng.standard_normal((B, S, cross_dim)).astype(np.float32)
    ts_np = np.array([200, 800], dtype=np.int64)

    with torch.no_grad():
        pt_out = pt_mod(
            torch.from_numpy(x_np),
            torch.from_numpy(enc_np),
            torch.from_numpy(ts_np),
        ).numpy()

    jax_out = np.asarray(
        jax_mod(jnp.asarray(x_np), jnp.asarray(enc_np), jnp.asarray(ts_np))
    )

    np.testing.assert_allclose(jax_out, pt_out, rtol=1e-5, atol=1e-5)
