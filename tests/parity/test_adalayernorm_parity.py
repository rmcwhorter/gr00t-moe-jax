"""Numerical parity test for AdaLayerNorm against the PyTorch reference.

This is the first end-to-end parity check for the port. If this passes,
the weight-copy convention and the `.value` assignment pattern work,
which unblocks parity tests for everything else.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
import torch
from flax import nnx

from gr00t_moe_jax.modules.adalayernorm import AdaLayerNorm
from gr00t_moe_jax.weights.converter import copy_linear, torch_to_jax


def _get_torch_adalayernorm():
    """Import the PyTorch reference AdaLayerNorm lazily, to skip cleanly if
    the GR00T source path isn't importable."""
    from tests.parity.conftest import gr00t_src_available, load_gr00t_dit

    if not gr00t_src_available():
        pytest.skip("Isaac-GR00T source directory not found")
    try:
        dit_module = load_gr00t_dit()
    except Exception as e:
        pytest.skip(f"Cannot load PyTorch reference dit.py: {e}")
    return dit_module.AdaLayerNorm


def _copy_adalayernorm(pt_mod, jax_mod: AdaLayerNorm) -> None:
    """Copy all params of an AdaLayerNorm. Only `linear` has params in GR00T's
    config (norm_elementwise_affine=False → no norm params)."""
    copy_linear(pt_mod.linear, jax_mod.linear)
    # In GR00T, AdaLayerNorm.norm has elementwise_affine=False → no scale/bias.
    # But our nnx.LayerNorm default has use_bias=True, use_scale=True — let's check.
    # If PT has no weight/bias, skip; our norm should also be non-affine for parity.
    if getattr(pt_mod.norm, "weight", None) is not None and jax_mod.norm.scale is not None:
        jax_mod.norm.scale.value = torch_to_jax(pt_mod.norm.weight)
    if getattr(pt_mod.norm, "bias", None) is not None and jax_mod.norm.bias is not None:
        jax_mod.norm.bias.value = torch_to_jax(pt_mod.norm.bias)


def test_adalayernorm_numerical_parity():
    TorchAdaLayerNorm = _get_torch_adalayernorm()

    embedding_dim = 64

    # Build both. PyTorch defaults match GR00T: elementwise_affine=False.
    torch.manual_seed(0)
    pt_mod = TorchAdaLayerNorm(embedding_dim=embedding_dim)
    pt_mod.eval()

    jax_mod = AdaLayerNorm(embedding_dim=embedding_dim, rngs=nnx.Rngs(0))
    _copy_adalayernorm(pt_mod, jax_mod)

    # Random input.
    B, T = 3, 5
    rng = np.random.default_rng(42)
    x_np = rng.standard_normal((B, T, embedding_dim)).astype(np.float32)
    temb_np = rng.standard_normal((B, embedding_dim)).astype(np.float32)

    # PyTorch forward.
    with torch.no_grad():
        pt_out = pt_mod(torch.from_numpy(x_np), torch.from_numpy(temb_np)).numpy()

    # JAX forward.
    jax_out = np.asarray(jax_mod(jnp.asarray(x_np), jnp.asarray(temb_np)))

    np.testing.assert_allclose(jax_out, pt_out, rtol=1e-5, atol=1e-5)
