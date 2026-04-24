"""Parity test for FeedForward against diffusers' implementation with
activation_fn='gelu-approximate' (which is what GR00T uses).

Diffusers FF stores weights as:
    net.0.proj.weight/bias  (the GELU's inner projection: dim → inner_dim)
    net.2.weight/bias       (the output projection: inner_dim → dim_out)
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
import torch
from flax import nnx

from gr00t_moe_jax.modules.feedforward import FeedForward
from gr00t_moe_jax.weights.converter import copy_linear


def _get_torch_feedforward():
    try:
        from diffusers.models.attention import FeedForward as TorchFeedForward
    except Exception as e:
        pytest.skip(f"Cannot import diffusers FeedForward: {e}")
    return TorchFeedForward


def _copy_ff(pt_mod, jax_mod: FeedForward) -> None:
    # net[0] is a GELU module with .proj; net[2] is the output Linear.
    copy_linear(pt_mod.net[0].proj, jax_mod.proj_in)
    copy_linear(pt_mod.net[2], jax_mod.proj_out)


def test_feedforward_gelu_approximate_parity():
    TorchFF = _get_torch_feedforward()

    dim = 64
    mult = 4  # inner_dim = 256 (matches diffusers default)
    torch.manual_seed(1)
    pt_mod = TorchFF(dim, mult=mult, activation_fn="gelu-approximate")
    pt_mod.eval()

    jax_mod = FeedForward(dim=dim, mult=mult, activation_fn="gelu-approximate", rngs=nnx.Rngs(0))
    _copy_ff(pt_mod, jax_mod)

    rng = np.random.default_rng(123)
    x_np = rng.standard_normal((2, 7, dim)).astype(np.float32)

    with torch.no_grad():
        pt_out = pt_mod(torch.from_numpy(x_np)).numpy()
    jax_out = np.asarray(jax_mod(jnp.asarray(x_np)))

    np.testing.assert_allclose(jax_out, pt_out, rtol=1e-5, atol=1e-5)


def test_feedforward_geglu_parity():
    """Also test GeGLU path (not used by GR00T, but we support it)."""
    TorchFF = _get_torch_feedforward()

    dim = 32
    mult = 2
    torch.manual_seed(2)
    pt_mod = TorchFF(dim, mult=mult, activation_fn="geglu")
    pt_mod.eval()

    jax_mod = FeedForward(dim=dim, mult=mult, activation_fn="geglu", rngs=nnx.Rngs(0))
    _copy_ff(pt_mod, jax_mod)  # GeGLU inner is also .proj

    rng = np.random.default_rng(7)
    x_np = rng.standard_normal((2, 5, dim)).astype(np.float32)

    with torch.no_grad():
        pt_out = pt_mod(torch.from_numpy(x_np)).numpy()
    jax_out = np.asarray(jax_mod(jnp.asarray(x_np)))

    np.testing.assert_allclose(jax_out, pt_out, rtol=1e-5, atol=1e-5)
