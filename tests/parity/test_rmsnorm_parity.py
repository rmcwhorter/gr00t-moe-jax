"""Parity test for RMSNorm against HF Qwen3RMSNorm."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
import torch
from flax import nnx

from gr00t_moe_jax.modules.rmsnorm import RMSNorm
from gr00t_moe_jax.weights.converter import torch_to_jax


def _get_torch_rmsnorm():
    try:
        from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm
    except Exception as e:
        pytest.skip(f"Cannot import Qwen3RMSNorm: {e}")
    return Qwen3RMSNorm


def test_rmsnorm_parity():
    TorchRMSNorm = _get_torch_rmsnorm()

    hidden_size = 128
    torch.manual_seed(0)
    pt_mod = TorchRMSNorm(hidden_size=hidden_size, eps=1e-6)
    # Randomize the weight so we test both paths (rsqrt + scale).
    with torch.no_grad():
        pt_mod.weight.copy_(torch.randn(hidden_size) * 0.5 + 1.0)

    jax_mod = RMSNorm(hidden_size=hidden_size, eps=1e-6, rngs=nnx.Rngs(0))
    jax_mod.weight[...] = torch_to_jax(pt_mod.weight)

    rng = np.random.default_rng(42)
    x_np = rng.standard_normal((2, 7, hidden_size)).astype(np.float32)

    with torch.no_grad():
        pt_out = pt_mod(torch.from_numpy(x_np)).numpy()
    jax_out = np.asarray(jax_mod(jnp.asarray(x_np)))

    np.testing.assert_allclose(jax_out, pt_out, rtol=1e-5, atol=1e-5)
