"""Parity test for TimestepEncoder against the GR00T reference
(which internally uses diffusers' Timesteps + TimestepEmbedding).
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
import torch
from flax import nnx

from gr00t_moe_jax.modules.timestep import TimestepEncoder
from gr00t_moe_jax.weights.converter import copy_linear


def _get_torch_timestep_encoder():
    from tests.parity.conftest import gr00t_src_available, load_gr00t_dit

    if not gr00t_src_available():
        pytest.skip("Isaac-GR00T source directory not found")
    try:
        mod = load_gr00t_dit()
    except Exception as e:
        pytest.skip(f"Cannot load dit.py: {e}")
    return mod.TimestepEncoder


def test_timestep_encoder_parity():
    TorchTE = _get_torch_timestep_encoder()

    embedding_dim = 128
    torch.manual_seed(0)
    pt_mod = TorchTE(embedding_dim=embedding_dim)
    pt_mod.eval()

    jax_mod = TimestepEncoder(embedding_dim=embedding_dim, rngs=nnx.Rngs(0))
    copy_linear(pt_mod.timestep_embedder.linear_1, jax_mod.linear_1)
    copy_linear(pt_mod.timestep_embedder.linear_2, jax_mod.linear_2)

    ts_np = np.array([0, 100, 500, 999], dtype=np.int64)
    with torch.no_grad():
        pt_out = pt_mod(torch.from_numpy(ts_np)).numpy()
    jax_out = np.asarray(jax_mod(jnp.asarray(ts_np)))

    np.testing.assert_allclose(jax_out, pt_out, rtol=1e-5, atol=1e-5)
