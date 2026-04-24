"""Parity test for MultiEmbodimentActionEncoder — composition of three
CategorySpecificLinear layers + a SinusoidalPositionalEncoding.

This is a good downstream test: it exercises CategorySpecificLinear in a
composite setting AND the sinusoidal timestep broadcasting, which is the
only non-trivial non-parametric bit.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
import torch
from flax import nnx

from gr00t_moe_jax.modules.action_encoder import MultiEmbodimentActionEncoder
from gr00t_moe_jax.weights.converter import torch_to_jax


def _get_torch_action_encoder():
    from tests.parity.conftest import gr00t_src_available, load_gr00t_embodiment_mlp

    if not gr00t_src_available():
        pytest.skip("Isaac-GR00T source directory not found")
    try:
        mod = load_gr00t_embodiment_mlp()
    except Exception as e:
        pytest.skip(f"Cannot load embodiment_conditioned_mlp.py: {e}")
    return mod.MultiEmbodimentActionEncoder


def _copy_action_encoder(pt_mod, jax_mod: MultiEmbodimentActionEncoder) -> None:
    for attr in ("W1", "W2", "W3"):
        pt_layer = getattr(pt_mod, attr)
        jax_layer = getattr(jax_mod, attr)
        jax_layer.W[...] = torch_to_jax(pt_layer.W)
        jax_layer.b[...] = torch_to_jax(pt_layer.b)


def test_action_encoder_parity():
    TorchMEAE = _get_torch_action_encoder()

    action_dim, hidden, num_emb = 16, 64, 4
    torch.manual_seed(0)
    pt_mod = TorchMEAE(action_dim=action_dim, hidden_size=hidden, num_embodiments=num_emb)
    pt_mod.eval()

    jax_mod = MultiEmbodimentActionEncoder(
        action_dim=action_dim,
        hidden_size=hidden,
        num_embodiments=num_emb,
        rngs=nnx.Rngs(0),
    )
    _copy_action_encoder(pt_mod, jax_mod)

    rng = np.random.default_rng(11)
    actions_np = rng.standard_normal((2, 10, action_dim)).astype(np.float32)
    timesteps_np = np.array([100, 500], dtype=np.float32)
    cat_np = np.array([0, 3], dtype=np.int64)

    with torch.no_grad():
        pt_out = pt_mod(
            torch.from_numpy(actions_np),
            torch.from_numpy(timesteps_np),
            torch.from_numpy(cat_np),
        ).numpy()
    jax_out = np.asarray(
        jax_mod(
            jnp.asarray(actions_np),
            jnp.asarray(timesteps_np),
            jnp.asarray(cat_np),
        )
    )

    np.testing.assert_allclose(jax_out, pt_out, rtol=1e-5, atol=1e-5)
