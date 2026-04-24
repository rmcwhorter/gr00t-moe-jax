"""Parity test for Qwen3MLP against HF Qwen3MLP."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
import torch
from flax import nnx

from gr00t_moe_jax.modules.qwen3_mlp import Qwen3MLP
from gr00t_moe_jax.weights.converter import copy_linear


def _get_torch_mlp_and_config():
    try:
        from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
        from transformers.models.qwen3.modeling_qwen3 import Qwen3MLP as TorchMLP
    except Exception as e:
        pytest.skip(f"Cannot import Qwen3MLP: {e}")
    return TorchMLP, Qwen3Config


def test_qwen3_mlp_parity():
    TorchMLP, Qwen3Config = _get_torch_mlp_and_config()

    hidden_size, intermediate_size = 64, 256
    cfg = Qwen3Config(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_hidden_layers=2,  # minimum needed for config validity
        num_attention_heads=4,
        num_key_value_heads=4,
        head_dim=16,
        hidden_act="silu",
    )

    torch.manual_seed(0)
    pt_mod = TorchMLP(cfg).eval()

    jax_mod = Qwen3MLP(
        hidden_size=hidden_size, intermediate_size=intermediate_size, rngs=nnx.Rngs(0)
    )
    copy_linear(pt_mod.gate_proj, jax_mod.gate_proj)
    copy_linear(pt_mod.up_proj, jax_mod.up_proj)
    copy_linear(pt_mod.down_proj, jax_mod.down_proj)

    rng = np.random.default_rng(7)
    x_np = rng.standard_normal((2, 9, hidden_size)).astype(np.float32)

    with torch.no_grad():
        pt_out = pt_mod(torch.from_numpy(x_np)).numpy()
    jax_out = np.asarray(jax_mod(jnp.asarray(x_np)))

    np.testing.assert_allclose(jax_out, pt_out, rtol=1e-5, atol=1e-5)
