"""Parity test for CategorySpecificLinear/MLP against GR00T's reference.

Both sides use the same (num_categories, in, out) W shape and bmm-equivalent
semantics, so no transpose is needed. Just direct param copy.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
import torch
from flax import nnx

from gr00t_moe_jax.modules.category_specific import (
    CategorySpecificLinear,
    CategorySpecificMLP,
)
from gr00t_moe_jax.weights.converter import torch_to_jax


def _get_torch_cat_modules():
    from tests.parity.conftest import gr00t_src_available, load_gr00t_embodiment_mlp

    if not gr00t_src_available():
        pytest.skip("Isaac-GR00T source directory not found")
    try:
        mod = load_gr00t_embodiment_mlp()
    except Exception as e:
        pytest.skip(f"Cannot load embodiment_conditioned_mlp.py: {e}")
    return mod.CategorySpecificLinear, mod.CategorySpecificMLP


def _copy_cat_linear(pt_mod, jax_mod: CategorySpecificLinear) -> None:
    jax_mod.W[...] = torch_to_jax(pt_mod.W)
    jax_mod.b[...] = torch_to_jax(pt_mod.b)


def _copy_cat_mlp(pt_mod, jax_mod: CategorySpecificMLP) -> None:
    _copy_cat_linear(pt_mod.layer1, jax_mod.layer1)
    _copy_cat_linear(pt_mod.layer2, jax_mod.layer2)


def test_category_specific_linear_parity():
    TorchCat, _ = _get_torch_cat_modules()

    torch.manual_seed(0)
    pt_mod = TorchCat(num_categories=4, input_dim=8, hidden_dim=16)
    pt_mod.eval()

    jax_mod = CategorySpecificLinear(
        num_categories=4, in_features=8, out_features=16, rngs=nnx.Rngs(0)
    )
    _copy_cat_linear(pt_mod, jax_mod)

    rng = np.random.default_rng(0)
    x_np = rng.standard_normal((3, 5, 8)).astype(np.float32)
    cat_np = np.array([0, 2, 1], dtype=np.int64)

    with torch.no_grad():
        pt_out = pt_mod(torch.from_numpy(x_np), torch.from_numpy(cat_np)).numpy()
    jax_out = np.asarray(jax_mod(jnp.asarray(x_np), jnp.asarray(cat_np)))

    np.testing.assert_allclose(jax_out, pt_out, rtol=1e-5, atol=1e-5)


def test_category_specific_mlp_parity():
    _, TorchCatMLP = _get_torch_cat_modules()

    torch.manual_seed(3)
    pt_mod = TorchCatMLP(num_categories=3, input_dim=8, hidden_dim=32, output_dim=4)
    pt_mod.eval()

    jax_mod = CategorySpecificMLP(
        num_categories=3,
        in_features=8,
        hidden_features=32,
        out_features=4,
        rngs=nnx.Rngs(0),
    )
    _copy_cat_mlp(pt_mod, jax_mod)

    rng = np.random.default_rng(5)
    x_np = rng.standard_normal((2, 7, 8)).astype(np.float32)
    cat_np = np.array([0, 2], dtype=np.int64)

    with torch.no_grad():
        pt_out = pt_mod(torch.from_numpy(x_np), torch.from_numpy(cat_np)).numpy()
    jax_out = np.asarray(jax_mod(jnp.asarray(x_np), jnp.asarray(cat_np)))

    np.testing.assert_allclose(jax_out, pt_out, rtol=1e-5, atol=1e-5)
