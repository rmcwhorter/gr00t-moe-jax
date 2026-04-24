"""Direct unit tests for the weight-converter helpers.

The parity tests use these helpers indirectly — if a helper is broken,
those tests would still fail, but the failure would be at the module
level and harder to diagnose. These direct tests exercise each helper in
isolation so a bug in the converter surfaces at the helper test first.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from gr00t_moe_jax.modules.category_specific import CategorySpecificLinear
from gr00t_moe_jax.weights.converter import (
    copy_embedding_param,
    copy_layernorm,
    copy_linear,
    torch_to_jax,
)

# Skip entire module if torch isn't installed (lets the shape-only suite
# still run in minimal `uv sync --extra dev` environments).
torch = pytest.importorskip("torch")


def test_torch_to_jax_roundtrip():
    t = torch.randn(3, 5)
    j = torch_to_jax(t)
    assert isinstance(j, jnp.ndarray)
    np.testing.assert_array_equal(np.asarray(j), t.numpy())


def test_copy_linear_produces_matching_output():
    torch.manual_seed(0)
    pt = torch.nn.Linear(8, 12)
    pt.eval()
    jax_ln = nnx.Linear(8, 12, rngs=nnx.Rngs(0))
    copy_linear(pt, jax_ln)

    x_np = np.random.default_rng(0).standard_normal((4, 8)).astype(np.float32)
    with torch.no_grad():
        pt_out = pt(torch.from_numpy(x_np)).numpy()
    jax_out = np.asarray(jax_ln(jnp.asarray(x_np)))
    np.testing.assert_allclose(jax_out, pt_out, rtol=1e-5, atol=1e-5)


def test_copy_linear_no_bias():
    torch.manual_seed(0)
    pt = torch.nn.Linear(8, 12, bias=False)
    jax_ln = nnx.Linear(8, 12, use_bias=False, rngs=nnx.Rngs(0))
    copy_linear(pt, jax_ln)  # should not error

    x_np = np.random.default_rng(0).standard_normal((2, 8)).astype(np.float32)
    with torch.no_grad():
        pt_out = pt(torch.from_numpy(x_np)).numpy()
    jax_out = np.asarray(jax_ln(jnp.asarray(x_np)))
    np.testing.assert_allclose(jax_out, pt_out, rtol=1e-5, atol=1e-5)


def test_copy_layernorm_affine():
    torch.manual_seed(0)
    pt = torch.nn.LayerNorm(16)  # elementwise_affine=True by default
    # Randomize the params so we actually test they're copied.
    with torch.no_grad():
        pt.weight.copy_(torch.randn(16))
        pt.bias.copy_(torch.randn(16))

    jax_ln = nnx.LayerNorm(16, rngs=nnx.Rngs(0))
    copy_layernorm(pt, jax_ln)

    x_np = np.random.default_rng(1).standard_normal((3, 5, 16)).astype(np.float32)
    with torch.no_grad():
        pt_out = pt(torch.from_numpy(x_np)).numpy()
    jax_out = np.asarray(jax_ln(jnp.asarray(x_np)))
    # LayerNorm with large random scale/bias amplifies fp32 differences between
    # PT and JAX implementations (they use different reduction orders). 1e-4
    # is still well below "anything a model would care about".
    np.testing.assert_allclose(jax_out, pt_out, rtol=1e-4, atol=1e-4)


def test_copy_layernorm_non_affine_is_noop():
    """Source has no scale/bias (elementwise_affine=False); dest also has none.
    copy_layernorm should not crash."""
    pt = torch.nn.LayerNorm(8, elementwise_affine=False)
    jax_ln = nnx.LayerNorm(8, use_bias=False, use_scale=False, rngs=nnx.Rngs(0))
    copy_layernorm(pt, jax_ln)  # should not error

    x_np = np.random.default_rng(2).standard_normal((2, 8)).astype(np.float32)
    with torch.no_grad():
        pt_out = pt(torch.from_numpy(x_np)).numpy()
    jax_out = np.asarray(jax_ln(jnp.asarray(x_np)))
    np.testing.assert_allclose(jax_out, pt_out, rtol=1e-5, atol=1e-5)


def test_copy_embedding_param():
    torch.manual_seed(0)
    pt_embed = torch.nn.Embedding(num_embeddings=10, embedding_dim=4)

    # Build a matching JAX nnx.Param and copy.
    jax_param = nnx.Param(jnp.zeros((10, 4), dtype=jnp.float32))
    copy_embedding_param(pt_embed.weight, jax_param)

    # Compare all positions.
    for i in range(10):
        with torch.no_grad():
            pt_row = pt_embed.weight[i].numpy()
        jax_row = np.asarray(jax_param[...][i])
        np.testing.assert_allclose(jax_row, pt_row, rtol=1e-6)


def test_copy_categoryspecific_linear_via_param_assignment():
    """Converter doesn't have a dedicated helper; we assign W/b directly.
    Lock down the expected shape + value convention."""
    num_cat, in_dim, out_dim = 3, 4, 5

    # PyTorch-side weights (random).
    torch.manual_seed(0)
    pt_W = torch.randn(num_cat, in_dim, out_dim)
    pt_b = torch.randn(num_cat, out_dim)

    jax_layer = CategorySpecificLinear(
        num_categories=num_cat, in_features=in_dim, out_features=out_dim, rngs=nnx.Rngs(0)
    )
    jax_layer.W[...] = torch_to_jax(pt_W)
    jax_layer.b[...] = torch_to_jax(pt_b)

    # Verify by running both sides through a forward pass — compute the PT
    # equivalent by hand since we don't want to import the reference here.
    x_np = np.random.default_rng(9).standard_normal((3, 7, in_dim)).astype(np.float32)
    cat_np = np.array([0, 2, 1], dtype=np.int64)

    # PyTorch ground truth: W[cat] @ x + b[cat] per batch item.
    pt_out = np.stack(
        [
            x_np[i] @ pt_W[cat_np[i]].numpy() + pt_b[cat_np[i]].numpy()
            for i in range(3)
        ]
    )
    jax_out = np.asarray(jax_layer(jnp.asarray(x_np), jnp.asarray(cat_np)))
    np.testing.assert_allclose(jax_out, pt_out, rtol=1e-5, atol=1e-5)
