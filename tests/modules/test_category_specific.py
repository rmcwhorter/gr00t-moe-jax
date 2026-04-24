import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from gr00t_moe_jax.modules.category_specific import (
    CategorySpecificLinear,
    CategorySpecificMLP,
)


def test_linear_forward_shape_3d():
    layer = CategorySpecificLinear(
        num_categories=4, in_features=8, out_features=16, rngs=nnx.Rngs(0)
    )
    x = jnp.ones((3, 5, 8))
    cat_ids = jnp.array([0, 2, 1])
    out = layer(x, cat_ids)
    assert out.shape == (3, 5, 16)


def test_linear_forward_shape_2d():
    layer = CategorySpecificLinear(
        num_categories=4, in_features=8, out_features=16, rngs=nnx.Rngs(0)
    )
    x = jnp.ones((3, 8))
    cat_ids = jnp.array([0, 2, 1])
    out = layer(x, cat_ids)
    assert out.shape == (3, 16)


def test_linear_different_categories_produce_different_outputs():
    layer = CategorySpecificLinear(
        num_categories=4, in_features=8, out_features=16, rngs=nnx.Rngs(42)
    )
    x = jnp.ones((2, 5, 8))
    out_0 = layer(x, jnp.array([0, 0]))
    out_1 = layer(x, jnp.array([1, 1]))
    assert not np.allclose(np.asarray(out_0), np.asarray(out_1))


def test_linear_bias_init_zero():
    layer = CategorySpecificLinear(
        num_categories=4, in_features=8, out_features=16, rngs=nnx.Rngs(0)
    )
    np.testing.assert_array_equal(np.asarray(layer.b[...]), np.zeros((4, 16)))


def test_mlp_forward_shape():
    mlp = CategorySpecificMLP(
        num_categories=3,
        in_features=8,
        hidden_features=32,
        out_features=4,
        rngs=nnx.Rngs(0),
    )
    x = jnp.ones((2, 7, 8))
    out = mlp(x, jnp.array([0, 2]))
    assert out.shape == (2, 7, 4)


def test_linear_grad_flows():
    layer = CategorySpecificLinear(
        num_categories=3, in_features=4, out_features=5, rngs=nnx.Rngs(0)
    )

    def loss_fn(m: CategorySpecificLinear):
        x = jnp.ones((2, 4))
        cat_ids = jnp.array([0, 1])
        return m(x, cat_ids).sum()

    grads = nnx.grad(loss_fn)(layer)
    # Grad pytree should have non-zero W/b grads for the categories we used.
    grad_W = grads.W[...]
    assert jnp.abs(grad_W[0]).sum() > 0
    assert jnp.abs(grad_W[1]).sum() > 0
    # Unused category 2 should have zero gradient.
    np.testing.assert_allclose(np.asarray(grad_W[2]), np.zeros_like(grad_W[2]))
