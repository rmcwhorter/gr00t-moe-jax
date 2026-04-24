import jax.numpy as jnp
import numpy as np
from flax import nnx

from gr00t_moe_jax.modules.feedforward import FeedForward


def test_shape_with_default_mult():
    ff = FeedForward(dim=1536, rngs=nnx.Rngs(0))
    x = jnp.ones((2, 41, 1536))
    out = ff(x)
    assert out.shape == (2, 41, 1536)
    assert ff.inner_dim == 4 * 1536


def test_shape_with_custom_inner_dim():
    ff = FeedForward(dim=64, inner_dim=128, rngs=nnx.Rngs(0))
    x = jnp.ones((3, 7, 64))
    out = ff(x)
    assert out.shape == (3, 7, 64)
    assert ff.inner_dim == 128


def test_not_identity():
    # Non-degenerate: ReLU-gated GeGLU on random input shouldn't equal input.
    ff = FeedForward(dim=32, rngs=nnx.Rngs(7))
    import jax
    x = jax.random.normal(jax.random.key(0), (2, 5, 32))
    out = np.asarray(ff(x))
    assert not np.allclose(out, np.asarray(x), atol=1e-3)
