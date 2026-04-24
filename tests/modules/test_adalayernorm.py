import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from gr00t_moe_jax.modules.adalayernorm import AdaLayerNorm


def test_output_shape():
    ln = AdaLayerNorm(embedding_dim=1536, rngs=nnx.Rngs(0))
    x = jnp.ones((2, 41, 1536))
    temb = jnp.ones((2, 1536))
    out = ln(x, temb)
    assert out.shape == (2, 41, 1536)


def test_zero_temb_gives_approx_identity_after_norm():
    """With temb=0: silu(0)=0, Linear(0)=linear_bias. By default Linear bias
    is initialized to zero, so scale=shift=0 → output = LayerNorm(x)."""
    ln = AdaLayerNorm(embedding_dim=16, rngs=nnx.Rngs(0))
    # Unit-variance input
    key = jax.random.key(42)
    x = jax.random.normal(key, (2, 5, 16))
    temb = jnp.zeros((2, 16))
    out = np.asarray(ln(x, temb))
    # Mean across the feature axis should be ~0 (LayerNorm's effect).
    mean = out.mean(axis=-1)
    np.testing.assert_allclose(mean, np.zeros_like(mean), atol=1e-5)


def test_scale_shift_separation():
    """Giving different temb per batch item should produce independent outputs."""
    ln = AdaLayerNorm(embedding_dim=16, rngs=nnx.Rngs(1))
    x = jnp.ones((2, 5, 16))
    temb = jnp.stack([jnp.zeros(16), jnp.ones(16)], axis=0)
    out = np.asarray(ln(x, temb))
    # Different timesteps should produce different outputs.
    assert not np.allclose(out[0], out[1])
