import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from gr00t_moe_jax.modules.rmsnorm import RMSNorm


def test_output_shape():
    ln = RMSNorm(hidden_size=16, rngs=nnx.Rngs(0))
    x = jax.random.normal(jax.random.key(0), (3, 5, 16))
    out = ln(x)
    assert out.shape == (3, 5, 16)


def test_ones_weight_rms_normalises_to_unit():
    """With weight=1, output's RMS should be ~1."""
    ln = RMSNorm(hidden_size=32, rngs=nnx.Rngs(0))
    x = jax.random.normal(jax.random.key(0), (4, 32)) * 10.0  # large input
    out = np.asarray(ln(x))
    rms = np.sqrt(np.mean(out**2, axis=-1))
    np.testing.assert_allclose(rms, np.ones_like(rms), atol=1e-3)


def test_weight_scales_output():
    ln = RMSNorm(hidden_size=4, rngs=nnx.Rngs(0))
    ln.weight[...] = jnp.array([2.0, 2.0, 2.0, 2.0])
    x = jnp.ones((1, 4))  # rms=1, so post-rsqrt is all 1s; multiplied by weight=2
    out = np.asarray(ln(x))
    np.testing.assert_allclose(out, 2.0 * np.ones((1, 4)), atol=1e-6)


def test_preserves_input_dtype():
    ln = RMSNorm(hidden_size=8, rngs=nnx.Rngs(0))
    x = jnp.ones((1, 8), dtype=jnp.bfloat16)
    out = ln(x)
    assert out.dtype == jnp.bfloat16
