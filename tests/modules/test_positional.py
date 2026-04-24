import jax.numpy as jnp
import numpy as np

from gr00t_moe_jax.modules.positional import SinusoidalPositionalEncoding


def test_output_shape():
    enc = SinusoidalPositionalEncoding(embedding_dim=64)
    timesteps = jnp.arange(12, dtype=jnp.float32).reshape(3, 4)
    out = enc(timesteps)
    assert out.shape == (3, 4, 64)


def test_sin_cos_concat_structure():
    """First half is sin, second half is cos — matches the PyTorch reference."""
    enc = SinusoidalPositionalEncoding(embedding_dim=32)
    timesteps = jnp.ones((1, 1), dtype=jnp.float32)
    out = np.asarray(enc(timesteps))  # (1, 1, 32)
    sin_half = out[0, 0, :16]
    cos_half = out[0, 0, 16:]
    # sin^2 + cos^2 = 1 at each frequency.
    np.testing.assert_allclose(sin_half**2 + cos_half**2, np.ones(16), atol=1e-6)


def test_zero_timestep_gives_sin_zero_cos_one():
    enc = SinusoidalPositionalEncoding(embedding_dim=32)
    out = np.asarray(enc(jnp.zeros((1, 1), dtype=jnp.float32)))
    np.testing.assert_allclose(out[0, 0, :16], 0.0, atol=1e-7)
    np.testing.assert_allclose(out[0, 0, 16:], 1.0, atol=1e-7)


def test_odd_embedding_dim_rejected():
    """Odd sizes would silently truncate (output = 2·(dim//2)). Reject at init."""
    import pytest

    with pytest.raises(ValueError, match="embedding_dim must be even"):
        SinusoidalPositionalEncoding(embedding_dim=5)
