import jax.numpy as jnp
import numpy as np
from flax import nnx

from gr00t_moe_jax.modules.timestep import TimestepEncoder, get_timestep_embedding


def test_raw_embedding_shape_and_flip():
    timesteps = jnp.arange(5, dtype=jnp.float32)
    emb = get_timestep_embedding(timesteps, 256, flip_sin_to_cos=True)
    assert emb.shape == (5, 256)


def test_flip_sin_to_cos_swaps_halves():
    # flip_sin_to_cos=False gives [sin, cos]; True gives [cos, sin].
    timesteps = jnp.array([1.0, 2.0])
    no_flip = np.asarray(get_timestep_embedding(timesteps, 64, flip_sin_to_cos=False))
    flip = np.asarray(get_timestep_embedding(timesteps, 64, flip_sin_to_cos=True))
    np.testing.assert_allclose(no_flip[:, :32], flip[:, 32:])
    np.testing.assert_allclose(no_flip[:, 32:], flip[:, :32])


def test_zero_timestep_has_fixed_value():
    # With t=0: sin(0)=0, cos(0)=1. With flip_sin_to_cos=True we get
    # [cos(0), sin(0)] = [1..1, 0..0].
    emb = np.asarray(get_timestep_embedding(jnp.zeros((1,)), 64, flip_sin_to_cos=True))
    np.testing.assert_allclose(emb[0, :32], np.ones(32), atol=1e-7)
    np.testing.assert_allclose(emb[0, 32:], np.zeros(32), atol=1e-7)


def test_encoder_output_shape():
    enc = TimestepEncoder(embedding_dim=1536, rngs=nnx.Rngs(0))
    out = enc(jnp.arange(4, dtype=jnp.float32))
    assert out.shape == (4, 1536)


def test_encoder_different_timesteps_different_outputs():
    enc = TimestepEncoder(embedding_dim=128, rngs=nnx.Rngs(0))
    a = np.asarray(enc(jnp.array([10.0])))
    b = np.asarray(enc(jnp.array([900.0])))
    assert not np.allclose(a, b)
