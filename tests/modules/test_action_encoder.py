import jax.numpy as jnp
import numpy as np
from flax import nnx

from gr00t_moe_jax.modules.action_encoder import MultiEmbodimentActionEncoder


def test_output_shape():
    enc = MultiEmbodimentActionEncoder(
        action_dim=132,
        hidden_size=1536,
        num_embodiments=32,
        rngs=nnx.Rngs(0),
    )
    actions = jnp.ones((2, 40, 132))  # GR00T's 40-step chunk, 132 max DoF
    timesteps = jnp.array([500, 250])  # discretized timesteps
    cat_ids = jnp.array([0, 5])

    out = enc(actions, timesteps, cat_ids)
    assert out.shape == (2, 40, 1536)


def test_different_embodiments_different_outputs():
    enc = MultiEmbodimentActionEncoder(
        action_dim=7, hidden_size=32, num_embodiments=4, rngs=nnx.Rngs(123)
    )
    actions = jnp.ones((2, 5, 7))
    timesteps = jnp.array([100, 100])

    out_a = enc(actions, timesteps, jnp.array([0, 0]))
    out_b = enc(actions, timesteps, jnp.array([1, 1]))
    assert not np.allclose(np.asarray(out_a), np.asarray(out_b))


def test_different_timesteps_different_outputs():
    enc = MultiEmbodimentActionEncoder(
        action_dim=7, hidden_size=32, num_embodiments=4, rngs=nnx.Rngs(0)
    )
    actions = jnp.ones((2, 5, 7))
    cat_ids = jnp.array([0, 0])

    out_a = enc(actions, jnp.array([0, 0]), cat_ids)
    out_b = enc(actions, jnp.array([500, 500]), cat_ids)
    assert not np.allclose(np.asarray(out_a), np.asarray(out_b))
