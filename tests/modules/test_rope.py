import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from gr00t_moe_jax.modules.rope import (
    RotaryEmbedding,
    apply_rotary_pos_emb,
    compute_default_inv_freq,
    rotate_half,
)


def test_inv_freq_shape_and_monotonic():
    inv_freq = compute_default_inv_freq(head_dim=64, rope_theta=10000.0)
    assert inv_freq.shape == (32,)
    # Frequencies decrease monotonically.
    diffs = np.diff(np.asarray(inv_freq))
    assert (diffs < 0).all()


def test_rotate_half():
    x = jnp.array([[1.0, 2.0, 3.0, 4.0]])
    out = np.asarray(rotate_half(x))
    np.testing.assert_array_equal(out, [[-3.0, -4.0, 1.0, 2.0]])


def test_rotary_embedding_output_shapes():
    rope = RotaryEmbedding(head_dim=64, rngs=nnx.Rngs(0))
    B, T = 2, 10
    hidden = jnp.zeros((B, T, 64))
    position_ids = jnp.arange(T)[None, :].repeat(B, axis=0)
    cos, sin = rope(hidden, position_ids)
    assert cos.shape == (B, T, 64)
    assert sin.shape == (B, T, 64)


def test_apply_rotary_pos_emb_shape():
    B, H, T, D = 2, 4, 8, 16
    q = jnp.ones((B, H, T, D))
    k = jnp.ones((B, H, T, D))
    cos = jnp.ones((B, T, D))
    sin = jnp.zeros((B, T, D))  # sin=0 → rotation is identity
    q_out, k_out = apply_rotary_pos_emb(q, k, cos, sin)
    assert q_out.shape == (B, H, T, D)
    # With sin=0, q_out = q * cos = q
    np.testing.assert_allclose(np.asarray(q_out), np.asarray(q))


def test_rope_at_zero_position_is_identity():
    """At position 0, all freqs=0, so cos=1 and sin=0 → apply_rotary is identity."""
    rope = RotaryEmbedding(head_dim=16, rngs=nnx.Rngs(0))
    B, H, T, D = 1, 2, 1, 16
    q = jax.random.normal(jax.random.key(0), (B, H, T, D))
    k = jax.random.normal(jax.random.key(1), (B, H, T, D))
    position_ids = jnp.zeros((B, T), dtype=jnp.int32)
    cos, sin = rope(q, position_ids)
    q_out, k_out = apply_rotary_pos_emb(q, k, cos, sin)
    np.testing.assert_allclose(np.asarray(q_out), np.asarray(q), atol=1e-6)
    np.testing.assert_allclose(np.asarray(k_out), np.asarray(k), atol=1e-6)
