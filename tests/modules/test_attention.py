import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from gr00t_moe_jax.modules.attention import Attention


def test_self_attention_shape():
    attn = Attention(query_dim=1536, heads=32, dim_head=48, rngs=nnx.Rngs(0))
    x = jax.random.normal(jax.random.key(0), (2, 41, 1536))
    out = attn(x)
    assert out.shape == (2, 41, 1536)


def test_cross_attention_shape():
    attn = Attention(
        query_dim=1536,
        heads=32,
        dim_head=48,
        cross_attention_dim=2048,
        rngs=nnx.Rngs(0),
    )
    q_in = jax.random.normal(jax.random.key(0), (2, 41, 1536))
    kv_in = jax.random.normal(jax.random.key(1), (2, 256, 2048))
    out = attn(q_in, encoder_hidden_states=kv_in)
    assert out.shape == (2, 41, 1536)


def test_mask_zero_keys_attend_to_first():
    """When mask keeps only the first key, attention output should equal v_out
    from that first key projected — verifies masking semantics."""
    attn = Attention(query_dim=8, heads=2, dim_head=4, rngs=nnx.Rngs(0))
    x = jax.random.normal(jax.random.key(0), (1, 3, 8))
    kv = jax.random.normal(jax.random.key(1), (1, 5, 8))
    mask = jnp.array([[True, False, False, False, False]])  # (1, 5)
    out = np.asarray(attn(x, encoder_hidden_states=kv, attention_mask=mask))
    # If only key 0 survives the mask, softmax is one-hot on it.
    # Then out = v_proj(kv[0]) reshaped — output shouldn't depend on kv[1:].
    kv_alt = kv.at[:, 1:, :].set(jnp.zeros((1, 4, 8)))
    out_alt = np.asarray(attn(x, encoder_hidden_states=kv_alt, attention_mask=mask))
    np.testing.assert_allclose(out, out_alt, atol=1e-5)


def test_different_inputs_different_outputs():
    attn = Attention(query_dim=16, heads=2, dim_head=8, rngs=nnx.Rngs(42))
    key = jax.random.key(0)
    x_a = jax.random.normal(key, (1, 4, 16))
    x_b = jax.random.normal(jax.random.key(1), (1, 4, 16))
    out_a = np.asarray(attn(x_a))
    out_b = np.asarray(attn(x_b))
    assert not np.allclose(out_a, out_b)
