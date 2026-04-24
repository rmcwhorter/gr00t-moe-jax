import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from gr00t_moe_jax.modules.qwen3_attention import Qwen3Attention, repeat_kv
from gr00t_moe_jax.modules.rope import RotaryEmbedding


def test_repeat_kv_factor_one_is_noop():
    x = jax.random.normal(jax.random.key(0), (2, 4, 5, 8))
    out = repeat_kv(x, 1)
    np.testing.assert_array_equal(np.asarray(out), np.asarray(x))


def test_repeat_kv_shape():
    x = jax.random.normal(jax.random.key(0), (2, 4, 5, 8))
    out = repeat_kv(x, 3)  # 4 * 3 = 12 heads
    assert out.shape == (2, 12, 5, 8)


def test_repeat_kv_correctness():
    """Each KV head is repeated n_rep times contiguously."""
    x = jnp.arange(2 * 2 * 1 * 3).reshape(2, 2, 1, 3).astype(jnp.float32)
    out = np.asarray(repeat_kv(x, 2))
    # Head 0 and head 1 of kv become [h0, h0, h1, h1] in the output.
    np.testing.assert_array_equal(out[0, 0], out[0, 1])
    np.testing.assert_array_equal(out[0, 2], out[0, 3])


def test_attention_output_shape():
    attn = Qwen3Attention(
        hidden_size=64,
        num_attention_heads=4,
        num_key_value_heads=2,  # GQA: 4 Q heads, 2 KV heads
        head_dim=16,
        rngs=nnx.Rngs(0),
    )
    B, T = 2, 6
    x = jax.random.normal(jax.random.key(0), (B, T, 64))
    pos = jnp.arange(T)[None, :].repeat(B, axis=0)
    rope = RotaryEmbedding(head_dim=16, rngs=nnx.Rngs(0))
    cos, sin = rope(x, pos)
    out = attn(x, position_embeddings=(cos, sin))
    assert out.shape == (B, T, 64)


def test_attention_with_causal_mask():
    attn = Qwen3Attention(
        hidden_size=32,
        num_attention_heads=4,
        num_key_value_heads=4,
        head_dim=8,
        rngs=nnx.Rngs(0),
    )
    B, T = 1, 5
    x = jax.random.normal(jax.random.key(0), (B, T, 32))
    pos = jnp.arange(T)[None, :]
    rope = RotaryEmbedding(head_dim=8, rngs=nnx.Rngs(0))
    cos, sin = rope(x, pos)
    # Build standard causal mask: (B, 1, T, T) with -inf above diagonal, 0 elsewhere.
    causal = jnp.triu(jnp.full((T, T), -jnp.inf), k=1)[None, None, :, :]
    out = attn(x, position_embeddings=(cos, sin), attention_mask=causal)
    assert out.shape == (B, T, 32)
