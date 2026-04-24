import jax
import jax.numpy as jnp
from flax import nnx

from gr00t_moe_jax.models.qwen3 import Qwen3DecoderLayer, Qwen3Model
from gr00t_moe_jax.modules.rope import RotaryEmbedding


def _build_rope(head_dim: int = 16):
    return RotaryEmbedding(head_dim=head_dim, rngs=nnx.Rngs(0))


def test_decoder_layer_output_shape():
    layer = Qwen3DecoderLayer(
        hidden_size=64,
        intermediate_size=128,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        rngs=nnx.Rngs(0),
    )
    B, T = 2, 7
    x = jax.random.normal(jax.random.key(0), (B, T, 64))
    pos = jnp.arange(T)[None, :].repeat(B, axis=0)
    rope = _build_rope()
    cos, sin = rope(x, pos)
    out = layer(x, position_embeddings=(cos, sin))
    assert out.shape == (B, T, 64)


def test_qwen3_model_embed_path():
    """Run the full model from input_ids through all layers."""
    model = Qwen3Model(
        vocab_size=100,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        rngs=nnx.Rngs(0),
    )
    B, T = 2, 5
    input_ids = jnp.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
    # Standard causal mask.
    causal = jnp.triu(jnp.full((T, T), -jnp.inf), k=1)[None, None, :, :]
    out = model(input_ids, attention_mask=causal)
    assert out.shape == (B, T, 32)


def test_qwen3_model_select_layer():
    """select_layer=1 → 1 layer applied, NO final norm."""
    model = Qwen3Model(
        vocab_size=20,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=4,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=8,
        rngs=nnx.Rngs(0),
    )
    B, T = 1, 3
    input_ids = jnp.array([[0, 1, 2]])
    out_full = model(input_ids)
    out_truncated = model(input_ids, select_layer=2)
    assert out_full.shape == out_truncated.shape == (B, T, 16)
    # Different because select_layer skips layers 2, 3 and the final norm.
    import numpy as np

    assert not np.allclose(np.asarray(out_full), np.asarray(out_truncated))


def test_inputs_embeds_path():
    """Pass inputs_embeds directly instead of input_ids."""
    model = Qwen3Model(
        vocab_size=10,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=8,
        rngs=nnx.Rngs(0),
    )
    B, T = 1, 4
    embeds = jax.random.normal(jax.random.key(0), (B, T, 16))
    out = model(inputs_embeds=embeds)
    assert out.shape == (B, T, 16)


def test_raises_on_both_inputs():
    import pytest

    model = Qwen3Model(
        vocab_size=10,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=4,
        rngs=nnx.Rngs(0),
    )
    with pytest.raises(ValueError, match="exactly one"):
        model(
            input_ids=jnp.array([[0]]),
            inputs_embeds=jnp.zeros((1, 1, 8)),
        )
