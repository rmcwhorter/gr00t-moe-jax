import jax
import jax.numpy as jnp
from flax import nnx

from gr00t_moe_jax.modules.transformer_block import BasicTransformerBlock

# GR00T N1.7 DiT spec.
DIM = 1536
HEADS = 32
HEAD_DIM = 48
VL_DIM = 2048


def test_self_attention_block_shape():
    block = BasicTransformerBlock(
        dim=DIM,
        num_attention_heads=HEADS,
        attention_head_dim=HEAD_DIM,
        cross_attention_dim=None,  # self-attention
        rngs=nnx.Rngs(0),
    )
    x = jax.random.normal(jax.random.key(0), (2, 41, DIM))
    temb = jax.random.normal(jax.random.key(1), (2, DIM))
    out = block(x, temb=temb)
    assert out.shape == (2, 41, DIM)


def test_cross_attention_block_shape():
    block = BasicTransformerBlock(
        dim=DIM,
        num_attention_heads=HEADS,
        attention_head_dim=HEAD_DIM,
        cross_attention_dim=VL_DIM,
        rngs=nnx.Rngs(0),
    )
    x = jax.random.normal(jax.random.key(0), (2, 41, DIM))
    temb = jax.random.normal(jax.random.key(1), (2, DIM))
    encoder = jax.random.normal(jax.random.key(2), (2, 200, VL_DIM))
    mask = jnp.ones((2, 200), dtype=jnp.bool_)
    out = block(
        x, temb=temb, encoder_hidden_states=encoder, encoder_attention_mask=mask
    )
    assert out.shape == (2, 41, DIM)


def test_unsupported_norm_type_raises():
    import pytest

    with pytest.raises(NotImplementedError):
        BasicTransformerBlock(
            dim=32,
            num_attention_heads=2,
            attention_head_dim=16,
            norm_type="layer_norm",
            rngs=nnx.Rngs(0),
        )
