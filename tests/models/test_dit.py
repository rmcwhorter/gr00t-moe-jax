import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from gr00t_moe_jax.models.dit import AlternateVLDiT, DiT

# GR00T N1.7 default shapes.
INNER_DIM = 1536  # 32 heads * 48 head_dim
VL_DIM = 2048
OUTPUT_DIM = 1024
ACTION_HORIZON = 40
CHUNK_TOKENS = 1 + ACTION_HORIZON  # state + actions

_SMALL = dict(  # small config for fast tests
    num_attention_heads=4,
    attention_head_dim=16,
    output_dim=32,
    num_layers=4,
    cross_attention_dim=64,
)
_SMALL_INNER = _SMALL["num_attention_heads"] * _SMALL["attention_head_dim"]


def test_dit_output_shape():
    model = DiT(**_SMALL, rngs=nnx.Rngs(0))
    B, T, S = 2, 10, 8
    x = jax.random.normal(jax.random.key(0), (B, T, _SMALL_INNER))
    encoder = jax.random.normal(jax.random.key(1), (B, S, _SMALL["cross_attention_dim"]))
    timestep = jnp.array([100, 500])
    out = model(x, encoder, timestep)
    assert out.shape == (B, T, _SMALL["output_dim"])


def test_dit_n17_scale_shape():
    """Smoke test at the real N1.7 dimensions (will use more memory)."""
    model = DiT(
        num_attention_heads=32,
        attention_head_dim=48,
        output_dim=OUTPUT_DIM,
        num_layers=2,  # 2 instead of 16 to keep test fast
        cross_attention_dim=VL_DIM,
        rngs=nnx.Rngs(0),
    )
    B = 1
    x = jax.random.normal(jax.random.key(0), (B, CHUNK_TOKENS, INNER_DIM))
    encoder = jax.random.normal(jax.random.key(1), (B, 64, VL_DIM))
    timestep = jnp.array([250])
    out = model(x, encoder, timestep)
    assert out.shape == (B, CHUNK_TOKENS, OUTPUT_DIM)


def test_alternate_vl_dit_shape():
    model = AlternateVLDiT(
        **_SMALL, attend_text_every_n_blocks=2, rngs=nnx.Rngs(0)
    )
    B, T, S = 2, 10, 8
    x = jax.random.normal(jax.random.key(0), (B, T, _SMALL_INNER))
    encoder = jax.random.normal(jax.random.key(1), (B, S, _SMALL["cross_attention_dim"]))
    timestep = jnp.array([100, 500])
    # Half image, half text tokens.
    image_mask = jnp.array([[True] * 4 + [False] * 4] * B)
    backbone_mask = jnp.ones((B, S), dtype=jnp.bool_)

    out = model(
        x,
        encoder,
        timestep,
        image_mask=image_mask,
        backbone_attention_mask=backbone_mask,
    )
    assert out.shape == (B, T, _SMALL["output_dim"])


def test_alternate_vl_dit_timestep_affects_output():
    model = AlternateVLDiT(
        **_SMALL, attend_text_every_n_blocks=2, rngs=nnx.Rngs(42)
    )
    B, T, S = 1, 6, 4
    x = jax.random.normal(jax.random.key(0), (B, T, _SMALL_INNER))
    encoder = jax.random.normal(jax.random.key(1), (B, S, _SMALL["cross_attention_dim"]))
    image_mask = jnp.array([[True, True, False, False]])
    backbone_mask = jnp.ones((B, S), dtype=jnp.bool_)

    out_a = np.asarray(
        model(x, encoder, jnp.array([0]), image_mask=image_mask, backbone_attention_mask=backbone_mask)
    )
    out_b = np.asarray(
        model(x, encoder, jnp.array([900]), image_mask=image_mask, backbone_attention_mask=backbone_mask)
    )
    assert not np.allclose(out_a, out_b)


def test_alternate_vl_dit_image_mask_affects_output():
    """Different image/text partitionings should produce different outputs
    (confirms the mask routing is wired correctly)."""
    model = AlternateVLDiT(
        **_SMALL, attend_text_every_n_blocks=2, rngs=nnx.Rngs(7)
    )
    B, T, S = 1, 6, 8
    x = jax.random.normal(jax.random.key(0), (B, T, _SMALL_INNER))
    encoder = jax.random.normal(jax.random.key(1), (B, S, _SMALL["cross_attention_dim"]))
    timestep = jnp.array([100])
    backbone_mask = jnp.ones((B, S), dtype=jnp.bool_)

    mask_a = jnp.array([[True, True, True, True, False, False, False, False]])
    mask_b = jnp.array([[False, False, False, False, True, True, True, True]])

    out_a = np.asarray(model(x, encoder, timestep, image_mask=mask_a, backbone_attention_mask=backbone_mask))
    out_b = np.asarray(model(x, encoder, timestep, image_mask=mask_b, backbone_attention_mask=backbone_mask))
    assert not np.allclose(out_a, out_b)


def test_alternate_vl_dit_requires_interleave():
    import pytest

    with pytest.raises(ValueError, match="interleave_self_attention"):
        AlternateVLDiT(
            num_attention_heads=2,
            attention_head_dim=8,
            output_dim=16,
            num_layers=2,
            cross_attention_dim=32,
            interleave_self_attention=False,
            rngs=nnx.Rngs(0),
        )
