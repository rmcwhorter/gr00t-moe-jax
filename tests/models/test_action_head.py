import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from gr00t_moe_jax.models.action_head import ActionHead

# Small config for fast tests, but matches N1.7 structure.
B = 2
ACTION_HORIZON = 6
STATE_DIM = 16
ACTION_DIM = 16
VL_DIM = 32
VL_SEQ = 20
INPUT_EMB = 64  # = 4 × 16 heads × dim
HEADS = 4
HEAD_DIM = 16


def _build(use_alternate_vl_dit: bool = True, use_vlln: bool = True, add_pos_embed: bool = True):
    return ActionHead(
        max_action_dim=ACTION_DIM,
        max_state_dim=STATE_DIM,
        action_horizon=ACTION_HORIZON,
        hidden_size=48,
        input_embedding_dim=INPUT_EMB,
        backbone_embedding_dim=VL_DIM,
        state_history_length=1,
        max_seq_len=128,
        max_num_embodiments=8,
        dit_num_layers=4,
        dit_num_attention_heads=HEADS,
        dit_attention_head_dim=HEAD_DIM,
        dit_output_dim=48,
        use_alternate_vl_dit=use_alternate_vl_dit,
        use_vlln=use_vlln,
        add_pos_embed=add_pos_embed,
        rngs=nnx.Rngs(0),
    )


def _inputs():
    key = jax.random.key(0)
    return dict(
        noisy_action=jax.random.normal(key, (B, ACTION_HORIZON, ACTION_DIM)),
        state=jax.random.normal(jax.random.key(1), (B, 1, STATE_DIM)),
        vl_embeds=jax.random.normal(jax.random.key(2), (B, VL_SEQ, VL_DIM)),
        timestep=jnp.array([250, 750]),
        embodiment_id=jnp.array([0, 3]),
        image_mask=jnp.array([[True] * 10 + [False] * 10] * B),
        backbone_attention_mask=jnp.ones((B, VL_SEQ), dtype=jnp.bool_),
    )


def test_forward_shape_with_alternate_vl_dit():
    head = _build()
    out = head(**_inputs())
    assert out.shape == (B, ACTION_HORIZON, ACTION_DIM)


def test_forward_shape_with_plain_dit():
    head = _build(use_alternate_vl_dit=False)
    inputs = _inputs()
    # Plain DiT doesn't use image_mask.
    inputs.pop("image_mask")
    out = head(**inputs)
    assert out.shape == (B, ACTION_HORIZON, ACTION_DIM)


def test_different_timesteps_produce_different_outputs():
    head = _build()
    inputs = _inputs()
    inputs_b = dict(inputs, timestep=jnp.array([50, 950]))
    out_a = np.asarray(head(**inputs))
    out_b = np.asarray(head(**inputs_b))
    assert not np.allclose(out_a, out_b)


def test_different_embodiments_produce_different_outputs():
    head = _build()
    inputs = _inputs()
    inputs_b = dict(inputs, embodiment_id=jnp.array([5, 7]))
    out_a = np.asarray(head(**inputs))
    out_b = np.asarray(head(**inputs_b))
    assert not np.allclose(out_a, out_b)


def test_no_vlln_no_pos_embed_still_works():
    head = _build(use_vlln=False, add_pos_embed=False)
    out = head(**_inputs())
    assert out.shape == (B, ACTION_HORIZON, ACTION_DIM)


def test_dimension_mismatch_raises():
    import pytest

    with pytest.raises(ValueError, match="input_embedding_dim"):
        ActionHead(
            max_action_dim=16,
            max_state_dim=16,
            action_horizon=4,
            input_embedding_dim=100,  # doesn't match heads×head_dim=64
            dit_num_attention_heads=4,
            dit_attention_head_dim=16,
            rngs=nnx.Rngs(0),
        )
