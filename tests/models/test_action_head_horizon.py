"""Regression test for the ActionHead state-token leak on short horizons
identified by the Codex audit.

Previously, ActionHead sliced by `self.action_horizon` (configured), so a
call with `noisy_action.shape[1] < action_horizon` would include the
prepended state token in the output. The reference slices by the actual
input length (gr00t_n1d7.py:259).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import nnx

from gr00t_moe_jax.models.action_head import ActionHead


def _make_head(action_horizon: int = 6) -> ActionHead:
    return ActionHead(
        max_action_dim=16,
        max_state_dim=16,
        action_horizon=action_horizon,
        hidden_size=48,
        input_embedding_dim=64,
        backbone_embedding_dim=32,
        state_history_length=1,
        max_seq_len=128,
        max_num_embodiments=8,
        dit_num_layers=2,
        dit_num_attention_heads=4,
        dit_attention_head_dim=16,
        dit_output_dim=48,
        use_alternate_vl_dit=True,
        attend_text_every_n_blocks=2,
        rngs=nnx.Rngs(0),
    )


def _inputs(horizon: int):
    B, S = 1, 8
    return dict(
        noisy_action=jax.random.normal(jax.random.key(0), (B, horizon, 16)),
        state=jax.random.normal(jax.random.key(1), (B, 1, 16)),
        vl_embeds=jax.random.normal(jax.random.key(2), (B, S, 32)),
        timestep=jnp.array([500]),
        embodiment_id=jnp.array([0]),
        image_mask=jnp.array([[True] * 4 + [False] * 4]),
        backbone_attention_mask=jnp.ones((B, S), dtype=jnp.bool_),
    )


def test_output_length_matches_input_horizon_when_shorter_than_config():
    # configured=6, input=3 → output should be 3 (not 4 with state-token leak,
    # not 6 with mismatched shape).
    head = _make_head(action_horizon=6)
    out = head(**_inputs(horizon=3))
    assert out.shape == (1, 3, 16)


def test_output_length_matches_input_horizon_when_equal_to_config():
    head = _make_head(action_horizon=6)
    out = head(**_inputs(horizon=6))
    assert out.shape == (1, 6, 16)


def test_output_length_matches_input_horizon_when_longer_than_config():
    # Flexible in both directions — the slice is driven by the input shape,
    # not the stored config.
    head = _make_head(action_horizon=4)
    out = head(**_inputs(horizon=7))
    assert out.shape == (1, 7, 16)
