"""BasicTransformerBlock, ported from `gr00t/model/modules/dit.py`.

Non-standard block layout — just ONE attention stage + FFN, not the usual
self+cross+FFN triple. Self vs cross is decided at construction by
`cross_attention_dim` (None = self-attention; otherwise cross-attention
with K/V drawn from `encoder_hidden_states`).

    x  →  AdaLN(x, temb)  →  Attention  →  + residual
       →  LayerNorm        →  GeGLU-FFN  →  + residual
"""

from __future__ import annotations

import jax.numpy as jnp
from flax import nnx

from gr00t_moe_jax.modules.adalayernorm import AdaLayerNorm
from gr00t_moe_jax.modules.attention import Attention
from gr00t_moe_jax.modules.feedforward import FeedForward


class BasicTransformerBlock(nnx.Module):
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        *,
        cross_attention_dim: int | None = None,
        attention_bias: bool = True,
        norm_type: str = "ada_norm",  # only ada_norm supported for GR00T's use
        norm_eps: float = 1e-5,
        activation_fn: str = "gelu-approximate",
        ff_inner_dim: int | None = None,
        ff_bias: bool = True,
        attention_out_bias: bool = True,
        rngs: nnx.Rngs,
    ):
        if norm_type != "ada_norm":
            raise NotImplementedError(
                f"norm_type={norm_type!r} not supported yet; only 'ada_norm' is implemented"
            )

        self.norm1 = AdaLayerNorm(dim, norm_eps=norm_eps, rngs=rngs)
        self.attn1 = Attention(
            query_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            cross_attention_dim=cross_attention_dim,
            bias=attention_bias,
            out_bias=attention_out_bias,
            rngs=rngs,
        )
        self.norm3 = nnx.LayerNorm(dim, epsilon=norm_eps, rngs=rngs)
        self.ff = FeedForward(
            dim,
            inner_dim=ff_inner_dim,
            activation_fn=activation_fn,
            bias=ff_bias,
            rngs=rngs,
        )

    def __call__(
        self,
        hidden_states: jnp.ndarray,  # (B, T, D)
        *,
        temb: jnp.ndarray,  # (B, D)
        encoder_hidden_states: jnp.ndarray | None = None,
        attention_mask: jnp.ndarray | None = None,  # self-attention mask
        encoder_attention_mask: jnp.ndarray | None = None,  # cross-attention mask
    ) -> jnp.ndarray:
        norm = self.norm1(hidden_states, temb)

        # If cross-attending, use encoder_attention_mask; else use attention_mask.
        attn_mask = encoder_attention_mask if encoder_hidden_states is not None else attention_mask
        attn_out = self.attn1(
            norm,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attn_mask,
        )
        hidden_states = hidden_states + attn_out

        ff_out = self.ff(self.norm3(hidden_states))
        hidden_states = hidden_states + ff_out
        return hidden_states
