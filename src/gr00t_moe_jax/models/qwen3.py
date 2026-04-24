"""Qwen3DecoderLayer and Qwen3Model, ported from
`transformers/models/qwen3/modeling_qwen3.py`.

DecoderLayer:
    residual = x; x = input_layernorm(x);          x = attn(x) + residual
    residual = x; x = post_attention_layernorm(x); x = mlp(x)  + residual

Model:
    embed_tokens -> (rotary_emb once) -> N decoder layers -> final norm

The GR00T backbone uses this architecture via `Qwen3VLForConditionalGeneration`,
but extracts hidden states at `select_layer=12` rather than running all layers
to the LM head. The LM head is not ported here — we only need the
feature-extraction path.

KV cache and sliding-window attention are intentionally omitted; GR00T
does not do autoregressive generation, and the default Cosmos-Reason2-2B
config uses full attention everywhere.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import nnx

from gr00t_moe_jax.modules.qwen3_attention import Qwen3Attention
from gr00t_moe_jax.modules.qwen3_mlp import Qwen3MLP
from gr00t_moe_jax.modules.rmsnorm import RMSNorm
from gr00t_moe_jax.modules.rope import RotaryEmbedding


class Qwen3DecoderLayer(nnx.Module):
    def __init__(
        self,
        *,
        hidden_size: int,
        intermediate_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        rms_norm_eps: float = 1e-6,
        attention_bias: bool = False,
        rngs: nnx.Rngs,
    ):
        self.self_attn = Qwen3Attention(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            head_dim=head_dim,
            rms_norm_eps=rms_norm_eps,
            attention_bias=attention_bias,
            rngs=rngs,
        )
        self.mlp = Qwen3MLP(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            rngs=rngs,
        )
        self.input_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps, rngs=rngs)
        self.post_attention_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps, rngs=rngs)

    def __call__(
        self,
        hidden_states: jnp.ndarray,
        *,
        position_embeddings: tuple[jnp.ndarray, jnp.ndarray],
        attention_mask: jnp.ndarray | None = None,
    ) -> jnp.ndarray:
        # Self-attention.
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
        )
        hidden_states = residual + hidden_states

        # MLP.
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class Qwen3Model(nnx.Module):
    """Qwen3 text backbone — embed_tokens + N decoder layers + final norm.

    Supports optional early-exit via `select_layer` to match GR00T's
    `Qwen3VLForConditionalGeneration` truncation at layer 12.
    """

    def __init__(
        self,
        *,
        vocab_size: int,
        hidden_size: int,
        intermediate_size: int,
        num_hidden_layers: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        rope_theta: float = 10000.0,
        rms_norm_eps: float = 1e-6,
        attention_bias: bool = False,
        rngs: nnx.Rngs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers

        # Embedding matrix — shape (vocab_size, hidden_size).
        embed_init = 0.02 * jax.random.normal(
            rngs.params(), (vocab_size, hidden_size), dtype=jnp.float32
        )
        self.embed_tokens = nnx.Param(embed_init)

        layers = []
        for _ in range(num_hidden_layers):
            layers.append(
                Qwen3DecoderLayer(
                    hidden_size=hidden_size,
                    intermediate_size=intermediate_size,
                    num_attention_heads=num_attention_heads,
                    num_key_value_heads=num_key_value_heads,
                    head_dim=head_dim,
                    rms_norm_eps=rms_norm_eps,
                    attention_bias=attention_bias,
                    rngs=rngs,
                )
            )
        self.layers = nnx.data(layers)

        self.norm = RMSNorm(hidden_size, eps=rms_norm_eps, rngs=rngs)
        self.rotary_emb = RotaryEmbedding(
            head_dim=head_dim, rope_theta=rope_theta, rngs=rngs
        )

    def __call__(
        self,
        input_ids: jnp.ndarray | None = None,
        *,
        inputs_embeds: jnp.ndarray | None = None,
        attention_mask: jnp.ndarray | None = None,
        position_ids: jnp.ndarray | None = None,
        select_layer: int | None = None,
    ) -> jnp.ndarray:
        """Returns hidden states of shape (B, T, hidden_size).

        - If `select_layer` is not None, returns the hidden state BEFORE the
          final norm, at the output of `layers[select_layer]` (matching the
          GR00T backbone's layer-truncation semantics). Otherwise runs all
          layers and applies the final norm.
        - `attention_mask` must be additive — 0 for attend, -inf for mask —
          and broadcastable to `(B, 1, T, T)`.
        """
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Pass exactly one of `input_ids` or `inputs_embeds`.")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens[...][input_ids]  # gather

        B, T, _ = inputs_embeds.shape
        if position_ids is None:
            position_ids = jnp.arange(T)[None, :].repeat(B, axis=0)

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        max_layer = len(self.layers) if select_layer is None else select_layer
        for idx in range(max_layer):
            hidden_states = self.layers[idx](
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
            )

        if select_layer is None:
            hidden_states = self.norm(hidden_states)
        return hidden_states
