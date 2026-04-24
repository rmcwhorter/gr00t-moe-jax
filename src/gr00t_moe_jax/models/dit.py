"""DiT and AlternateVLDiT, ported from `gr00t/model/modules/dit.py`.

The DiT is GR00T's action head: it takes the noised action-chunk tokens
(plus a state token prepended by the outer action head), conditions on a
flow-matching timestep via AdaLN, cross-attends to VL embeddings from the
Qwen3-VL backbone, and outputs a velocity prediction per token.

Block layout (with `interleave_self_attention=True`, which N1.7 uses):
- Even blocks (idx 0, 2, 4, ...): cross-attention, `cross_attention_dim=2048`
- Odd  blocks (idx 1, 3, 5, ...): self-attention,  `cross_attention_dim=None`

AlternateVLDiT further splits the cross-attention blocks based on their
index modulo (2 · attend_text_every_n_blocks):
- idx % (2·N) == 0:   attend to non-image (text) tokens only
- idx % (2·N) != 0:   attend to image tokens only
(with N = attend_text_every_n_blocks = 2 for GR00T N1.7)

The final output block is an AdaLN-style shift/scale modulation
(computed from temb via SiLU → Linear → chunk(2)) applied to a non-affine
LayerNorm, followed by a projection to `output_dim` = 1024.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import nnx

from gr00t_moe_jax.modules.transformer_block import BasicTransformerBlock
from gr00t_moe_jax.modules.timestep import TimestepEncoder


class DiT(nnx.Module):
    """Standard DiT with optional self-attention interleaving."""

    def __init__(
        self,
        *,
        num_attention_heads: int = 32,
        attention_head_dim: int = 48,
        output_dim: int = 1024,
        num_layers: int = 16,
        attention_bias: bool = True,
        activation_fn: str = "gelu-approximate",
        norm_type: str = "ada_norm",
        norm_eps: float = 1e-5,
        interleave_self_attention: bool = True,
        cross_attention_dim: int | None = None,
        # Kwargs accepted for GR00T compatibility; ignored for now:
        positional_embeddings: str | None = None,
        max_num_positional_embeddings: int = 512,
        dropout: float = 0.0,
        final_dropout: bool = True,
        norm_elementwise_affine: bool = False,
        num_embeds_ada_norm: int | None = 1000,
        upcast_attention: bool = False,
        compute_dtype=None,
        rngs: nnx.Rngs,
    ):
        del (  # silence: accepted for config parity, not used in inference port
            positional_embeddings,
            max_num_positional_embeddings,
            dropout,
            final_dropout,
            norm_elementwise_affine,
            num_embeds_ada_norm,
            upcast_attention,
            compute_dtype,
        )

        self.inner_dim = num_attention_heads * attention_head_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.interleave_self_attention = interleave_self_attention
        self.cross_attention_dim = cross_attention_dim

        self.timestep_encoder = TimestepEncoder(embedding_dim=self.inner_dim, rngs=rngs)

        # Alternate cross/self-attention across blocks.
        blocks = []
        for idx in range(num_layers):
            use_self_attn = interleave_self_attention and (idx % 2 == 1)
            curr_cross_dim = None if use_self_attn else cross_attention_dim
            blocks.append(
                BasicTransformerBlock(
                    self.inner_dim,
                    num_attention_heads,
                    attention_head_dim,
                    cross_attention_dim=curr_cross_dim,
                    attention_bias=attention_bias,
                    activation_fn=activation_fn,
                    norm_type=norm_type,
                    norm_eps=norm_eps,
                    rngs=rngs,
                )
            )
        # NNX strict mode: wrap container of modules as data so it joins the pytree.
        self.transformer_blocks = nnx.data(blocks)

        # Output AdaLN + projection.
        self.norm_out = nnx.LayerNorm(
            self.inner_dim,
            epsilon=1e-6,
            use_bias=False,
            use_scale=False,
            rngs=rngs,
        )
        self.proj_out_1 = nnx.Linear(self.inner_dim, 2 * self.inner_dim, rngs=rngs)
        self.proj_out_2 = nnx.Linear(self.inner_dim, output_dim, rngs=rngs)

    def _apply_output_head(
        self, hidden_states: jnp.ndarray, temb: jnp.ndarray
    ) -> jnp.ndarray:
        shift_scale = self.proj_out_1(jax.nn.silu(temb))  # (B, 2·D)
        shift, scale = jnp.split(shift_scale, 2, axis=-1)
        hidden_states = (
            self.norm_out(hidden_states) * (1.0 + scale[:, None, :])
            + shift[:, None, :]
        )
        return self.proj_out_2(hidden_states)

    def __call__(
        self,
        hidden_states: jnp.ndarray,  # (B, T, inner_dim)
        encoder_hidden_states: jnp.ndarray,  # (B, S, cross_attention_dim)
        timestep: jnp.ndarray,  # (B,)
        *,
        encoder_attention_mask: jnp.ndarray | None = None,
    ) -> jnp.ndarray:
        temb = self.timestep_encoder(timestep)

        for idx, block in enumerate(self.transformer_blocks):
            if self.interleave_self_attention and idx % 2 == 1:
                # Self-attention block.
                hidden_states = block(hidden_states, temb=temb)
            else:
                # Cross-attention block.
                hidden_states = block(
                    hidden_states,
                    temb=temb,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                )

        return self._apply_output_head(hidden_states, temb)


class AlternateVLDiT(DiT):
    """DiT variant that alternates cross-attention between image-only and
    non-image (text) tokens across blocks."""

    def __init__(self, *args, attend_text_every_n_blocks: int = 2, **kwargs):
        super().__init__(*args, **kwargs)
        self.attend_text_every_n_blocks = attend_text_every_n_blocks
        if not self.interleave_self_attention:
            raise ValueError(
                "AlternateVLDiT requires interleave_self_attention=True"
            )

    def __call__(  # type: ignore[override]
        self,
        hidden_states: jnp.ndarray,  # (B, T, inner_dim)
        encoder_hidden_states: jnp.ndarray,  # (B, S, cross_attention_dim)
        timestep: jnp.ndarray,  # (B,)
        *,
        image_mask: jnp.ndarray,  # (B, S) bool, True where token is image
        backbone_attention_mask: jnp.ndarray,  # (B, S) bool, True = non-padding
    ) -> jnp.ndarray:
        temb = self.timestep_encoder(timestep)

        image_mask_bool = image_mask.astype(jnp.bool_)
        backbone_mask_bool = backbone_attention_mask.astype(jnp.bool_)
        image_attn_mask = image_mask_bool & backbone_mask_bool  # image tokens only
        text_attn_mask = (~image_mask_bool) & backbone_mask_bool  # non-image only

        period = 2 * self.attend_text_every_n_blocks
        for idx, block in enumerate(self.transformer_blocks):
            if idx % 2 == 1:
                # Self-attention block.
                hidden_states = block(hidden_states, temb=temb)
            else:
                # Cross-attention block — alternate between text and image.
                if idx % period == 0:
                    curr_mask = text_attn_mask
                else:
                    curr_mask = image_attn_mask
                hidden_states = block(
                    hidden_states,
                    temb=temb,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=curr_mask,
                )

        return self._apply_output_head(hidden_states, temb)
