"""ActionHead — the top-level JAX module for GR00T's action head.

Glues together:
- `vlln`: optional LayerNorm on the VL context from the Qwen3-VL backbone
- `state_encoder`: CategorySpecificMLP encoding (state_dim × history) → inner_dim
- `action_encoder`: MultiEmbodimentActionEncoder for noised actions + timestep
- `position_embedding`: learned per-chunk-position embedding
- `model`: DiT or AlternateVLDiT — produces velocity predictions
- `action_decoder`: CategorySpecificMLP mapping DiT output → action_dim

This module's `__call__` is the pure architectural forward: given all the
inputs (including the already-noised trajectory and timestep), it returns
the predicted velocity vector field. Flow-matching noise sampling and
Euler integration live outside the module (see `flow_matching.py` when we
add it), following Flax's separation of architecture vs. training logic.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import nnx

from gr00t_moe_jax.models.dit import AlternateVLDiT, DiT
from gr00t_moe_jax.modules.action_encoder import MultiEmbodimentActionEncoder
from gr00t_moe_jax.modules.category_specific import CategorySpecificMLP


class ActionHead(nnx.Module):
    """GR00T N1.7 action head. Predicts flow-matching velocity for an action chunk."""

    def __init__(
        self,
        *,
        # Shapes.
        max_action_dim: int = 132,
        max_state_dim: int = 132,
        action_horizon: int = 40,
        hidden_size: int = 1024,
        input_embedding_dim: int = 1536,
        backbone_embedding_dim: int = 2048,
        state_history_length: int = 1,
        max_seq_len: int = 1024,
        max_num_embodiments: int = 32,
        # DiT config.
        dit_num_layers: int = 16,
        dit_num_attention_heads: int = 32,
        dit_attention_head_dim: int = 48,
        dit_output_dim: int = 1024,
        dit_activation_fn: str = "gelu-approximate",
        dit_norm_type: str = "ada_norm",
        dit_interleave_self_attention: bool = True,
        # Options.
        use_alternate_vl_dit: bool = True,
        attend_text_every_n_blocks: int = 2,
        use_vlln: bool = True,
        add_pos_embed: bool = True,
        rngs: nnx.Rngs,
    ):
        if input_embedding_dim != dit_num_attention_heads * dit_attention_head_dim:
            raise ValueError(
                f"input_embedding_dim ({input_embedding_dim}) must equal "
                f"dit_num_attention_heads × dit_attention_head_dim "
                f"({dit_num_attention_heads} × {dit_attention_head_dim})"
            )

        self.action_horizon = action_horizon
        self.max_action_dim = max_action_dim
        self.max_state_dim = max_state_dim
        self.state_history_length = state_history_length
        self.input_embedding_dim = input_embedding_dim
        self.use_alternate_vl_dit = use_alternate_vl_dit
        self.add_pos_embed = add_pos_embed
        self.use_vlln = use_vlln

        # --- DiT / AlternateVLDiT --------------------------------------------
        dit_kwargs = dict(
            num_attention_heads=dit_num_attention_heads,
            attention_head_dim=dit_attention_head_dim,
            output_dim=dit_output_dim,
            num_layers=dit_num_layers,
            activation_fn=dit_activation_fn,
            norm_type=dit_norm_type,
            interleave_self_attention=dit_interleave_self_attention,
            cross_attention_dim=backbone_embedding_dim,
            rngs=rngs,
        )
        if use_alternate_vl_dit:
            self.model = AlternateVLDiT(
                attend_text_every_n_blocks=attend_text_every_n_blocks,
                **dit_kwargs,
            )
        else:
            self.model = DiT(**dit_kwargs)

        # --- Encoders / decoder ----------------------------------------------
        self.state_encoder = CategorySpecificMLP(
            num_categories=max_num_embodiments,
            in_features=max_state_dim * state_history_length,
            hidden_features=hidden_size,
            out_features=input_embedding_dim,
            rngs=rngs,
        )
        self.action_encoder = MultiEmbodimentActionEncoder(
            action_dim=max_action_dim,
            hidden_size=input_embedding_dim,
            num_embodiments=max_num_embodiments,
            rngs=rngs,
        )
        self.action_decoder = CategorySpecificMLP(
            num_categories=max_num_embodiments,
            in_features=dit_output_dim,
            hidden_features=hidden_size,
            out_features=max_action_dim,
            rngs=rngs,
        )

        # --- VL pre-DiT normalization ---------------------------------------
        if use_vlln:
            self.vlln = nnx.LayerNorm(backbone_embedding_dim, rngs=rngs)
        else:
            self.vlln = None

        # --- Per-chunk-position learnable embedding --------------------------
        if add_pos_embed:
            # nn.Embedding initialised N(0, 0.02) in the PyTorch reference.
            pe_init = 0.02 * jax.random.normal(
                rngs.params(), (max_seq_len, input_embedding_dim), dtype=jnp.float32
            )
            self.position_embedding = nnx.Param(pe_init)
        else:
            self.position_embedding = None

    def __call__(
        self,
        *,
        noisy_action: jnp.ndarray,  # (B, action_horizon, max_action_dim)
        state: jnp.ndarray,  # (B, state_history_length, max_state_dim)
        vl_embeds: jnp.ndarray,  # (B, S, backbone_embedding_dim)
        timestep: jnp.ndarray,  # (B,) — discretised flow-matching timestep
        embodiment_id: jnp.ndarray,  # (B,) int
        image_mask: jnp.ndarray | None = None,  # (B, S) bool — AlternateVLDiT only
        backbone_attention_mask: jnp.ndarray | None = None,  # (B, S) bool
    ) -> jnp.ndarray:
        """Returns predicted velocity field: (B, action_horizon, max_action_dim)."""
        # --- Process VL context ---------------------------------------------
        if self.vlln is not None:
            vl_embeds = self.vlln(vl_embeds)

        # --- Encode state ---------------------------------------------------
        B = state.shape[0]
        state_flat = state.reshape(B, 1, -1)  # flatten history × dim → (B, 1, D_s)
        state_features = self.state_encoder(state_flat, embodiment_id)  # (B, 1, input_emb)

        # --- Encode noised actions + timestep --------------------------------
        action_features = self.action_encoder(
            noisy_action, timestep.astype(jnp.float32), embodiment_id
        )  # (B, action_horizon, input_emb)

        if self.position_embedding is not None:
            pe = self.position_embedding[...][: action_features.shape[1]]  # (T, input_emb)
            action_features = action_features + pe[None, ...]

        # --- Concat state + action tokens into DiT input ---------------------
        sa_embs = jnp.concatenate([state_features, action_features], axis=1)
        # (B, 1 + action_horizon, input_emb)

        # --- DiT forward ----------------------------------------------------
        if self.use_alternate_vl_dit:
            if image_mask is None or backbone_attention_mask is None:
                raise ValueError(
                    "AlternateVLDiT requires `image_mask` and `backbone_attention_mask`"
                )
            dit_out = self.model(
                sa_embs,
                vl_embeds,
                timestep,
                image_mask=image_mask,
                backbone_attention_mask=backbone_attention_mask,
            )
        else:
            dit_out = self.model(
                sa_embs,
                vl_embeds,
                timestep,
                encoder_attention_mask=backbone_attention_mask,
            )

        # --- Decode action tokens back to action space ----------------------
        decoded = self.action_decoder(dit_out, embodiment_id)
        # The state token is prepended — slice it off to get action predictions.
        return decoded[:, -self.action_horizon :]
