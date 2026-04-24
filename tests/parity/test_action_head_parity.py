"""Top-level parity test for ActionHead.

Can't import the real `Gr00tN1d7ActionHead` directly (its config is a
dataclass that breaks with modern transformers), so we rebuild an
equivalent PyTorch wrapper inline using the same constituent modules
imported from `dit.py` and `embodiment_conditioned_mlp.py`. The
wrapper's forward exactly mirrors what the real GR00T action head does:

    vl_embeds -> vlln
    state -> state_encoder(CategorySpecificMLP)
    noised action + t -> action_encoder(MultiEmbodimentActionEncoder)
    action_features += position_embedding
    cat[state, action] -> AlternateVLDiT
    dit_out -> action_decoder(CategorySpecificMLP)
    return last `action_horizon` steps
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
import torch
from flax import nnx
from torch import nn

from gr00t_moe_jax.models.action_head import ActionHead
from gr00t_moe_jax.weights.converter import copy_linear, torch_to_jax
from tests.parity.test_dit_parity import _copy_dit


def _get_torch_modules():
    from tests.parity.conftest import (
        gr00t_src_available,
        load_gr00t_dit,
        load_gr00t_embodiment_mlp,
    )

    if not gr00t_src_available():
        pytest.skip("Isaac-GR00T source directory not found")
    try:
        dit_mod = load_gr00t_dit()
        emb_mod = load_gr00t_embodiment_mlp()
    except Exception as e:
        pytest.skip(f"Cannot load reference modules: {e}")
    return dit_mod, emb_mod


def _build_pt_action_head(dit_mod, emb_mod, *, config):
    """Construct an equivalent PyTorch ActionHead out of GR00T's primitives."""
    AlternateVLDiT = dit_mod.AlternateVLDiT
    CategorySpecificMLP = emb_mod.CategorySpecificMLP
    MultiEmbodimentActionEncoder = emb_mod.MultiEmbodimentActionEncoder

    class PTActionHead(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = AlternateVLDiT(
                num_attention_heads=config["heads"],
                attention_head_dim=config["head_dim"],
                output_dim=config["dit_output_dim"],
                num_layers=config["num_layers"],
                cross_attention_dim=config["backbone_dim"],
                activation_fn="gelu-approximate",
                norm_type="ada_norm",
                interleave_self_attention=True,
                positional_embeddings=None,
                attend_text_every_n_blocks=2,
            )
            self.state_encoder = CategorySpecificMLP(
                num_categories=config["num_emb"],
                input_dim=config["state_dim"] * config["hist"],
                hidden_dim=config["hidden"],
                output_dim=config["input_emb"],
            )
            self.action_encoder = MultiEmbodimentActionEncoder(
                action_dim=config["action_dim"],
                hidden_size=config["input_emb"],
                num_embodiments=config["num_emb"],
            )
            self.action_decoder = CategorySpecificMLP(
                num_categories=config["num_emb"],
                input_dim=config["dit_output_dim"],
                hidden_dim=config["hidden"],
                output_dim=config["action_dim"],
            )
            self.vlln = nn.LayerNorm(config["backbone_dim"])
            self.position_embedding = nn.Embedding(
                config["max_seq_len"], config["input_emb"]
            )
            nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)
            self.action_horizon = config["action_horizon"]

        def forward(
            self,
            noisy_action,
            state,
            vl_embeds,
            timestep,
            embodiment_id,
            image_mask,
            backbone_mask,
        ):
            vl = self.vlln(vl_embeds)
            state_flat = state.view(state.shape[0], 1, -1)
            state_feat = self.state_encoder(state_flat, embodiment_id)
            action_feat = self.action_encoder(
                noisy_action, timestep.float(), embodiment_id
            )
            pe = self.position_embedding(
                torch.arange(action_feat.shape[1], device=action_feat.device)
            )
            action_feat = action_feat + pe.unsqueeze(0)
            sa = torch.cat([state_feat, action_feat], dim=1)
            out = self.model(
                sa,
                vl,
                timestep,
                image_mask=image_mask,
                backbone_attention_mask=backbone_mask,
            )
            decoded = self.action_decoder(out, embodiment_id)
            return decoded[:, -self.action_horizon :]

    return PTActionHead()


def _copy_action_head(pt_mod, jax_mod: ActionHead) -> None:
    # DiT
    _copy_dit(pt_mod.model, jax_mod.model)

    # state_encoder (CategorySpecificMLP has layer1 + layer2 of CategorySpecificLinear)
    for which in ("layer1", "layer2"):
        pt_layer = getattr(pt_mod.state_encoder, which)
        jax_layer = getattr(jax_mod.state_encoder, which)
        jax_layer.W[...] = torch_to_jax(pt_layer.W)
        jax_layer.b[...] = torch_to_jax(pt_layer.b)

    # action_encoder (three CategorySpecificLinear layers W1, W2, W3)
    for which in ("W1", "W2", "W3"):
        pt_layer = getattr(pt_mod.action_encoder, which)
        jax_layer = getattr(jax_mod.action_encoder, which)
        jax_layer.W[...] = torch_to_jax(pt_layer.W)
        jax_layer.b[...] = torch_to_jax(pt_layer.b)

    # action_decoder (same as state_encoder structure)
    for which in ("layer1", "layer2"):
        pt_layer = getattr(pt_mod.action_decoder, which)
        jax_layer = getattr(jax_mod.action_decoder, which)
        jax_layer.W[...] = torch_to_jax(pt_layer.W)
        jax_layer.b[...] = torch_to_jax(pt_layer.b)

    # vlln (LayerNorm with affine)
    jax_mod.vlln.scale[...] = torch_to_jax(pt_mod.vlln.weight)
    jax_mod.vlln.bias[...] = torch_to_jax(pt_mod.vlln.bias)

    # position_embedding (nn.Embedding → nnx.Param)
    jax_mod.position_embedding[...] = torch_to_jax(pt_mod.position_embedding.weight)


def test_action_head_parity():
    dit_mod, emb_mod = _get_torch_modules()

    cfg = dict(
        heads=4,
        head_dim=16,
        dit_output_dim=48,
        num_layers=4,
        backbone_dim=32,
        num_emb=8,
        state_dim=16,
        action_dim=16,
        hist=1,
        hidden=48,
        input_emb=64,  # = heads × head_dim
        max_seq_len=128,
        action_horizon=6,
    )

    torch.manual_seed(0)
    pt_head = _build_pt_action_head(dit_mod, emb_mod, config=cfg)
    pt_head.eval()

    jax_head = ActionHead(
        max_action_dim=cfg["action_dim"],
        max_state_dim=cfg["state_dim"],
        action_horizon=cfg["action_horizon"],
        hidden_size=cfg["hidden"],
        input_embedding_dim=cfg["input_emb"],
        backbone_embedding_dim=cfg["backbone_dim"],
        state_history_length=cfg["hist"],
        max_seq_len=cfg["max_seq_len"],
        max_num_embodiments=cfg["num_emb"],
        dit_num_layers=cfg["num_layers"],
        dit_num_attention_heads=cfg["heads"],
        dit_attention_head_dim=cfg["head_dim"],
        dit_output_dim=cfg["dit_output_dim"],
        use_alternate_vl_dit=True,
        attend_text_every_n_blocks=2,
        use_vlln=True,
        add_pos_embed=True,
        rngs=nnx.Rngs(0),
    )
    _copy_action_head(pt_head, jax_head)

    # Inputs.
    B = 2
    S = 16  # VL context length
    rng = np.random.default_rng(0)
    noisy_np = rng.standard_normal((B, cfg["action_horizon"], cfg["action_dim"])).astype(np.float32)
    state_np = rng.standard_normal((B, cfg["hist"], cfg["state_dim"])).astype(np.float32)
    vl_np = rng.standard_normal((B, S, cfg["backbone_dim"])).astype(np.float32)
    ts_np = np.array([100, 700], dtype=np.int64)
    emb_np = np.array([0, 3], dtype=np.int64)
    image_mask_np = np.array([[True] * 8 + [False] * 8] * B, dtype=bool)
    backbone_mask_np = np.ones((B, S), dtype=bool)

    with torch.no_grad():
        pt_out = pt_head(
            torch.from_numpy(noisy_np),
            torch.from_numpy(state_np),
            torch.from_numpy(vl_np),
            torch.from_numpy(ts_np),
            torch.from_numpy(emb_np),
            torch.from_numpy(image_mask_np),
            torch.from_numpy(backbone_mask_np),
        ).numpy()

    jax_out = np.asarray(
        jax_head(
            noisy_action=jnp.asarray(noisy_np),
            state=jnp.asarray(state_np),
            vl_embeds=jnp.asarray(vl_np),
            timestep=jnp.asarray(ts_np),
            embodiment_id=jnp.asarray(emb_np),
            image_mask=jnp.asarray(image_mask_np),
            backbone_attention_mask=jnp.asarray(backbone_mask_np),
        )
    )

    np.testing.assert_allclose(jax_out, pt_out, rtol=1e-5, atol=1e-5)
