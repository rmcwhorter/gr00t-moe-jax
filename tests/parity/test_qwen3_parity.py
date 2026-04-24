"""End-to-end parity tests for Qwen3DecoderLayer and Qwen3Model."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
import torch
from flax import nnx

from gr00t_moe_jax.models.qwen3 import Qwen3DecoderLayer, Qwen3Model
from gr00t_moe_jax.modules.rope import RotaryEmbedding
from gr00t_moe_jax.weights.converter import copy_linear, torch_to_jax


def _get_torch_bits():
    try:
        from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
        from transformers.models.qwen3.modeling_qwen3 import (
            Qwen3DecoderLayer as TorchLayer,
        )
        from transformers.models.qwen3.modeling_qwen3 import (
            Qwen3Model as TorchModel,
        )
        from transformers.models.qwen3.modeling_qwen3 import (
            Qwen3RotaryEmbedding as TorchRope,
        )
    except Exception as e:
        pytest.skip(f"Cannot import Qwen3 modules: {e}")
    return TorchLayer, TorchModel, TorchRope, Qwen3Config


def _copy_attn(pt_attn, jax_attn) -> None:
    copy_linear(pt_attn.q_proj, jax_attn.q_proj)
    copy_linear(pt_attn.k_proj, jax_attn.k_proj)
    copy_linear(pt_attn.v_proj, jax_attn.v_proj)
    copy_linear(pt_attn.o_proj, jax_attn.o_proj)
    jax_attn.q_norm.weight[...] = torch_to_jax(pt_attn.q_norm.weight)
    jax_attn.k_norm.weight[...] = torch_to_jax(pt_attn.k_norm.weight)


def _copy_mlp(pt_mlp, jax_mlp) -> None:
    copy_linear(pt_mlp.gate_proj, jax_mlp.gate_proj)
    copy_linear(pt_mlp.up_proj, jax_mlp.up_proj)
    copy_linear(pt_mlp.down_proj, jax_mlp.down_proj)


def _copy_layer(pt_layer, jax_layer: Qwen3DecoderLayer) -> None:
    _copy_attn(pt_layer.self_attn, jax_layer.self_attn)
    _copy_mlp(pt_layer.mlp, jax_layer.mlp)
    jax_layer.input_layernorm.weight[...] = torch_to_jax(pt_layer.input_layernorm.weight)
    jax_layer.post_attention_layernorm.weight[...] = torch_to_jax(
        pt_layer.post_attention_layernorm.weight
    )


def _small_config(hidden_size=64, heads=4, kv_heads=2, head_dim=16, num_layers=1, vocab_size=32):
    _, _, _, Qwen3Config = _get_torch_bits()
    return Qwen3Config(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        intermediate_size=2 * hidden_size,
        num_hidden_layers=num_layers,
        num_attention_heads=heads,
        num_key_value_heads=kv_heads,
        head_dim=head_dim,
        attention_bias=False,
        rms_norm_eps=1e-6,
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0},
        use_sliding_window=False,
    )


def test_decoder_layer_parity():
    TorchLayer, _, TorchRope, _ = _get_torch_bits()
    cfg = _small_config()

    torch.manual_seed(0)
    pt_layer = TorchLayer(cfg, layer_idx=0).eval()
    pt_rope = TorchRope(cfg).eval()

    jax_layer = Qwen3DecoderLayer(
        hidden_size=cfg.hidden_size,
        intermediate_size=cfg.intermediate_size,
        num_attention_heads=cfg.num_attention_heads,
        num_key_value_heads=cfg.num_key_value_heads,
        head_dim=cfg.head_dim,
        rngs=nnx.Rngs(0),
    )
    _copy_layer(pt_layer, jax_layer)
    jax_rope = RotaryEmbedding(head_dim=cfg.head_dim, rngs=nnx.Rngs(0))

    B, T = 2, 8
    rng = np.random.default_rng(1)
    x_np = rng.standard_normal((B, T, cfg.hidden_size)).astype(np.float32)
    pos_np = np.tile(np.arange(T), (B, 1)).astype(np.int64)
    causal_np = np.triu(np.full((T, T), -np.inf, dtype=np.float32), k=1)[None, None, :, :]
    causal_np = np.broadcast_to(causal_np, (B, 1, T, T)).copy()

    with torch.no_grad():
        pt_cos, pt_sin = pt_rope(torch.from_numpy(x_np), torch.from_numpy(pos_np))
        pt_out = pt_layer(
            torch.from_numpy(x_np),
            attention_mask=torch.from_numpy(causal_np),
            position_embeddings=(pt_cos, pt_sin),
        )
        # Some wrappers return a tuple; unwrap if needed.
        if isinstance(pt_out, tuple):
            pt_out = pt_out[0]
        pt_out = pt_out.numpy()

    jax_cos, jax_sin = jax_rope(jnp.asarray(x_np), jnp.asarray(pos_np))
    jax_out = np.asarray(
        jax_layer(
            jnp.asarray(x_np),
            position_embeddings=(jax_cos, jax_sin),
            attention_mask=jnp.asarray(causal_np),
        )
    )
    np.testing.assert_allclose(jax_out, pt_out, rtol=1e-5, atol=1e-5)


def test_qwen3_model_parity():
    """Full model parity: embeddings + all layers + final norm.

    The HF model automatically builds a causal mask internally from the
    default attention_mask (None). Our JAX port requires a mask to be
    passed explicitly, so construct the same causal mask on both sides.
    """
    _, TorchModel, _, _ = _get_torch_bits()
    cfg = _small_config(num_layers=3, vocab_size=50)

    torch.manual_seed(0)
    pt_model = TorchModel(cfg).eval()

    jax_model = Qwen3Model(
        vocab_size=cfg.vocab_size,
        hidden_size=cfg.hidden_size,
        intermediate_size=cfg.intermediate_size,
        num_hidden_layers=cfg.num_hidden_layers,
        num_attention_heads=cfg.num_attention_heads,
        num_key_value_heads=cfg.num_key_value_heads,
        head_dim=cfg.head_dim,
        rngs=nnx.Rngs(0),
    )

    # Copy weights: embeddings + each layer + final norm.
    jax_model.embed_tokens[...] = torch_to_jax(pt_model.embed_tokens.weight)
    for pt_layer, jax_layer in zip(pt_model.layers, jax_model.layers, strict=True):
        _copy_layer(pt_layer, jax_layer)
    jax_model.norm.weight[...] = torch_to_jax(pt_model.norm.weight)

    B, T = 2, 6
    rng = np.random.default_rng(7)
    input_ids_np = rng.integers(0, cfg.vocab_size, size=(B, T)).astype(np.int64)
    causal_np = np.triu(np.full((T, T), -np.inf, dtype=np.float32), k=1)[None, None, :, :]
    causal_np = np.broadcast_to(causal_np, (B, 1, T, T)).copy()

    with torch.no_grad():
        pt_out = pt_model(input_ids=torch.from_numpy(input_ids_np)).last_hidden_state.numpy()

    jax_out = np.asarray(
        jax_model(jnp.asarray(input_ids_np), attention_mask=jnp.asarray(causal_np))
    )
    np.testing.assert_allclose(jax_out, pt_out, rtol=1e-5, atol=1e-5)


def test_qwen3_model_select_layer_parity():
    """select_layer=K in our JAX model should match running K layers of HF
    WITHOUT the final norm — matching GR00T's truncation semantics."""
    _, TorchModel, _, _ = _get_torch_bits()
    cfg = _small_config(num_layers=4, vocab_size=30)

    torch.manual_seed(1)
    pt_model = TorchModel(cfg).eval()

    jax_model = Qwen3Model(
        vocab_size=cfg.vocab_size,
        hidden_size=cfg.hidden_size,
        intermediate_size=cfg.intermediate_size,
        num_hidden_layers=cfg.num_hidden_layers,
        num_attention_heads=cfg.num_attention_heads,
        num_key_value_heads=cfg.num_key_value_heads,
        head_dim=cfg.head_dim,
        rngs=nnx.Rngs(0),
    )
    jax_model.embed_tokens[...] = torch_to_jax(pt_model.embed_tokens.weight)
    for pt_layer, jax_layer in zip(pt_model.layers, jax_model.layers, strict=True):
        _copy_layer(pt_layer, jax_layer)
    jax_model.norm.weight[...] = torch_to_jax(pt_model.norm.weight)

    B, T, K = 1, 5, 2  # truncate after K=2 layers
    rng = np.random.default_rng(13)
    input_ids_np = rng.integers(0, cfg.vocab_size, size=(B, T)).astype(np.int64)
    causal_np = np.triu(np.full((T, T), -np.inf, dtype=np.float32), k=1)[None, None, :, :]
    causal_np = np.broadcast_to(causal_np, (B, 1, T, T)).copy()

    # Reference: run embed + K layers by hand.
    with torch.no_grad():
        pt_embeds = pt_model.embed_tokens(torch.from_numpy(input_ids_np))
        pt_position_ids = torch.arange(T)[None, :]
        pt_cos, pt_sin = pt_model.rotary_emb(pt_embeds, pt_position_ids)
        pt_h = pt_embeds
        for idx in range(K):
            pt_h = pt_model.layers[idx](
                pt_h,
                attention_mask=torch.from_numpy(causal_np),
                position_embeddings=(pt_cos, pt_sin),
            )
            if isinstance(pt_h, tuple):
                pt_h = pt_h[0]
        pt_out = pt_h.numpy()

    jax_out = np.asarray(
        jax_model(
            jnp.asarray(input_ids_np),
            attention_mask=jnp.asarray(causal_np),
            select_layer=K,
        )
    )
    np.testing.assert_allclose(jax_out, pt_out, rtol=1e-5, atol=1e-5)
