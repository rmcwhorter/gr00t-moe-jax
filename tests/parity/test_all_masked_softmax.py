"""Pin the "all keys masked" attention behavior.

When every key is masked for a given query, softmax over all-`-inf` scores
yields a uniform distribution rather than zero. Our Attention inherits
this from PyTorch/diffusers exactly (both use the same masked-softmax
pattern). We lock in the current behavior so that:

1. The parity claim stays true — if we "fixed" this to return zero, we
   would silently diverge from the reference.
2. Anyone refactoring the attention path sees the intent and doesn't
   accidentally change it.

If we ever need safe-mode (zero output when no valid keys), it should be
a new keyword arg that's off by default.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
import torch
from flax import nnx

from gr00t_moe_jax.modules.attention import Attention
from gr00t_moe_jax.weights.converter import copy_linear


def _get_torch_attention():
    try:
        from diffusers.models.attention_processor import Attention as TorchAttention
    except Exception as e:
        pytest.skip(f"Cannot import diffusers Attention: {e}")
    return TorchAttention


def _copy_attn(pt_mod, jax_mod: Attention) -> None:
    copy_linear(pt_mod.to_q, jax_mod.to_q)
    copy_linear(pt_mod.to_k, jax_mod.to_k)
    copy_linear(pt_mod.to_v, jax_mod.to_v)
    copy_linear(pt_mod.to_out[0], jax_mod.out_proj)


def test_all_masked_softmax_is_uniform_not_zero():
    """When every key is masked, output should NOT be zero — it's the mean
    of (out_proj ∘ v_proj)(V) weighted uniformly. Same as diffusers."""
    TorchAttn = _get_torch_attention()
    query_dim, kv_dim, heads, dim_head = 16, 32, 2, 8

    torch.manual_seed(0)
    pt_mod = TorchAttn(query_dim=query_dim, cross_attention_dim=kv_dim, heads=heads, dim_head=dim_head)
    pt_mod.eval()

    jax_mod = Attention(query_dim=query_dim, heads=heads, dim_head=dim_head, cross_attention_dim=kv_dim, rngs=nnx.Rngs(0))
    _copy_attn(pt_mod, jax_mod)

    rng = np.random.default_rng(0)
    q_np = rng.standard_normal((1, 4, query_dim)).astype(np.float32)
    kv_np = rng.standard_normal((1, 6, kv_dim)).astype(np.float32)
    all_masked = np.zeros((1, 6), dtype=bool)  # every key masked out

    # Our JAX output.
    jax_out = np.asarray(
        jax_mod(
            jnp.asarray(q_np),
            encoder_hidden_states=jnp.asarray(kv_np),
            attention_mask=jnp.asarray(all_masked),
        )
    )

    # (a) It is NOT zero — documents the behavior.
    assert np.linalg.norm(jax_out) > 1e-3, "all-masked output should not collapse to zero"

    # (b) It matches the PyTorch reference. We can't easily pass diffusers
    #     a boolean mask directly, so reconstruct the expected value: with
    #     attention_probs uniform (1/6 per key), out = (out_proj ∘ v_proj)(mean(kv)).
    kv_mean = torch.from_numpy(kv_np).mean(dim=1, keepdim=True)  # (1, 1, kv_dim)
    with torch.no_grad():
        v = pt_mod.to_v(kv_mean)  # (1, 1, inner_dim)
        expected = pt_mod.to_out[0](v).numpy()  # (1, 1, query_dim)
    expected_broadcast = np.broadcast_to(expected, jax_out.shape)

    np.testing.assert_allclose(jax_out, expected_broadcast, rtol=1e-5, atol=1e-5)
