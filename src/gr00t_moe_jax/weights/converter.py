"""PyTorch → Flax NNX weight converter utilities.

These helpers mutate JAX modules in place, copying parameters from an
equivalent PyTorch module. We use the `variable[...] =...` assignment
pattern (still supported in current Flax NNX) to overwrite params.

Key shape conventions:
- PyTorch `Linear.weight` has shape `(out, in)`.
- Flax NNX `Linear.kernel` has shape `(in, out)`. So we transpose on copy.
- PyTorch `Embedding.weight` and Flax `nnx.Param` for embeddings both have
  shape `(num_embeddings, embedding_dim)`. No transpose.
- `CategorySpecificLinear.W` has shape `(num_categories, in_features, out_features)`
  in both PyTorch and JAX (our einsum convention matches PT's bmm). No transpose.
- LayerNorm weight (scale) and bias have the same shape in both.
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
import numpy as np
from flax import nnx


def torch_to_jax(t: Any) -> jnp.ndarray:
    """Convert a torch.Tensor (CPU) to a jnp.ndarray via numpy."""
    return jnp.asarray(np.asarray(t.detach().cpu().numpy()))


def copy_linear(pt_linear: Any, jax_linear: nnx.Linear) -> None:
    """Copy nn.Linear → nnx.Linear, transposing the weight matrix."""
    jax_linear.kernel[...] =torch_to_jax(pt_linear.weight).T
    if getattr(pt_linear, "bias", None) is not None and jax_linear.bias is not None:
        jax_linear.bias[...] =torch_to_jax(pt_linear.bias)


def copy_layernorm(pt_ln: Any, jax_ln: nnx.LayerNorm) -> None:
    """Copy nn.LayerNorm → nnx.LayerNorm. Handles both affine and non-affine cases."""
    # PyTorch LayerNorm stores `weight` and `bias` only when elementwise_affine=True.
    if getattr(pt_ln, "weight", None) is not None and jax_ln.scale is not None:
        jax_ln.scale[...] =torch_to_jax(pt_ln.weight)
    if getattr(pt_ln, "bias", None) is not None and jax_ln.bias is not None:
        jax_ln.bias[...] =torch_to_jax(pt_ln.bias)


def copy_embedding_param(pt_embed_weight: Any, jax_param: nnx.Param) -> None:
    """Copy nn.Embedding.weight → nnx.Param. Same shape convention."""
    jax_param[...] =torch_to_jax(pt_embed_weight)


def copy_categoryspecific_linear(pt_W: Any, pt_b: Any, jax_layer: Any) -> None:
    """Copy CategorySpecificLinear's W and b params. Shapes match both sides."""
    jax_layer.W[...] =torch_to_jax(pt_W)
    jax_layer.b[...] =torch_to_jax(pt_b)
