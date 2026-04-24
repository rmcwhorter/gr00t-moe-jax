import jax
from flax import nnx

from gr00t_moe_jax.modules.qwen3_mlp import Qwen3MLP


def test_output_shape():
    mlp = Qwen3MLP(hidden_size=64, intermediate_size=256, rngs=nnx.Rngs(0))
    x = jax.random.normal(jax.random.key(0), (2, 5, 64))
    out = mlp(x)
    assert out.shape == (2, 5, 64)


def test_param_shapes():
    mlp = Qwen3MLP(hidden_size=8, intermediate_size=32, rngs=nnx.Rngs(0))
    # Flax Linear: kernel is (in, out).
    assert mlp.gate_proj.kernel[...].shape == (8, 32)
    assert mlp.up_proj.kernel[...].shape == (8, 32)
    assert mlp.down_proj.kernel[...].shape == (32, 8)


def test_no_bias_by_default():
    mlp = Qwen3MLP(hidden_size=8, intermediate_size=16, rngs=nnx.Rngs(0))
    assert mlp.gate_proj.bias is None
    assert mlp.up_proj.bias is None
    assert mlp.down_proj.bias is None
