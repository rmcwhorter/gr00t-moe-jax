# gr00t-moe-jax

JAX/Flax port of NVIDIA's [GR00T N1.7](https://github.com/NVIDIA/Isaac-GR00T) VLA, with the eventual goal of distilling its dense action DiT into a mixture-of-experts variant via sparse upcycling.

## What this is

A research project exploring whether the dense Diffusion Transformer action head in modern flow-matching VLAs (GR00T, π₀, SmolVLA) can be cheaply converted into a sparse MoE at fixed active-FLOPs — letting you scale total capacity without paying for it at inference time.

GR00T N1.7 is the concrete target because:

- It's Apache-2.0 licensed with weights on HuggingFace.
- Its DiT action head is structurally representative of current SOTA (16 blocks, cross-attention to VL context, AdaLN timestep conditioning, 4-step Euler flow-matching inference over 40-step action chunks).
- The `CategorySpecificLinear` embodiment-handling mechanism is effectively a hand-coded hard-routed MoE — a natural starting point for a learned gated version.

## Method (two stages)

1. **Port GR00T to JAX/Flax.** Dense inference parity with the PyTorch reference, bit-exact within fp32 tolerance. Includes the Cosmos-Reason2-2B (Qwen3-VL) backbone.
2. **MoE-ify the DiT.** Via sparse upcycling (Komatsuzaki et al., 2023) and/or MoEfication (Zhang et al., 2022): warm-start a MoE-DiT from the dense checkpoint, then fine-tune end-to-end with flow-matching on a public robot-data mix (DROID, LIBERO, OXE).

See [PLAN.md](./PLAN.md) for the phase breakdown.

## Status

Phase 1 complete — the full action-head stack (DiT, AlternateVLDiT,
CategorySpecificLinear/MLP, MultiEmbodimentActionEncoder, TimestepEncoder,
AdaLayerNorm, ActionHead) is ported to Flax NNX and verified bit-exact
against the PyTorch reference within 1e-5 fp32 tolerance.

## Development

```bash
# One-time env setup.
uv venv --python 3.12
uv sync --extra dev               # shape tests only
uv sync --extra dev --extra reference  # also enables parity tests (needs torch+diffusers)

# Run tests.
uv run pytest tests/              # everything
uv run pytest tests/modules/      # just the shape-level tests

# Parity tests additionally require a local Isaac-GR00T clone:
git clone https://github.com/NVIDIA/Isaac-GR00T.git ~/repos/robotics/Isaac-GR00T
GR00T_REF_PATH=~/repos/robotics/Isaac-GR00T uv run pytest tests/parity/
```

If `GR00T_REF_PATH` is unset and the default location doesn't exist, parity
tests skip with an explanatory message (not silently pass).

## License

Apache-2.0. Compatible with GR00T N1.7's license.

## References

- [GR00T N1: An Open Foundation Model for Generalist Humanoid Robots](https://arxiv.org/abs/2503.14734)
- [Sparse Upcycling: Training Mixture-of-Experts from Dense Checkpoints](https://arxiv.org/abs/2212.05055)
- [MoEfication: Transformer Feed-forward Layers are Mixtures of Experts](https://arxiv.org/abs/2110.01786)
- [Scalable Diffusion Models with Transformers (DiT)](https://arxiv.org/abs/2212.09748)
- [π₀: A Vision-Language-Action Flow Model for General Robot Control](https://www.pi.website/download/pi0.pdf)
