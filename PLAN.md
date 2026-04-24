# Plan

Phased plan. Each phase has a concrete deliverable that unblocks the next.

## Phase 0 — Scaffold (½ day)

- [x] Create public repo
- [x] License (Apache-2.0)
- [x] pyproject.toml with JAX deps
- [x] Empty package + tests layout
- [ ] CI skeleton (pytest on CPU via GitHub Actions)

## Phase 1 — Port GR00T's DiT to Flax NNX (1–2 weeks)

The DiT itself is modest — 16 blocks of {AdaLN, attention, FFN}. Mechanical port.

- [x] `CategorySpecificLinear` / `CategorySpecificMLP`
- [x] `MultiEmbodimentActionEncoder` (including sinusoidal timestep broadcast)
- [x] `AdaLayerNorm` with timestep conditioning
- [x] `TimestepEncoder` (Timesteps + MLP, matching diffusers' pipeline)
- [x] `FeedForward` (supports gelu / gelu-approximate / geglu)
- [x] `Attention` (self and cross, with encoder padding mask)
- [x] `BasicTransformerBlock` (single-attention + FFN layout, matches GR00T)
- [x] `DiT` top-level (16 blocks, 32 heads × 48 head_dim = 1536 inner, 1024 out)
- [x] `AlternateVLDiT` variant (image/text cross-attention alternation)
- [x] `ActionHead` (glues encoders, DiT, position embedding, vlln)
- [x] Shape-level unit tests (42 passing)
- [x] **Numerical parity tests against PyTorch reference** — 14 passing
      (AdaLayerNorm, Attention ×2, FeedForward ×2, CategorySpecific ×2,
       ActionEncoder, TimestepEncoder, TransformerBlock ×2, DiT ×2, ActionHead)
      All modules match PyTorch within 1e-5 fp32 tolerance.
- [ ] `compute_loss` / `sample_actions` flow-matching wrappers (Phase 3)

## Weight loading (done alongside Phase 1)

- [x] PyTorch → Flax NNX converter utilities (`gr00t_moe_jax/weights/converter.py`)
- [x] Direct-file import of reference modules via importlib (sidesteps
      GR00T's transformers-5.x dataclass incompatibility)
- [x] Reconstructable PyTorch ActionHead equivalent from primitives (since
      the top-level `Gr00tN1d7ActionHead` isn't directly importable)
- [ ] Full N1.7 HuggingFace checkpoint loader (needs real nvidia/* weights;
      deferred until we need end-to-end inference)

## Phase 2 — Port the Qwen3-VL backbone (1–3 weeks)

Cosmos-Reason2-2B is stock `Qwen3VLForConditionalGeneration` — not a custom NVIDIA fork.

### Phase 2.1 — Text transformer (done)

- [x] `RMSNorm` (Qwen2/Qwen3 RMSNorm, fp32 variance + cast back)
- [x] `Qwen3MLP` (SwiGLU: gate × up → down, bias-free)
- [x] RoPE: `compute_default_inv_freq`, `RotaryEmbedding`, `rotate_half`, `apply_rotary_pos_emb`
- [x] `Qwen3Attention` (GQA + per-head RMSNorm on Q/K, additive mask, fp32 softmax)
- [x] `Qwen3DecoderLayer` (pre-norm residual, attn + mlp)
- [x] `Qwen3Model` (embed + N layers + final norm, with `select_layer` truncation for GR00T's layer-12 cut)
- [x] **Parity vs HF Qwen3** (9 parity tests, all bit-exact within 1e-5 fp32):
      RMSNorm, MLP, RoPE ×2, Attention MHA + GQA, DecoderLayer, Model full, Model truncated

*Deferred:* NTK frequency scaling, sliding-window attention, KV cache
(not needed — GR00T treats the backbone as a frozen feature extractor,
and Cosmos-Reason2-2B uses full attention everywhere).

### Phase 2.2 — Vision tower (next)

- [ ] Qwen3-VL native-aspect-ratio ViT
- [ ] Patch embedding + 2D position encoding
- [ ] Vision attention (no GQA, no q_norm/k_norm — differs from text side)
- [ ] Vision-to-language projection

### Phase 2.3 — Multimodal integration

- [ ] Image/text token interleaving: `<image>` → expanded patch tokens
- [ ] Attention-mask construction for mixed-modality sequences
- [ ] Full `Qwen3VLForConditionalGeneration` wrapper (feature-extraction flavor)

### Phase 2.4 — Checkpoint loading + end-to-end parity

- [ ] HF safetensors → Flax pytree converter for `nvidia/Cosmos-Reason2-2B`
- [ ] End-to-end parity test: same (image + text) input through HF model and
      our JAX port should produce identical hidden states at `select_layer=12`

## Phase 3 — End-to-end dense inference (1 week)

Full JAX pipeline: image + language + state → VL embeddings → DiT → action chunk.

- [ ] Processor port: image preprocessing (native aspect, 256×256 crop), state normalization
- [ ] Flow-matching inference loop (4-step Euler, `actions = actions + dt * v * vel_strength`)
- [ ] Load full GR00T N1.7 checkpoint and run on LIBERO sample
- [ ] Compare success-rate on LIBERO vs PyTorch reference

## Phase 4 — MoE infrastructure (1–2 weeks)

- [ ] `MoEFeedForward` with top-K routing (token-choice)
- [ ] Expert-choice routing variant (guarantees load balance)
- [ ] Load-balance auxiliary loss
- [ ] Router initialization utilities
- [ ] Tests: dense-equivalence at init, correct gradient flow

## Phase 5 — Initialization strategies (1 week)

- [ ] `sparse_upcycling.py` — replicate dense FFN N times, zero router
- [ ] `moefication.py` — K-means cluster FFN hidden neurons on teacher activations, partition weights
- [ ] Tests: upcycled MoE matches dense output (sparse upcycling)
- [ ] Tests: moefied MoE approximates dense output (MoEfication)

## Phase 6 — Warm-start distillation (1 week experiment)

- [ ] Frozen dense teacher, MoE student
- [ ] `MSE(v_student, v_teacher)` distillation loop
- [ ] Uses random `(noise, t)` pairs + real `cond` from unlabeled robot video
- [ ] Converges to teacher parity before end-to-end training

## Phase 7 — End-to-end flow-matching fine-tune (2–4 weeks, compute-bound)

- [ ] Data loader for DROID + LIBERO + OXE mix
- [ ] Multi-device sharding (pjit / shard_map)
- [ ] Standard GR00T flow-matching loss: `MSE(pred, actions - noise) * action_mask`
- [ ] Plus MoE aux loss
- [ ] Checkpointing, resumption, logging

## Phase 8 — Evaluation (2 weeks)

- [ ] LIBERO sim benchmark (success rate)
- [ ] Active-FLOPs vs total-params measurement
- [ ] N-experts sweep (N = 2, 4, 8, 16)
- [ ] Upcycling vs MoEfication vs train-from-scratch ablation
- [ ] Expert-specialization analysis (do experts specialize by timestep / embodiment / action mode?)

## Risks

- **Qwen3-VL RoPE subtleties**: axis ordering, NTK scaling details, frequency caching. Mitigate with careful reference-comparison tests early.
- **Vision tower fusion**: Qwen3-VL's native-aspect-ratio handling is non-trivial. May require image-tower port to be a separate sub-phase.
- **Router collapse at small scale**: 41-token sequences, modest robot-data scale. Expert-choice routing + aggressive aux loss.
- **Compute for Phase 7**: ~$1–3k on cloud H100/A100. Budget determines whether this is a 2-week or 4-week phase.
- **License stacking**: AgiBot is CC BY-NC-SA; DROID/LIBERO are permissive. Starting with permissive-only for commercial-clean first release.
