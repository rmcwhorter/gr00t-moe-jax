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

- [ ] `BasicTransformerBlock` in Flax NNX (self/cross-attention + GeGLU FFN + AdaLN)
- [ ] `AdaLayerNorm` with timestep conditioning
- [ ] `TimestepEncoder` (sinusoidal pos embed → MLP)
- [ ] `DiT` top-level module (16 blocks, 32 heads × 48 head_dim = 1536 inner)
- [ ] `AlternateVLDiT` variant (every-other-block text attention)
- [ ] `MultiEmbodimentActionEncoder` / `CategorySpecificLinear` / `CategorySpecificMLP`
- [ ] Sinusoidal position embedding
- [ ] Unit tests against randomly-initialized parity with PyTorch

## Phase 2 — Port the Qwen3-VL backbone (1–3 weeks)

Cosmos-Reason2-2B is stock `Qwen3VLForConditionalGeneration` — not a custom NVIDIA fork.

- [ ] Text transformer: RoPE + NTK scaling, GQA, SwiGLU, RMSNorm, tied embeddings
- [ ] Vision tower (Qwen3-VL native-aspect-ratio ViT)
- [ ] Image/text token interleaving + attention mask construction
- [ ] KV cache for inference (Flax scan-based implementation)
- [ ] Weight converter: HuggingFace safetensors → Flax pytree
- [ ] Numerical parity tests vs HuggingFace `transformers` Qwen3-VL (fp32 tolerance)

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
