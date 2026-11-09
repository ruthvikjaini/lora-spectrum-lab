# Research notes — lora-spectrum-lab (Day 16, 2026-11-09)

## Trending-first check (2026-11-09, ~00:00 ET)

GitHub daily/weekly trending is again dominated by skill/prompt packs and
agent frameworks/platforms, with no repo offering a genuine measurable
one-night CPU weakness:

- Skill packs: `anthropics/skills` (+792/day), `claude-mem` (~115K),
  `superpowers`, `hallmark`, `cloudflare/security-audit-skill`.
- Agent frameworks/platforms: `deepseek-ai/deepseek-harness` (276K, +8.2K/7d),
  `stablyai/orca` (119K), `Ebony-Vinyl/dsh-our-free-model` (66K, +11K/7d),
  `meta/astryx` (+875/day, design system for agent UI),
  `vercel-labs/agent-browser`, `milind-soni/OpenMausBot`.
- CPU inference: `synchopate/llama-modchip` (470 stars, +257/day — runs a
  284B DeepSeek model on a 64GB CPU) and `FareedKhan-dev/kimi-k3-in-c`
  (11.3K) both need multi-hundred-GB checkpoints that cannot be fetched
  or benchmarked in one night; no measurable weakness to optimize.
- Papers-with-code: `google-research/timesfm` (35K, needs a checkpoint),
  `LearningCircuit/local-deep-research` (needs a real LLM),
  `chenxiachan/textfold` (a UI folding tool, no benchmarkable weakness).

No trending repo picked — paper fallback per the standing rule.

## Paper chosen

Liu, Wang, Dong & Lei (NYU / ISTA), **"Spectral and Subspace Dynamics of
LoRA in Supervised Fine-Tuning and Reinforcement Learning"**,
arXiv:2611.08414v1, submitted 5 Nov 2026. Verified on the arXiv abstract
page and the full HTML version on 2026-11-09. No paper code was read or
used; the implementation is from the equations in the HTML text.

Why this paper:
- Very current (4 days old), and it opens a new thread for this profile:
  PEFT/LoRA training dynamics — nothing in Days 1–15 covers LoRA
  spectra, subspace rotation, or stage-specific optimizers.
- It has a sharp, testable mechanism (group-mean baseline subtraction)
  plus a concrete optimizer intervention (norm-preserving spectral
  flattening, Muon-style), so the project is a reproduction *and* a
  build, not just a measurement.
- The paper's own minimal model (quadratic multi-index) is exactly
  reproducible on CPU, and its transformer experiments scale down to a
  tiny transformer + verifiable arithmetic task.

## Paper facts used (all from the arXiv HTML, Sec. 2–4)

- LoRA increment `dW = (alpha/r) B A`; diagnostics: effective rank
  `erank = exp(-sum p_i log p_i)`, `p_i = sigma_i^2 / sum sigma_j^2`;
  subspace overlaps `ovlp_L = (1/r)||U1^T U2||_F^2`,
  `ovlp_R = (1/r)||V1^T V2||_F^2` (smaller = faster rotation).
- Ladder SFT → RFT (weights `r_i`) → RFT-B (weights `r_i - rbar`) →
  GRPO (online + clipping + fixed-length norm, no KL). LoRA r=16,
  alpha=32, A0 Kaiming-uniform, B0=0, Adam, eta in
  {5e-6, 5e-5, 5e-4, 5e-3}; diagnostics averaged across adapters.
- Fig. 1 claim: GRPO/RFT-B erank decreases at all rates; SFT/RFT erank
  increases at intermediate rates (from ~step 25). Fig. 2 claim: GRPO/RFT-B
  rotate faster (ovlp vs step-5 reference falls faster) at small rates.
- Sec. 3 quadratic model: `f_W(x) = (||W^T x||^2 - ||W||_F^2)/sqrt(2)`,
  teacher `F_delta = U0 Sigma0 + delta Z`, correlated Gaussian pairs
  (rho_a along pretrained directions, rho_o along newly useful ones, q
  independent coords), losses `1/4(eps+^2 + eps-^2)` vs
  `1/4((eps+ - epsbar)^2 + (eps- - epsbar)^2)`, SGD on both factors.
  Theorem 3.1: without a baseline erank increases at eta_H and decreases
  at eta_L; with a baseline it decreases at both; baseline gives smaller
  `ovlp_L(t, t+1)` at every adjacent update; the risk-favorable rate is
  higher without a baseline and lower with one.
- Sec. 4 flattening: `M(D) = (||D||_F/sqrt(rank D)) D [(D^T D)^+]^{1/2}`,
  norm-preserving, replaces nonzero singular values by
  `||D||_F/sqrt(rank D)`. SFT config: B uses SGD + M, A uses Adam.
  RL config: Adam both; B uses `P_t D + M((I - P_t) D)` with
  `P_t = B_t (B_t^T B_t)^+ B_t^T`. Table 1 (GSM8K): SFT Adam/Adam
  22.24/42.99 (1B/3B), Adam/M(SGD) 22.82/45.72 (best); RL Adam/Adam
  7.61/42.36, Adam/M_perp(Adam) 10.00/47.66 (best). Paper numbers are
  kept separate from our tiny-model measurements in the README.

## Papers considered and rejected

- Jia et al., "Reward Budgeting Reduces Premature Convergence in RL for
  LLM Reasoning" (arXiv:2611.05942, v3 6 Nov 2026, NeurIPS 2026) — a
  clean GRPO advantage variant (fixed reward budget per prompt, +12.50
  AMC23 / +10.00 AIME). Strong candidate, held for a future night; the
  LoRA paper was preferred because it opens a new (PEFT/spectral) thread
  rather than a seventh GRPO-variant project.
- Liu, "No Free Lunch in RLVR" (arXiv:2610.39295) — held a seventh night;
  analysis paper, no single trainable mechanism for one night.
- Bangera & Gupta, "RLVR Sharpens the Pencil, Not the Mind"
  (arXiv:2611.02574, 2 Nov 2026) — measurement paper overlapping the Liu
  held item; noted for a future night.
- Octopus multi-agent RL with latent-KV DAGs (arXiv:2611.06956) — needs a
  real model + GPU. SwarmEvolve — needs an LLM. KAPPO sharpening
  (arXiv:2611.03146) — held (same RLVR cluster).

## Testbed decisions (and where we deviate from the paper)

- Quadratic model: d=512, k=8 index directions, k_new=8 newly useful,
  q=496 task-irrelevant coords (large q approaches the paper's
  asymptotic regime, where the window grows with q); delta=0.05;
  rho_a=0.7, rho_o=-0.5; SGD; n_pairs=32 per step. Population risk is
  exact: `||W W^T - F F^T||_F^2` for x ~ N(0, I) (derived and
  Monte-Carlo-checked in the test suite).
- Transformer: 1-layer single-head causal transformer, d=64, MLP 128,
  vocab 15, manual backprop, verified against finite differences for
  every parameter including LoRA factors. Base W0 is a fully fine-tuned
  transformer (1500 steps, Adam 3e-3, all-tier SFT) reaching greedy
  1.00 / 0.31 / 0.28 (easy/medium/hard), 0.53 overall on held-out
  arithmetic — moderate, with headroom on medium/hard. LoRA r=8,
  alpha=16 on Wq, Wv, Wo, W1 (r=8 rather than the paper's 16, so the
  spectrum has room to evolve without saturating at the rank cap; the
  first-step Adam jump already pushes erank near r with r=4).
- The ladder's RFT/RFT-B/GRPO all start from the same base + same LoRA
  init (same seed), as in the paper. Deviation: the paper's four settings
  all start from the raw pretrained base, while ours start from the
  SFT-pretrained base — a raw tiny base cannot do arithmetic at all, so
  GRPO/RFT would have no signal. The comparison that matters (RFT vs
  RFT-B, same completions, only the baseline differs) is preserved
  exactly.
- GRPO loss normalizer: token losses summed over completion tokens and
  divided by the batch total (one global normalizer, Dr. GRPO style),
  not per-response length.

## Bugs and design fixes found during the night

1. `spectral_flatten` / `column_projector` counted numerical-zero
   singular values (~1e-17) as rank (`s > 0`), so a rank-1 input was
   flattened to rank 2. Fixed with the standard tolerance
   `max(shape) * eps * s[0]`.
2. `overlap_of_deltas` used all `min(d_o, d_i)` left singular vectors;
   only the first r span the increment's subspace (rank <= r). Fixed to
   slice `U[:, :r]` / `Vt[:r, :]`.
3. Layernorm backward divided by the sequence length T instead of the
   feature dimension d. Caught by the finite-difference gradient check
   (embedding gradient error 0.056).
4. `parse_final` used the FIRST '=' instead of the last, so multi-step
   completions parsed to None. Fixed to use the last '='.
5. **Adam sign bug (the big one):** `AdamState.direction` returned the
   gradient direction, and the trainer *added* `eta * direction`, so
   every run climbed the loss (3.0 -> 26 in 100 steps). Fixed by making
   `direction` return the descent direction `-mhat/(sqrt(vhat)+eps)`,
   matching the paper's `B_{t+1} = B_t + eta D_t` convention. Added a
   regression test (loss must decrease on a fixed batch); the
   finite-difference tests never covered the optimizer.
6. **PAD pollution in generation:** `evaluate`/`sample_and_score`
   right-padded prompts with PAD and generated after the padding, so
   the model conditioned on an essentially untrained PAD embedding.
   Base accuracy read 0.20 (easy 0.00) with the bug vs 0.53 (easy 1.00)
   without it. Fixed with `generate_grouped` (group prompts by length,
   generate each group unpadded). Training batches are unaffected
   (real tokens never attend to trailing PADs under the causal mask).
7. Quadratic model: eta >= 0.02 diverges (the loss is quartic in W, so
   gradients grow superlinearly once ||W|| grows). Stable range found
   by sweep: 1e-5 .. 1e-3 at d=512. Added a divergence guard
   (non-finite or ||B||, ||A|| > 50 stops the run and records it).
8. The teacher is the paper's literal form `F_delta = U0 Sigma0 + delta Z`
   (no V0^T on the first term); at delta=0 the teacher and the student
   make identical predictions because `f_W` depends on W only through
   `W W^T` and `||W||_F^2`.

## Key findings (from pilots; final numbers in results/)

- Quadratic model (d=512, seed 0): baseline subtraction lowers the
  post-startup erank slope at every eta and lowers adjacent-step overlap
  at every eta (faster rotation) — the mechanism direction is right —
  but the absolute "erank decreases at both rates" holds only at low
  rates, and both rules prefer the highest stable eta for risk. Partial
  reproduction of Theorem 3.1; the paper's theorem is asymptotic in q.
- Transformer ladder (lr 3e-3, 200 steps, seed 0): RFT-B erank decreases
  after step ~20 (2.22 at step 200, down from a 3.43 peak) and rotates
  fastest (ovlp5 0.309 vs SFT 0.387); SFT/RFT/GRPO erank rises late.
  BUT unclipped RFT-B collapses accuracy (0.53 -> 0.10 at lr 3e-3,
  0.53 -> 0.23 at lr 1e-3) while clipped GRPO stays stable (0.50).
  Mechanism: the negative weights push down incorrect completions with
  no clipping, and incorrect multi-step completions share correct
  intermediate derivations, so the push destroys the derivation skill.
  GRPO's clipping stops the gradient once a completion's ratio leaves
  [1-eps, 1+eps], which is what keeps it stable. This is a boundary
  condition on the paper's ladder, not a refutation of the spectral
  mechanism: the spectrum/rotation signatures reproduce while the
  accuracy benefit inverts on a small model.
- SFT erank is U-shaped (Adam first-step jump to ~5.7, concentration to
  ~3.3-3.6, then a slow rise) — the paper's "increase from step 25" is
  the late phase of that U on this testbed.
