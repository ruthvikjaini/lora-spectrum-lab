# lora-spectrum-lab

Reproduction and extension of **Liu, Wang, Dong & Lei, "Spectral and
Subspace Dynamics of LoRA in Supervised Fine-Tuning and Reinforcement
Learning"** (arXiv:2611.08414, 5 Nov 2026).

The paper asks a simple question with a non-obvious answer: when you train
a LoRA adapter with SFT versus with RL, does the *low-rank increment
itself* evolve differently — and if so, why? The answer it finds is that
group-mean baseline subtraction (the `r_i - rbar` at the heart of GRPO) is
the ingredient that separates the two regimes: it concentrates the singular
spectrum of the LoRA increment and rotates its singular subspaces faster.
This repo reproduces that claim on CPU in two testbeds — the paper's own
quadratic multi-index model, and a tiny transformer trained with LoRA on
verifiable multi-step arithmetic — and then tests the paper's
dynamics-informed optimizer configurations (Sec. 4).

No paper code was read or used. Everything here is an independent
implementation from the equations in the paper. No LLM calls, no GPU,
NumPy only.

## What the paper claims

- **Effective rank** of the LoRA increment `dW = (alpha/r) B A`,
  `erank = exp(-sum p_i log p_i)` with `p_i = sigma_i^2 / sum sigma_j^2`:
  GRPO and RFT-with-baseline (RFT-B) show *decreasing* erank at every
  learning rate, while SFT and RFT show *increasing* erank at intermediate
  rates (paper Fig. 1).
- **Subspace rotation**, `ovlp_L(t1, t2) = (1/r)||U_t1^T U_t2||_F^2`
  (smaller = faster rotation): GRPO and RFT-B rotate faster than SFT and
  RFT at small learning rates (paper Fig. 2, overlap vs step 5).
- **Mechanism** (Sec. 3, quadratic multi-index model): subtracting a
  group-mean baseline cancels the gradient component shared across a
  group's responses, which drives coherent fitting along pretrained
  directions, while retaining most of the within-group contrasts — so
  spectral concentration wins over the redistributing feedback from
  gradient noise, and the update directions rotate faster. Theorem 3.1:
  without a baseline erank increases at a high rate and decreases at a
  low rate; with a baseline it decreases at both; baseline gives smaller
  `ovlp_L(t, t+1)` at every adjacent update; the risk-favorable learning
  rate is higher without a baseline and lower with one.
- **Optimizer** (Sec. 4): a norm-preserving spectral flattening
  `M(D) = (||D||_F/sqrt(rank D)) D [(D^T D)^+]^{1/2}` (a Muon variant).
  SFT config: B uses SGD + `M`, A uses Adam. RL config: Adam for both,
  B uses `P_t D + M((I - P_t) D)` where `P_t` projects onto the column
  space of `B_t` — flatten only the component orthogonal to the current
  subspace, preserving the within-subspace update. Paper Table 1 (GSM8K):
  SFT Adam/Adam 22.24/42.99 (Llama-3.2-1B/3B) vs Adam/M(SGD) **22.82/45.72**;
  RL Adam/Adam 7.61/42.36 vs Adam/M_perp(Adam) **10.00/47.66**.

## What this repo finds (measured, tiny-model numbers)

All numbers below are measured in this repo (see `results/`), not paper
numbers. Paper numbers are quoted separately above.

### 1. Quadratic multi-index model — the mechanism, reproduced in direction

Grid: learning rates {1e-5, 3e-5, 1e-4, 3e-4, 1e-3} x baseline {off, on}
x 3 seeds, 600 steps, 32 pairs/step, d=512, k=8, k_new=8, q=496,
delta=0.05, rho_a=0.7, rho_o=-0.5, SGD. Post-startup erank slope is the
linear fit over steps 100–600; adjacent overlap is the mean of
`ovlp_L(t, t+1)` over all steps.

| eta | erank slope /100 steps, no baseline | erank slope /100 steps, baseline | adjacent ovlp_L, no baseline | adjacent ovlp_L, baseline | final risk, no baseline | final risk, baseline |
|---|---|---|---|---|---|---|
| 1e-5 | -0.008 | -0.014 | 0.99251 | 0.99187 | 0.579 | 0.584 |
| 3e-5 | -0.005 | -0.012 | 0.99251 | 0.99188 | 0.553 | 0.568 |
| 1e-4 | +0.006 | -0.006 | 0.99251 | 0.99188 | 0.485 | 0.520 |
| 3e-4 | +0.037 | +0.014 | 0.99246 | 0.99186 | 0.386 | 0.437 |
| 1e-3 | +0.115 | +0.078 | 0.99193 | 0.99156 | 0.273 | 0.330 |

- **Baseline subtraction suppresses erank growth at every learning
  rate**: the baseline slope is below the raw slope in 14 of 15
  (eta, seed) pairs, and in the 3-seed mean at all 5 rates. At low rates
  the baseline slope is negative (concentration); at high rates it is
  still positive but roughly a third smaller than without a baseline.
  The paper's absolute "decreases at both rates" holds only at low
  rates here — a partial reproduction, plausibly a finite-q effect (the
  theorem is asymptotic in q; ours is q=496).
- **Baseline subtraction rotates faster at every learning rate**:
  adjacent-step overlap is smaller with a baseline in 14 of 15
  (eta, seed) pairs (mean gap ~0.0006). The single exception is the
  highest rate, seed 1, where the baseline overlap exceeds the raw one
  by 0.00008 — noise level. This is the cleanest reproduction of
  Theorem 3.1.
- **Risk**: the raw rule reaches lower risk than the baseline rule at
  every eta, and both rules prefer the highest stable eta. The paper's
  claim that the risk-favorable rate is *lower* with a baseline does not
  reproduce in this regime — consistent with the paper's own caveat that
  baseline subtraction "can come at a cost for prediction" (it shrinks
  the risk reduction from coherent fitting), but the rate-preference flip
  itself is absent here.

### 2. Transformer LoRA ladder — the spectral signatures reproduce; the accuracy benefit inverts

Grid: 4 settings x learning rates {1e-3, 3e-3, 1e-2} x 2 seeds, 300
steps, LoRA r=8, alpha=16 on Wq/Wv/Wo/W1, frozen base (greedy 1.00 /
0.31 / 0.28 easy/medium/hard, 0.53 overall). `erank@30`/`erank@300` are
the mean effective rank across adapters; `ovlp5@300` is the mean
left-subspace overlap vs the step-5 reference (smaller = faster rotation).

| setting | lr | erank@30 | erank@300 | ovlp5@300 | acc | easy | med |
|---|---|---|---|---|---|---|---|
| SFT | 1e-3 | 4.01 | 4.03 | 0.385 | 0.54 | 1.00 | 0.30 |
| SFT | 3e-3 | 3.88 | 4.69 | 0.344 | 0.54 | 1.00 | 0.27 |
| SFT | 1e-2 | 4.63 | 4.95 | 0.278 | 0.53 | 1.00 | 0.28 |
| RFT | 1e-3 | 3.85 | 4.04 | 0.316 | 0.49 | 1.00 | 0.25 |
| RFT | 3e-3 | 3.82 | 4.31 | 0.301 | 0.47 | 1.00 | 0.23 |
| RFT | 1e-2 | 3.85 | 4.91 | 0.232 | 0.36 | 0.81 | 0.16 |
| RFT-B | 1e-3 | 3.57 | 2.84 | 0.304 | 0.18 | 0.41 | 0.08 |
| RFT-B | 3e-3 | 3.24 | 2.66 | 0.298 | 0.06 | 0.16 | 0.02 |
| RFT-B | 1e-2 | 2.48 | 2.42 | 0.264 | 0.00 | 0.00 | 0.00 |
| GRPO | 1e-3 | 4.47 | 4.07 | 0.348 | 0.53 | 1.00 | 0.33 |
| GRPO | 3e-3 | 4.15 | 4.28 | 0.337 | 0.52 | 1.00 | 0.27 |
| GRPO | 1e-2 | 4.15 | 4.36 | 0.256 | 0.36 | 0.77 | 0.19 |

- **Baseline subtraction concentrates the spectrum.** RFT-B's erank
  falls from step 30 to step 300 at lr 1e-3 (3.57 → 2.84) and 3e-3
  (3.24 → 2.66), ending far below SFT (4.03/4.69), RFT (4.04/4.31)
  and GRPO (4.07/4.28) — the paper's Fig. 1 pattern for the
  baseline-subtracted setting. SFT and RFT erank *rise* over the same
  window at every rate. GRPO is mixed: it falls at 1e-3 (4.47 → 4.07)
  but rises at 3e-3 and 1e-2, so the paper's "GRPO decreases at all
  rates" does **not** cleanly reproduce here.
- **Baseline settings rotate fastest at low rates.** At lr 1e-3 and
  3e-3, RFT-B has the lowest overlap vs step 5 (0.304/0.298 vs SFT
  0.385/0.344); GRPO rotates faster than SFT at every rate
  (0.348/0.337/0.256 vs 0.385/0.344/0.278). One wrinkle: RFT also
  rotates faster than SFT at every rate (0.316/0.301/0.232), so the
  paper's clean SFT/RFT grouping does not hold — training on the
  model's own correct completions alone already speeds rotation.
- **The accuracy benefit inverts: unclipped RFT-B collapses.** RFT-B
  accuracy falls from the base's 0.53 to 0.18 / 0.06 / 0.00 across the
  three rates, while clipped GRPO holds 0.53 / 0.52 at low rates. The
  mechanism is visible in the loss: RFT-B's weighted loss goes strongly
  negative (down to about -6.6 at lr 3e-3) as it pushes incorrect
  completions down without bound, and then returns to ~0 once every
  completion is wrong — with all rewards equal to zero, the advantage
  `r - rbar` vanishes and the gradient dies, so the collapse is
  self-locking. Incorrect multi-step completions share correct
  intermediate derivations with correct ones, so the downward push
  destroys the derivation skill itself. GRPO uses the *same*
  advantages but with PPO clipping, which cuts the gradient to zero
  once a completion's probability ratio leaves [1-eps, 1+eps] — that
  clipping is what keeps it stable. So on a small model, the paper's
  spectral mechanism reproduces while its accuracy implication needs
  clipping as a boundary condition. (RFT-B also collapses at lr 1e-3,
  so this is not a high-learning-rate artifact.)

### 3. Dynamics-informed optimizers (paper Sec. 4)

Best learning rate per config (paper Table 1 style), 200 steps, 2-seed
means. Accuracy differences are small (96 held-out items, ~5-point
standard error), so read the ordering, not the gap.

SFT stage:

| config | best lr | acc | erank@200 | ovlp5@200 |
|---|---|---|---|---|
| Adam / Adam | 1e-3 | 0.51 | 3.69 | 0.407 |
| Adam / M(Adam) | 1e-2 | 0.52 | 5.53 | 0.297 |
| Adam / M(SGD) — paper's SFT config | 3e-4 | 0.53 | 4.06 | 0.505 |

GRPO stage:

| config | best lr | acc | erank@200 | ovlp5@200 |
|---|---|---|---|---|
| Adam / Adam | 3e-3 | 0.50 | 4.30 | 0.356 |
| Adam / M(Adam) | 3e-3 | 0.49 | 5.88 | 0.301 |
| Adam / M_perp(Adam) — paper's RL config | 3e-3 | 0.52 | 4.43 | 0.276 |

- The ordering matches the paper in both stages: the paper's
  dynamics-informed config is best (SFT 0.53 > 0.52 > 0.51; GRPO
  0.52 > 0.50 > 0.49), but the gaps are within noise on this testbed.
- The flattening operator does what it claims: applying `M` to the
  full B update raises erank sharply (SFT 5.53–5.94 vs Adam 3.69–4.79;
  GRPO 5.43–5.88 vs 4.30–4.78) — the spectrum is flatter. The
  orthogonal-only variant `M_perp` rotates fastest in the GRPO stage
  (ovlp5 0.276 vs Adam 0.356), which is exactly the targeted effect:
  flattening only the component outside the current column space
  promotes subspace rotation while preserving the within-subspace update.

## How to run it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .[dev]
python -m pytest tests/ -q        # 19 tests, incl. finite-difference gradient checks
python scripts/demo.py             # ~20 s miniature of both testbeds
```

Full grids:

```bash
python scripts/run_quadratic.py --etas 0.00001 0.00003 0.0001 0.0003 0.001 \
    --steps 600 --seeds 0 1 2 --out results/quadratic.json
python scripts/pretrain_base.py --steps 1500 --lr 0.003 --seed 0 \
    --out results/base_seed0.npz
bash scripts/run_all_grids.sh     # ladder + optimizer grids
python scripts/aggregate.py       # summary tables
python scripts/crossfoot.py       # re-checks every README number
```

## Layout

- `src/lora_spectrum/diagnostics.py` — effective rank, subspace overlaps,
  spectral flattening `M(D)`, column projector, orthogonal-component flattening.
- `src/lora_spectrum/quadratic.py` — the paper's Sec. 3 multi-index model
  (exact population risk `||W W^T - F F^T||_F^2`, both loss rules, SGD).
- `src/lora_spectrum/transformer.py` — tiny causal transformer with LoRA
  adapters (manual backprop, finite-difference verified).
- `src/lora_spectrum/tasks.py` — verifiable multi-step arithmetic
  (1/2/3-op chains, binary reward on the final answer).
- `src/lora_spectrum/training.py` — Adam/SGD directions, SFT/RFT/RFT-B/
  GRPO losses, grouped generation, evaluation.
- `scripts/` — experiment drivers, demo, aggregation, crossfoot.
- `docs/RESEARCH.md` — trending check, paper notes, rejected candidates,
  and every bug found during the night.
- `results/` — raw JSON from every run (`quadratic.json`,
  `ladder_*.json`, `opt_*.json`, `pretrain_seed0.json`).

## Limitations (stated plainly)

- Tiny testbeds: a 1-layer d=64 transformer and a quadratic model, not
  1–3B LLMs on GSM8K. Magnitudes differ from the paper; orderings are
  what we compare.
- The ladder's RFT/RFT-B/GRPO start from an SFT-pretrained base (a raw
  tiny base cannot do arithmetic at all), whereas the paper's four
  settings start from the raw pretrained model. The RFT-vs-RFT-B
  isolation (same completions, only the baseline differs) is preserved.
- LoRA rank is 8, not the paper's 16, and only 4 adapters (Wq, Wv, Wo,
  W1) rather than all linear modules.
- The quadratic model's Theorem 3.1 window is asymptotic in q; our q=496
  is finite, and seed variance in the erank slope is large (one of three
  seeds has negative slopes at every eta for both rules).
- 2 seeds for the ladder, 3 for the quadratic model. Standard errors are
  in `results/` via the per-seed files.

## Sources

- Liu, Wang, Dong & Lei, "Spectral and Subspace Dynamics of LoRA in
  Supervised Fine-Tuning and Reinforcement Learning", arXiv:2611.08414v1
  (5 Nov 2026). https://arxiv.org/abs/2611.08414
- Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models",
  arXiv:2106.09685 (the adapter this paper studies).
- Jordan et al., "Muon: An optimizer for hidden layers in neural
  networks" (2024) — the flattening operator's origin.
- Trending check, 2026-11-09: GitHub trending / gitdiscover / findarepo /
  upstar (skill packs and agent frameworks dominate; no optimizable repo
  found — see `docs/RESEARCH.md`).
