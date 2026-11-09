"""Training for the transformer testbed: SFT / RFT / RFT-B / GRPO ladders.

Implements the four training settings of Liu et al. (arXiv:2611.08414),
Section 2.1, which interpolate between SFT and RL:
  (i)   SFT:   supervised training on gold completions.
  (ii)  RFT:   rejection-sampling fine-tuning on model-generated correct
               completions only (weights r_i for binary rewards).
  (iii) RFT-B: RFT with group-mean baseline subtraction (weights r_i - rbar).
  (iv)  GRPO:  RFT-B extended to online sampling, plus policy-ratio
               clipping and fixed-length loss normalization (Dr. GRPO
               style), without KL regularization.

Only the LoRA factors (B, A) are trainable in the ladder; the base weights
W0 are frozen.  A separate full fine-tuning path pretrains the base.

Optimizers: Adam and SGD directions, plus the paper's Section 4
spectral-flattening transforms (M and M-perp) applied to the B-factor
update direction.
"""

import numpy as np

from lora_spectrum.diagnostics import flatten_orthogonal_component, spectral_flatten
from lora_spectrum.tasks import parse_final


# ---------------------------------------------------------------- optimizers

class AdamState:
    """Adam direction provider: direction(grad) = -mhat / (sqrt(vhat) + eps).

    The returned direction is the DESCENT direction (negative Adam step),
    matching the paper's convention B_{t+1} = B_t + eta * D_t where D_t is
    the signed update direction.  Callers add eta * direction to the
    parameters.  State is kept per parameter array (keyed by id).
    """

    def __init__(self, beta1=0.9, beta2=0.999, eps=1e-8):
        self.beta1, self.beta2, self.eps = beta1, beta2, eps
        self.m = {}
        self.v = {}
        self.t = {}

    def direction(self, key, grad):
        g = np.asarray(grad, dtype=np.float64)
        if key not in self.m:
            self.m[key] = np.zeros_like(g)
            self.v[key] = np.zeros_like(g)
            self.t[key] = 0
        self.t[key] += 1
        t = self.t[key]
        self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * g
        self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * (g * g)
        mhat = self.m[key] / (1 - self.beta1 ** t)
        vhat = self.v[key] / (1 - self.beta2 ** t)
        return -mhat / (np.sqrt(vhat) + self.eps)


def sgd_direction(grad):
    return -np.asarray(grad, dtype=np.float64)


def apply_lora_step(model, glora, adam, eta, b_mode="adam", b_opt="adam"):
    """Apply one LoRA update step.

    b_mode: how the B-factor direction is transformed:
      "adam"  -> plain Adam direction (baseline)
      "M"     -> spectral flattening M(D) of the Adam direction
      "Mperp" -> P D + M((I - P) D), P = projector onto col(B_t)
      "sgdM"  -> SGD direction with M applied (paper's SFT config)
    b_opt: base optimizer for B ("adam" or "sgd"); A always uses Adam.
    The A factor always uses the plain Adam direction.
    """
    for name, fac in model.lora.items():
        # A factor: Adam.
        dA = adam.direction(f"A:{name}", glora[name]["A"])
        fac["A"] = fac["A"] + eta * dA
        # B factor: base direction, then optional transform.
        if b_opt == "sgd":
            dB = sgd_direction(glora[name]["B"])
        else:
            dB = adam.direction(f"B:{name}", glora[name]["B"])
        if b_mode == "M":
            dB = spectral_flatten(dB)
        elif b_mode == "Mperp":
            dB = flatten_orthogonal_component(dB, fac["B"])
        elif b_mode == "sgdM":
            dB = spectral_flatten(dB)
        fac["B"] = fac["B"] + eta * dB


def apply_full_step(model, grads, adam, eta):
    """One full fine-tuning Adam step over all base parameters."""
    for name, arr in model.p.items():
        d = adam.direction(name, grads[name])
        model.p[name] = arr + eta * d


# ---------------------------------------------------------------- losses

def ce_loss(model, tok, mask):
    """Mean cross-entropy over completion tokens; returns (loss, dlogits)."""
    logits, cache = model.forward(tok)
    B, T, V = logits.shape
    z = logits[:, :-1, :] - logits[:, :-1, :].max(axis=-1, keepdims=True)
    e = np.exp(z)
    p = e / e.sum(axis=-1, keepdims=True)
    tgt = tok[:, 1:]
    m = mask[:, 1:]
    denom = m.sum()
    nll = -np.log(p[np.arange(B)[:, None], np.arange(T - 1)[None, :], tgt] + 1e-12)
    loss = float((nll * m).sum() / max(denom, 1.0))
    dlogits = np.zeros_like(logits)
    onehot = np.zeros_like(p)
    onehot[np.arange(B)[:, None], np.arange(T - 1)[None, :], tgt] = 1.0
    dlogits[:, :-1, :] = (p - onehot) * (m / max(denom, 1.0))[:, :, None]
    return loss, dlogits, cache


def weighted_ce_loss(model, tok, mask, weights):
    """Sum_i w_i * CE_i, normalized by sum |w_i| (per-sequence CE).

    weights: (B,) per-sequence weights (may be negative, as in RFT-B).
    Returns (loss, dlogits).
    """
    logits, cache = model.forward(tok)
    B, T, V = logits.shape
    z = logits[:, :-1, :] - logits[:, :-1, :].max(axis=-1, keepdims=True)
    e = np.exp(z)
    p = e / e.sum(axis=-1, keepdims=True)
    tgt = tok[:, 1:]
    m = mask[:, 1:]
    nll = -np.log(p[np.arange(B)[:, None], np.arange(T - 1)[None, :], tgt] + 1e-12)
    per_seq = (nll * m).sum(axis=1) / np.maximum(m.sum(axis=1), 1.0)
    denom = np.abs(weights).sum()
    loss = float((weights * per_seq).sum() / max(denom, 1.0))
    dlogits = np.zeros_like(logits)
    onehot = np.zeros_like(p)
    onehot[np.arange(B)[:, None], np.arange(T - 1)[None, :], tgt] = 1.0
    # d/dlogits of w_i * CE_i = w_i * (p - onehot) * m / len_i / denom
    coef = (weights / max(denom, 1.0)) / np.maximum(m.sum(axis=1), 1.0)
    dlogits[:, :-1, :] = (p - onehot) * (m * coef[:, None])[:, :, None]
    return loss, dlogits, cache


def grpo_loss(model, tok, mask, adv, logp_old_tok, clip=0.2):
    """Clipped GRPO surrogate on sampled sequences.

    tok: sampled sequences (prompt + sampled completion); mask: completion
    positions; adv: (B,) group advantages; logp_old_tok: (B, T-1)
    per-token log-probs of the sampled tokens under the sampling policy
    (detached, aligned with positions :-1).
    Returns (loss, dlogits).  The token-level loss is summed over
    completion tokens and divided by the total number of completion
    tokens in the batch (one global normalizer, Dr. GRPO style).
    """
    logits, cache = model.forward(tok)
    B, T, V = logits.shape
    z = logits[:, :-1, :] - logits[:, :-1, :].max(axis=-1, keepdims=True)
    e = np.exp(z)
    p = e / e.sum(axis=-1, keepdims=True)
    tgt = tok[:, 1:]
    m = mask[:, 1:]
    lp = np.log(p[np.arange(B)[:, None], np.arange(T - 1)[None, :], tgt] + 1e-12)
    ratio_tok = np.exp(lp - logp_old_tok)
    # PPO gradient mask: gradient flows only where the unclipped term is
    # the min (standard clipped-surrogate derivative).
    flow = np.ones_like(ratio_tok)
    for i in range(B):
        if adv[i] >= 0:
            flow[i] = np.where(ratio_tok[i] > 1.0 + clip, 0.0, 1.0)
        else:
            flow[i] = np.where(ratio_tok[i] < 1.0 - clip, 0.0, 1.0)
    # Surrogate value (for logging): mean over tokens of -min(rho*A, clip*A).
    surr = np.minimum(ratio_tok * adv[:, None],
                      np.clip(ratio_tok, 1 - clip, 1 + clip) * adv[:, None])
    denom = m.sum()
    loss = float(-(surr * m).sum() / max(denom, 1.0))
    # d/dlogits: -A * rho * (onehot - p) * flow / denom, per token.
    onehot = np.zeros_like(p)
    onehot[np.arange(B)[:, None], np.arange(T - 1)[None, :], tgt] = 1.0
    coef = (-adv[:, None] * ratio_tok * flow * m) / max(denom, 1.0)
    dlogits = np.zeros_like(logits)
    dlogits[:, :-1, :] = coef[:, :, None] * (onehot - p)
    return loss, dlogits, cache


# ---------------------------------------------------------------- evaluation

def generate_grouped(model, prompts, max_new, rng, temperature=0.0):
    """Generate completions for a list of prompt token-lists.

    Groups prompts by length and generates each group with no padding:
    right-padding with PAD tokens would pollute the causal context (the
    model never attends to PAD in training, and the PAD embedding is
    essentially untrained).  Returns a list of full token lists, in the
    input order.
    """
    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
    out = [None] * len(prompts)
    i = 0
    while i < len(order):
        j = i
        while j < len(order) and len(prompts[order[j]]) == len(prompts[order[i]]):
            j += 1
        idxs = order[i:j]
        arr = np.array([prompts[k] for k in idxs], dtype=np.int64)
        gen = model.generate(arr, max_new=max_new, rng=rng,
                             temperature=temperature)
        for k, row in zip(idxs, gen):
            out[k] = list(row)
        i = j
    return out


def evaluate(model, items, batch=64, max_new=16):
    """Greedy accuracy per tier (and overall) on held-out items."""
    rng = np.random.default_rng(0)  # unused; greedy is deterministic
    correct = {}
    total = {}
    prompts = [it["tokens"][:it["prompt_len"]] for it in items]
    gens = generate_grouped(model, prompts, max_new=max_new, rng=rng,
                            temperature=0.0)
    for it, gen in zip(items, gens):
        got = parse_final(gen)
        ok = got == it["final"]
        correct[it["tier"]] = correct.get(it["tier"], 0) + int(ok)
        total[it["tier"]] = total.get(it["tier"], 0) + 1
    acc = {t: correct[t] / total[t] for t in total}
    acc["all"] = sum(correct.values()) / sum(total.values())
    return acc


def sample_and_score(model, items, G, rng, temperature=1.0, max_new=16):
    """Sample G completions per prompt item; score them.

    Returns (tok, mask, rewards, logp_old_tok, prompt_lens):
      tok: (P*G, T) sampled sequences (prompt + completion, PAD-padded);
      mask: (P*G, T) 1 on completion positions;
      rewards: (P*G,) binary rewards (parsed final == gold final);
      logp_old_tok: (P*G, T-1) per-token log-probs of the sampled tokens
        under the sampling policy (computed before any update);
      prompt_lens: (P*G,) prompt lengths.
    """
    P = len(items)
    prompts = [it["tokens"][:it["prompt_len"]] for it in items]
    repeated = [p for p in prompts for _ in range(G)]
    gens = generate_grouped(model, repeated, max_new=max_new, rng=rng,
                            temperature=temperature)
    L = max(len(g) for g in gens)
    gen = np.full((len(gens), L), 14, dtype=np.int64)
    for i, g in enumerate(gens):
        gen[i, :len(g)] = g
    # Completion mask: positions >= prompt_len, up to EOS (inclusive).
    prompt_lens = np.array([len(p) for p in prompts for _ in range(G)])
    mask = np.zeros_like(gen, dtype=np.float64)
    for i in range(gen.shape[0]):
        end = prompt_lens[i] + max_new
        mask[i, prompt_lens[i]:end] = 1.0
        row = gen[i, prompt_lens[i]:end]
        eos = np.where(row == 13)[0]
        if len(eos):
            mask[i, prompt_lens[i] + eos[0] + 1:end] = 0.0
    # Per-token old log-probs (teacher-forced on the sampled sequences).
    logprobs = model.logprobs(gen)
    tgt = gen[:, 1:]
    lp_tok = np.take_along_axis(logprobs[:, :-1, :], tgt[:, :, None],
                                axis=2)[:, :, 0]
    rewards = np.zeros(P * G)
    for i in range(P * G):
        got = parse_final(gen[i])
        rewards[i] = 1.0 if got == items[i // G]["final"] else 0.0
    return gen, mask, rewards, lp_tok, prompt_lens
