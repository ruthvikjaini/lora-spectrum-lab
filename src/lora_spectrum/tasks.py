"""Verifiable multi-step arithmetic task for the transformer testbed.

Format (tokens are single characters; digits are separate tokens):
  easy   (1 op):  "a op b = r"
  medium (2 ops): "a op1 b op2 c = inner = final"
  hard   (3 ops): "a op1 b op2 c op3 d = i1 = i2 = final"

Operands are 0-9, ops are + and x (keeps every intermediate non-negative).
The prompt is everything up to and including the first '='; the completion
is the derivation plus EOS, and the loss/reward is computed on the
completion only.  Reward is binary: 1 iff the final number after the last
'=' equals the gold final answer.

Token ids: digits 0-9 -> 0-9, '+' -> 10, 'x' -> 11, '=' -> 12,
EOS -> 13, PAD -> 14.  Vocab size 15.
"""

import numpy as np

VOCAB = 15
PAD = 14
EOS = 13
EQ = 12
PLUS = 10
TIMES = 11

OPS = ["+", "x"]


def _num_tokens(n):
    return [int(ch) for ch in str(n)]


def _apply(op, a, b):
    return a + b if op == "+" else a * b


def encode(a, ops, operands):
    """Full token sequence for a chain; also returns the gold final answer.

    a: first operand; ops: list of op chars; operands: remaining operands.
    Example: encode(3, ["+", "x"], [4, 2]) -> "3 + 4 x 2 = 7 = 14".
    """
    vals = [a] + list(operands)
    toks = [a]
    cur = a
    steps = []
    for op, b in zip(ops, operands):
        toks += [PLUS if op == "+" else TIMES, b]
        cur = _apply(op, cur, b)
        steps.append(cur)
    toks.append(EQ)
    for s in steps:
        toks += _num_tokens(s) + [EQ]
    # Remove the trailing '=' added after the last step, then add EOS.
    toks = toks[:-1] + [EOS]
    return toks, steps[-1]


def prompt_len_of(toks):
    """Length of the prompt: tokens up to and including the first '='."""
    return toks.index(EQ) + 1


def sample_problem(rng, tier):
    """Returns (tokens, gold_final). tier in {"easy", "medium", "hard"}."""
    n_ops = {"easy": 1, "medium": 2, "hard": 3}[tier]
    a = int(rng.integers(0, 10))
    ops = [str(rng.choice(OPS)) for _ in range(n_ops)]
    operands = [int(rng.integers(0, 10)) for _ in range(n_ops)]
    toks, final = encode(a, ops, operands)
    return toks, final


def make_dataset(rng, n_per_tier, tiers=("easy", "medium", "hard")):
    items = []
    for tier in tiers:
        for _ in range(n_per_tier):
            toks, final = sample_problem(rng, tier)
            items.append({"tokens": toks, "tier": tier, "final": final,
                          "prompt_len": prompt_len_of(toks)})
    return items


def parse_final(tokens):
    """Final number after the last '=' in a token sequence, or None."""
    toks = list(tokens)
    if EOS in toks:
        toks = toks[:toks.index(EOS)]
    if EQ not in toks:
        return None
    last_eq = len(toks) - 1 - toks[::-1].index(EQ)
    tail = toks[last_eq + 1:]
    if not tail or any(t > 9 for t in tail):
        return None
    return int("".join(str(t) for t in tail))


def batchify(items):
    """Pads items to a (B, T) array; returns (tok, loss_mask, prompt_lens).

    loss_mask is 1 on completion positions (targets), i.e. positions t
    whose token is predicted from position t-1 and lies after the prompt.
    We predict token t from position t-1, so the mask marks target indices
    t >= prompt_len.
    """
    T = max(len(it["tokens"]) for it in items)
    B = len(items)
    tok = np.full((B, T), PAD, dtype=np.int64)
    mask = np.zeros((B, T), dtype=np.float64)
    for i, it in enumerate(items):
        ts = it["tokens"]
        tok[i, :len(ts)] = ts
        mask[i, it["prompt_len"]:len(ts)] = 1.0
    return tok, mask


def completion_logprob_sum(logprobs, tok, mask):
    """Sum of log-probs of completion tokens, per sequence.

    logprobs: (B, T, V) log-probs at each position; token t is predicted
    from position t-1.
    """
    B, T, _ = logprobs.shape
    tgt = tok[:, 1:]
    lp = np.take_along_axis(logprobs[:, :-1, :], tgt[:, :, None], axis=2)[:, :, 0]
    return (lp * mask[:, 1:]).sum(axis=1)


def completion_token_count(mask):
    return mask[:, 1:].sum(axis=1)
