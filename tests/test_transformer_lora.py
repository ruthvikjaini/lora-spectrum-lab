"""Finite-difference checks for the transformer backward (incl. LoRA)."""

import numpy as np
import pytest

from lora_spectrum.tasks import batchify, encode
from lora_spectrum.transformer import Transformer


def _ce_loss_and_dlogits(model, tok):
    """Mean next-token cross-entropy over all non-pad positions."""
    logits, cache = model.forward(tok)
    B, T, V = logits.shape
    z = logits[:, :-1, :] - logits[:, :-1, :].max(axis=-1, keepdims=True)
    e = np.exp(z)
    p = e / e.sum(axis=-1, keepdims=True)
    tgt = tok[:, 1:]
    nll = -np.log(p[np.arange(B)[:, None], np.arange(T - 1)[None, :], tgt] + 1e-12)
    loss = nll.mean()
    dlogits = np.zeros_like(logits)
    onehot = np.zeros_like(p)
    onehot[np.arange(B)[:, None], np.arange(T - 1)[None, :], tgt] = 1.0
    dlogits[:, :-1, :] = (p - onehot) / ((B * (T - 1)))
    return loss, dlogits, cache


def _check_grads(model, tok, rtol=1e-4, atol=1e-6):
    loss, dlogits, cache = _ce_loss_and_dlogits(model, tok)
    g, glora = model.backward(dlogits, cache)
    eps = 1e-6

    def num_grad(get, set_):
        orig = get()
        num = np.zeros_like(orig, dtype=np.float64)
        it = np.nditer(orig, flags=["multi_index"])
        for _ in it:
            idx = it.multi_index
            old = orig[idx]
            tmp = orig.copy(); tmp[idx] = old + eps; set_(tmp)
            lp, _, _ = _ce_loss_and_dlogits(model, tok)
            tmp = orig.copy(); tmp[idx] = old - eps; set_(tmp)
            lm, _, _ = _ce_loss_and_dlogits(model, tok)
            num[idx] = (lp - lm) / (2 * eps)
            set_(orig)
        return num

    worst = 0.0
    for name, arr in model.p.items():
        num = num_grad(lambda n=name: model.p[n],
                       lambda v, n=name: model.p.__setitem__(n, v))
        err = np.max(np.abs(num - g[name]))
        worst = max(worst, err)
        assert np.allclose(num, g[name], rtol=rtol, atol=atol), (
            f"grad mismatch for {name}: max abs err {err}")
    for name, fac in model.lora.items():
        for part in ("B", "A"):
            num = num_grad(lambda n=name, q=part: model.lora[n][q],
                           lambda v, n=name, q=part: model.lora[n].__setitem__(q, v))
            err = np.max(np.abs(num - glora[name][part]))
            worst = max(worst, err)
            assert np.allclose(num, glora[name][part], rtol=rtol, atol=atol), (
                f"LoRA grad mismatch for {name}.{part}: max abs err {err}")
    return worst


def test_transformer_backward_no_lora():
    rng = np.random.default_rng(0)
    model = Transformer(vocab=15, d=8, hidden=16, max_len=16, seed=0)
    tok = rng.integers(0, 15, size=(2, 6))
    _check_grads(model, tok)


def test_transformer_backward_with_lora():
    rng = np.random.default_rng(1)
    model = Transformer(vocab=15, d=8, hidden=16, max_len=16, seed=0,
                        lora_rank=2, lora_alpha=4.0)
    tok = rng.integers(0, 15, size=(2, 6))
    _check_grads(model, tok)


def test_lora_init_is_identity():
    """B0 = 0 means the LoRA model equals the base model at init."""
    m0 = Transformer(vocab=15, d=16, hidden=32, max_len=16, seed=3)
    m1 = Transformer(vocab=15, d=16, hidden=32, max_len=16, seed=3,
                     lora_rank=2, lora_alpha=4.0)
    rng = np.random.default_rng(4)
    tok = rng.integers(0, 15, size=(2, 8))
    l0, _ = m0.forward(tok)
    l1, _ = m1.forward(tok)
    assert np.allclose(l0, l1, atol=1e-12)


def test_generate_shapes_and_greedy():
    model = Transformer(vocab=15, d=8, hidden=16, max_len=16, seed=0,
                        lora_rank=2, lora_alpha=4.0)
    rng = np.random.default_rng(5)
    prompt = rng.integers(0, 15, size=(3, 4))
    out = model.generate(prompt, max_new=5, rng=rng, temperature=0.0)
    assert out.shape == (3, 9)
    assert np.array_equal(out[:, :4], prompt)
    out2 = model.generate(prompt, max_new=5, rng=rng, temperature=1.0)
    assert out2.shape == (3, 9)


def test_adam_direction_descends():
    """AdamState.direction must be a descent direction (regression test).

    A sign error here makes every training run climb the loss; the
    finite-difference tests do not cover the optimizer.
    """
    from lora_spectrum.tasks import batchify, make_dataset
    from lora_spectrum.training import AdamState, apply_full_step, ce_loss

    rng = np.random.default_rng(1000)
    model = Transformer(vocab=15, d=16, hidden=32, max_len=16, seed=0)
    train = make_dataset(rng, 100)
    tok, mask = batchify([train[i] for i in range(32)])
    adam = AdamState()
    loss0, _, _ = ce_loss(model, tok, mask)
    for _ in range(10):
        loss, dlogits, cache = ce_loss(model, tok, mask)
        grads, _ = model.backward(dlogits, cache)
        apply_full_step(model, grads, adam, 1e-3)
    loss1, _, _ = ce_loss(model, tok, mask)
    assert loss1 < loss0, f"loss did not decrease: {loss0} -> {loss1}"


def test_generate_grouped_matches_single():
    """Grouped generation must equal per-item generation (no PAD pollution)."""
    from lora_spectrum.training import generate_grouped

    model = Transformer(vocab=15, d=8, hidden=16, max_len=16, seed=0,
                        lora_rank=2, lora_alpha=4.0)
    rng = np.random.default_rng(5)
    prompts = [[3, 10, 4, 12], [1, 11, 2, 10, 3, 12], [5, 10, 6, 12],
               [2, 11, 3, 10, 4, 11, 5, 12]]
    grouped = generate_grouped(model, prompts, max_new=6, rng=rng,
                               temperature=0.0)
    for p, g in zip(prompts, grouped):
        single = model.generate(np.array([p]), max_new=6, rng=rng,
                                temperature=0.0)[0]
        assert list(g) == list(single), f"grouped {g} != single {single}"


def test_encode_and_parse_roundtrip():
    toks, final = encode(3, ["+", "x"], [4, 2])
    # 3 + 4 = 7, 7 x 2 = 14
    assert final == 14
    from lora_spectrum.tasks import parse_final, prompt_len_of
    assert parse_final(toks) == 14
    assert prompt_len_of(toks) == 6  # 3 + 4 x 2 =
    toks2, f2 = encode(9, ["x"], [9])
    assert f2 == 81
    assert parse_final(toks2) == 81
