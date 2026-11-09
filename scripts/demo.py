"""Quick demo: the paper's mechanism in miniature (runs in ~20 seconds).

1. Quadratic multi-index model (paper Sec. 3): at a fixed learning rate,
   baseline subtraction lowers the post-startup effective-rank slope and
   lowers the adjacent-step subspace overlap (faster rotation).
2. Transformer ladder (paper Sec. 2): a few SFT LoRA steps on the frozen
   base, showing the effective rank and the step-5 overlap moving.

Usage: python scripts/demo.py
"""

import numpy as np

from lora_spectrum.diagnostics import erank_of_delta, overlap_of_deltas
from lora_spectrum.quadratic import QuadraticModel


def quad_mini(eta=3e-4, steps=300, seed=0):
    out = {}
    for baseline in (False, True):
        m = QuadraticModel(seed=seed)
        eranks, risks, ov = [], [], []
        prev = None
        for t in range(steps + 1):
            if t % 50 == 0:
                eranks.append(erank_of_delta(m.B, m.A, 1.0, m.r))
                risks.append(m.risk())
            if prev is not None:
                o, _ = overlap_of_deltas(prev[0], prev[1], m.B, m.A, 1.0, m.r)
                ov.append(o)
            prev = (m.B.copy(), m.A.copy())
            if t < steps:
                m.step(eta=eta, baseline=baseline, n_pairs=16)
        xs = np.arange(0, steps + 1, 50)
        slope = float(np.polyfit(xs[2:], eranks[2:], 1)[0] * 100)
        out[baseline] = (eranks, risks, float(np.mean(ov)), slope)
    return out


def main():
    print("== Quadratic multi-index model (paper Sec. 3), eta=3e-4, 300 steps ==")
    res = quad_mini()
    for baseline in (False, True):
        eranks, risks, ov, slope = res[baseline]
        print(f"  baseline={'on ' if baseline else 'off'}: "
              f"erank {eranks[0]:.2f} -> {eranks[-1]:.2f} "
              f"(post-startup slope {slope:+.3f}/100 steps), "
              f"risk {risks[0]:.3f} -> {risks[-1]:.3f}, "
              f"mean adjacent ovlp_L {ov:.4f}")
    print("  -> baseline subtraction: slower erank growth, smaller adjacent "
          "overlap (faster rotation).")

    print("== Transformer LoRA, 20 SFT steps on the frozen base ==")
    from lora_spectrum.tasks import batchify, make_dataset
    from lora_spectrum.training import (AdamState, apply_full_step,
                                        apply_lora_step, ce_loss)
    from lora_spectrum.transformer import Transformer

    import os
    rng = np.random.default_rng(2000)
    model = Transformer(vocab=15, d=64, hidden=128, max_len=32, seed=0,
                        lora_rank=8, lora_alpha=16.0)
    ckpt = "results/base_seed0.npz"
    if os.path.exists(ckpt):
        # The shipped checkpoint (reproduced by scripts/pretrain_base.py).
        data = np.load(ckpt)
        model.p = {k: data[k] for k in data.files}
    else:
        # Fresh clone: the checkpoint is not shipped (it is a generated
        # artifact), so train a small base on the fly (~6 s) instead.
        print("  (no results/base_seed0.npz — training a small base on the fly)")
        rng_pre = np.random.default_rng(1000)
        train_pre = make_dataset(rng_pre, 600)
        adam_pre = AdamState()
        for step in range(1, 301):
            idx = rng_pre.integers(0, len(train_pre), size=32)
            tok, mask = batchify([train_pre[i] for i in idx])
            loss, dlogits, cache = ce_loss(model, tok, mask)
            grads, _ = model.backward(dlogits, cache)
            apply_full_step(model, grads, adam_pre, 0.003)
    train = make_dataset(rng, 600)
    adam = AdamState()
    for step in range(1, 21):
        idx = rng.integers(0, len(train), size=32)
        tok, mask = batchify([train[i] for i in idx])
        loss, dlogits, cache = ce_loss(model, tok, mask)
        grads, glora = model.backward(dlogits, cache)
        apply_lora_step(model, glora, adam, 0.003)
        if step in (1, 10, 20):
            er = np.mean([erank_of_delta(f["B"], f["A"], 16.0, 8)
                          for f in model.lora.values()])
            print(f"  step {step:>2}: loss {loss:.3f}, mean erank {er:.3f}")
    print("Done. See README.md for the full grids.")


if __name__ == "__main__":
    main()
