"""Pretrain the frozen base transformer (full fine-tuning, no LoRA).

The base is the analogue of the paper's pretrained LLM: it must know the
arithmetic format well enough that LoRA stages have a learning signal,
but not so well that there is no headroom left.  Target: moderate
mixed-tier greedy accuracy (~0.5).

Usage:
    python scripts/pretrain_base.py --steps 1200 --lr 0.003 --seed 0 \\
        --out results/base_seed0.npz
"""

import argparse
import json
import time

import numpy as np

from lora_spectrum.tasks import batchify, make_dataset
from lora_spectrum.training import AdamState, apply_full_step, ce_loss, evaluate
from lora_spectrum.transformer import Transformer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1200)
    ap.add_argument("--lr", type=float, default=0.003)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--n-per-tier", type=int, default=600)
    ap.add_argument("--eval-every", type=int, default=200)
    ap.add_argument("--out", default="results/base_seed0.npz")
    ap.add_argument("--log", default="results/pretrain_seed0.json")
    args = ap.parse_args()

    rng = np.random.default_rng(1000 + args.seed)
    model = Transformer(vocab=15, d=64, hidden=128, max_len=32, seed=args.seed)
    train = make_dataset(rng, args.n_per_tier)
    held = make_dataset(np.random.default_rng(555), 32)
    adam = AdamState()
    hist = []
    t0 = time.time()
    for step in range(1, args.steps + 1):
        idx = rng.integers(0, len(train), size=args.batch)
        batch = [train[i] for i in idx]
        tok, mask = batchify(batch)
        loss, dlogits, cache = ce_loss(model, tok, mask)
        grads, _ = model.backward(dlogits, cache)
        apply_full_step(model, grads, adam, args.lr)
        if step % args.eval_every == 0 or step == 1:
            acc = evaluate(model, held)
            hist.append({"step": step, "loss": round(loss, 4),
                         "acc": {k: round(v, 3) for k, v in acc.items()},
                         "wall_s": round(time.time() - t0, 1)})
            print(f"step {step} loss {loss:.4f} acc {acc} "
                  f"({time.time() - t0:.0f}s)", flush=True)
    np.savez(args.out, **{k: v for k, v in model.p.items()})
    with open(args.log, "w") as fh:
        json.dump({"config": vars(args), "history": hist}, fh, indent=2)
    print(f"saved base to {args.out}")


if __name__ == "__main__":
    main()
