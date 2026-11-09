"""Transformer LoRA ladder (paper Sec. 2.1): SFT / RFT / RFT-B / GRPO.

All four settings start from the same frozen base checkpoint and the same
LoRA initialization (same seed), and share the same training prompt pool,
so the only difference between RFT and RFT-B is the group-mean baseline
subtraction -- exactly the isolation the paper uses to pin down the
mechanism.  RFT and RFT-B additionally share the same offline completions.

Diagnostics recorded every --diag-every steps: effective rank and
left-subspace overlap vs the step-5 reference, per adapter, averaged
across adapters (paper Sec. 2.1).  Greedy held-out accuracy is recorded
every --eval-every steps.

Usage:
    python scripts/run_ladder.py --setting sft --lr 0.003 --seed 0 \\
        --steps 120 --base results/base_seed0.npz --out results/ladder_sft.json
    python scripts/run_ladder.py --setting grpo --lr 0.003 --seed 0 \\
        --steps 120 --b-mode Mperp --out results/ladder_grpo_mperp.json
"""

import argparse
import json
import time

import numpy as np

from lora_spectrum.diagnostics import erank_of_delta, overlap_of_deltas
from lora_spectrum.tasks import batchify, make_dataset
from lora_spectrum.training import (
    AdamState,
    apply_lora_step,
    ce_loss,
    evaluate,
    grpo_loss,
    sample_and_score,
    weighted_ce_loss,
)
from lora_spectrum.transformer import Transformer

LORA_LAYERS = ("Wq", "Wv", "Wo", "W1")
RANK = 8
ALPHA = 16.0


def load_base(path, seed, rank=RANK, alpha=ALPHA):
    model = Transformer(vocab=15, d=64, hidden=128, max_len=32, seed=seed,
                        lora_rank=rank, lora_alpha=alpha,
                        lora_layers=LORA_LAYERS)
    ckpt = np.load(path)
    model.p = {k: ckpt[k] for k in ckpt.files}
    return model


def diagnostics(model, ref):
    """Mean erank and mean ovlp_L vs reference across adapters."""
    eranks, ovlps = [], []
    for name, fac in model.lora.items():
        eranks.append(erank_of_delta(fac["B"], fac["A"], ALPHA, RANK))
        if ref is not None:
            rb, ra = ref[name]
            ovlp_l, _ = overlap_of_deltas(rb, ra, fac["B"], fac["A"],
                                          ALPHA, RANK)
            ovlps.append(ovlp_l)
    return {
        "erank_mean": float(np.mean(eranks)),
        "erank_per_adapter": {n: round(e, 4) for n, e in
                              zip(model.lora, eranks)},
        "ovlp_L_vs_step5_mean": (float(np.mean(ovlps)) if ovlps else None),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--setting", choices=["sft", "rft", "rft-b", "grpo"],
                    required=True)
    ap.add_argument("--lr", type=float, default=0.003)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=120)
    ap.add_argument("--base", default="results/base_seed0.npz")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--group", type=int, default=8, help="GRPO/RFT group size")
    ap.add_argument("--prompts-per-step", type=int, default=12)
    ap.add_argument("--b-mode", default="adam",
                    choices=["adam", "M", "Mperp", "sgdM"])
    ap.add_argument("--b-opt", default="adam", choices=["adam", "sgd"])
    ap.add_argument("--eval-every", type=int, default=30)
    ap.add_argument("--diag-every", type=int, default=10)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rng = np.random.default_rng(2000 + args.seed)
    model = load_base(args.base, args.seed)
    train = make_dataset(rng, 600)
    held = make_dataset(np.random.default_rng(555), 32)
    adam = AdamState()
    hist = []
    ref = None  # step-5 snapshot of (B, A) per adapter

    # Offline completions shared by RFT and RFT-B (sampled from the base,
    # i.e. the LoRA model with B0 = 0, so this is exactly the base policy).
    offline = None
    if args.setting in ("rft", "rft-b"):
        pool = [train[i] for i in rng.integers(0, len(train), size=192)]
        tok, mask, rewards, _, _ = sample_and_score(
            model, pool, args.group, rng, temperature=1.0)
        offline = {"pool": pool, "tok": tok, "mask": mask,
                   "rewards": rewards}
        print(f"offline completions: {len(pool)} prompts x {args.group}, "
              f"mean reward {rewards.mean():.3f}", flush=True)

    t0 = time.time()
    for step in range(1, args.steps + 1):
        if args.setting == "sft":
            idx = rng.integers(0, len(train), size=args.batch)
            batch = [train[i] for i in idx]
            tok, mask = batchify(batch)
            loss, dlogits, cache = ce_loss(model, tok, mask)
        elif args.setting in ("rft", "rft-b"):
            # One batch = P prompts x G completions from the offline pool.
            P = args.batch // args.group
            pidx = rng.integers(0, len(offline["pool"]), size=P)
            rows = []
            weights = []
            for pi in pidx:
                base = pi * args.group
                rs = offline["rewards"][base:base + args.group]
                rbar = rs.mean()
                for gi in range(args.group):
                    rows.append(base + gi)
                    if args.setting == "rft":
                        weights.append(1.0 if rs[gi] == 1.0 else 0.0)
                    else:
                        weights.append(float(rs[gi] - rbar))
            tok = offline["tok"][rows]
            mask = offline["mask"][rows]
            if args.setting == "rft":
                keep = [i for i, w in enumerate(weights) if w > 0]
                if not keep:
                    continue
                tok, mask = tok[keep], mask[keep]
                loss, dlogits, cache = ce_loss(model, tok, mask)
            else:
                loss, dlogits, cache = weighted_ce_loss(
                    model, tok, mask, np.array(weights))
        else:  # grpo
            idx = rng.integers(0, len(train), size=args.prompts_per_step)
            prompts = [train[i] for i in idx]
            tok, mask, rewards, lp_old, _ = sample_and_score(
                model, prompts, args.group, rng, temperature=1.0)
            adv = np.zeros_like(rewards)
            for pi in range(args.prompts_per_step):
                sl = slice(pi * args.group, (pi + 1) * args.group)
                adv[sl] = rewards[sl] - rewards[sl].mean()
            loss, dlogits, cache = grpo_loss(model, tok, mask, adv, lp_old)
        grads, glora = model.backward(dlogits, cache)
        apply_lora_step(model, glora, adam, args.lr,
                        b_mode=args.b_mode, b_opt=args.b_opt)
        if step == 5:
            ref = {n: (f["B"].copy(), f["A"].copy())
                   for n, f in model.lora.items()}
        if step % args.diag_every == 0 or step == 1:
            d = diagnostics(model, ref)
            rec = {"step": step, "loss": round(loss, 4),
                   "wall_s": round(time.time() - t0, 1), **d}
            if step % args.eval_every == 0:
                acc = evaluate(model, held)
                rec["acc"] = {k: round(v, 3) for k, v in acc.items()}
                print(f"step {step} loss {loss:.4f} erank {d['erank_mean']:.3f} "
                      f"ovlp5 {d['ovlp_L_vs_step5_mean']} acc {acc} "
                      f"({time.time() - t0:.0f}s)", flush=True)
            hist.append(rec)
    acc = evaluate(model, held)
    out = {"config": vars(args), "history": hist,
           "final_acc": {k: round(v, 4) for k, v in acc.items()},
           "wall_s": round(time.time() - t0, 1)}
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"final acc {acc} | wrote {args.out}")


if __name__ == "__main__":
    main()
