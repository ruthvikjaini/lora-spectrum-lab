"""Quadratic multi-index model grid (paper Sec. 3 / Theorem 3.1).

For each (learning rate, baseline on/off, seed) run, tracks:
  * effective rank of the LoRA increment dW = B A (unit scaling),
  * exact population risk ||W W^T - F F^T||_F^2,
  * adjacent-step left-subspace overlap ovlp_L(t, t+1),
  * training loss.

Paper predictions (Theorem 3.1, informal):
  1. Without a baseline, erank increases at a high rate and decreases at
     a low rate; with a group baseline it decreases at both rates.
  2. Baseline subtraction gives smaller expected ovlp_L(t, t+1) at every
     adjacent update (faster subspace rotation).
  3. The risk-favorable learning rate is higher without a baseline and
     lower with one.

Usage:
    python scripts/run_quadratic.py --etas 0.01 0.05 0.2 0.5 --steps 400 \\
        --seeds 0 1 2 --out results/quadratic.json
"""

import argparse
import json
import time

import numpy as np

from lora_spectrum.diagnostics import erank_of_delta, overlap_of_deltas
from lora_spectrum.quadratic import QuadraticModel


def run_one(eta, baseline, seed, steps=400, n_pairs=8, record_every=10,
             d=512, k=8, k_new=8, r=4):
    model = QuadraticModel(d=d, k=k, k_new=k_new, r=r, seed=seed)
    series = {"step": [], "erank": [], "risk": [], "loss": []}
    snaps = []  # (B copy, A copy) every step, for adjacent/reference overlaps
    t0 = time.time()
    for t in range(steps + 1):
        diverged = (not (np.isfinite(model.B).all() and np.isfinite(model.A).all())
                    or np.linalg.norm(model.B) > 50.0
                    or np.linalg.norm(model.A) > 50.0)
        if diverged:
            return {
                "eta": eta, "baseline": baseline, "seed": seed,
                "steps": steps, "n_pairs": n_pairs, "diverged": True,
                "diverged_at_step": t,
                "wall_s": round(time.time() - t0, 1),
                "series": series, "adj_overlap_mean": None,
                "adj_overlap_windows": {}, "erank_start": series["erank"][0],
                "erank_end": None, "risk_start": series["risk"][0],
                "risk_end": None,
            }
        if t % record_every == 0 or t == steps:
            series["step"].append(t)
            series["erank"].append(erank_of_delta(model.B, model.A, 1.0, model.r))
            series["risk"].append(model.risk())
        snaps.append((model.B.copy(), model.A.copy()))
        if t == steps:
            break
        loss = model.step(eta=eta, baseline=baseline, n_pairs=n_pairs)
        if t % record_every == 0:
            series["loss"].append(loss)
    # Adjacent-step left overlaps ovlp_L(t, t+1), averaged in windows.
    adj = []
    for (B1, A1), (B2, A2) in zip(snaps[:-1], snaps[1:]):
        ovlp_l, _ = overlap_of_deltas(B1, A1, B2, A2, 1.0, model.r)
        adj.append(ovlp_l)
    adj = np.array(adj)
    windows = {}
    for lo in range(0, steps, 100):
        seg = adj[lo:min(lo + 100, steps)]
        windows[f"{lo}-{min(lo + 100, steps)}"] = float(seg.mean()) if len(seg) else None
    # Overlap vs a post-startup reference (step 50), paper Fig. 2 style.
    ref_step = 50
    ref_ovlp = {}
    if steps >= ref_step:
        B0, A0 = snaps[ref_step]
        for t in range(ref_step, steps + 1, 100):
            ovlp_l, _ = overlap_of_deltas(B0, A0, snaps[t][0], snaps[t][1],
                                          1.0, model.r)
            ref_ovlp[str(t)] = float(ovlp_l)
    # Post-startup erank slope (linear fit over steps >= 100).
    post = [(s, e) for s, e in zip(series["step"], series["erank"]) if s >= 100]
    if len(post) >= 2:
        xs = np.array([p[0] for p in post], dtype=float)
        ys = np.array([p[1] for p in post], dtype=float)
        slope = float(np.polyfit(xs, ys, 1)[0] * 100.0)  # per 100 steps
    else:
        slope = None
    return {
        "eta": eta,
        "baseline": baseline,
        "seed": seed,
        "steps": steps,
        "n_pairs": n_pairs,
        "d": d, "k": k, "k_new": k_new, "r": r,
        "diverged": False,
        "wall_s": round(time.time() - t0, 1),
        "series": series,
        "adj_overlap_mean": float(adj.mean()),
        "adj_overlap_windows": windows,
        "ref_overlap_vs_step50": ref_ovlp,
        "erank_slope_per_100_steps_post100": slope,
        "erank_start": series["erank"][0],
        "erank_end": series["erank"][-1],
        "risk_start": series["risk"][0],
        "risk_end": series["risk"][-1],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--etas", type=float, nargs="+", default=[0.0001, 0.0003, 0.001, 0.003, 0.01])
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--n-pairs", type=int, default=32)
    ap.add_argument("--out", default="results/quadratic.json")
    args = ap.parse_args()

    runs = []
    for eta in args.etas:
        for baseline in (False, True):
            for seed in args.seeds:
                res = run_one(eta, baseline, seed, steps=args.steps,
                              n_pairs=args.n_pairs)
                runs.append(res)
                print(f"eta={eta:<6} baseline={int(baseline)} seed={seed} "
                      f"{'DIVERGED@' + str(res.get('diverged_at_step')) if res.get('diverged') else ''}"
                      f"erank {res['erank_start']:.3f}->{res['erank_end'] if res['erank_end'] is None else round(res['erank_end'], 3)} "
                      f"slope/100={None if res.get('erank_slope_per_100_steps_post100') is None else round(res['erank_slope_per_100_steps_post100'], 4)} "
                      f"risk {res['risk_start']:.4f}->{res['risk_end'] if res['risk_end'] is None else round(res['risk_end'], 4)} "
                      f"adj_ovlp={res['adj_overlap_mean']}", flush=True)
    with open(args.out, "w") as fh:
        json.dump({"config": vars(args), "runs": runs}, fh, indent=2)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
