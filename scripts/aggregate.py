"""Aggregate all experiment results into summary tables (stdout).

Reads results/quadratic.json, results/ladder_*.json and results/opt_*.json
and prints seed-mean tables: quadratic erank slopes / adjacent overlaps /
final risks, ladder erank endpoints + late slopes + ovlp5 + final accuracy,
and optimizer-config accuracy/erank/rotation at each learning rate.
Used to fill in the README numbers; scripts/crossfoot.py re-checks them.
"""

import glob
import json
import os

import numpy as np


def mean(xs):
    return float(np.mean(xs))


def sem(xs):
    xs = np.asarray(xs, dtype=float)
    return float(xs.std(ddof=1) / np.sqrt(len(xs))) if len(xs) > 1 else 0.0


def quad_table():
    d = json.load(open("results/quadratic.json"))
    print("== quadratic model: 3-seed means (600 steps, n_pairs=32) ==")
    print(f"{'eta':>8} {'slope_raw':>9} {'slope_base':>10} {'ovlp_raw':>9} "
          f"{'ovlp_base':>9} {'risk_raw':>9} {'risk_base':>9}")
    for eta in sorted({r["eta"] for r in d["runs"]}):
        raw = [r for r in d["runs"] if r["eta"] == eta and not r["baseline"]]
        base = [r for r in d["runs"] if r["eta"] == eta and r["baseline"]]
        print(f"{eta:>8} "
              f"{mean([r['erank_slope_per_100_steps_post100'] for r in raw]):>9.4f} "
              f"{mean([r['erank_slope_per_100_steps_post100'] for r in base]):>10.4f} "
              f"{mean([r['adj_overlap_mean'] for r in raw]):>9.5f} "
              f"{mean([r['adj_overlap_mean'] for r in base]):>9.5f} "
              f"{mean([r['risk_end'] for r in raw]):>9.4f} "
              f"{mean([r['risk_end'] for r in base]):>9.4f}")
    # Per-pair win counts for the two mechanism claims.
    n_slope = n_ovlp = tot = 0
    for eta in sorted({r["eta"] for r in d["runs"]}):
        for seed in (0, 1, 2):
            raw = [r for r in d["runs"] if r["eta"] == eta
                   and not r["baseline"] and r["seed"] == seed][0]
            base = [r for r in d["runs"] if r["eta"] == eta
                    and r["baseline"] and r["seed"] == seed][0]
            tot += 1
            n_slope += int(base["erank_slope_per_100_steps_post100"]
                           < raw["erank_slope_per_100_steps_post100"])
            n_ovlp += int(base["adj_overlap_mean"] < raw["adj_overlap_mean"])
    print(f"pairs with baseline slope < raw slope: {n_slope}/{tot}")
    print(f"pairs with baseline adj-ovlp < raw adj-ovlp: {n_ovlp}/{tot}")


def ladder_table():
    print("\n== ladder: 2-seed means (300 steps) ==")
    print(f"{'setting':>6} {'lr':>6} {'erank@300':>9} {'erank slope':>11} "
          f"{'ovlp5@300':>10} {'acc':>6} {'acc_easy':>8} {'acc_med':>7}")
    for setting in ("sft", "rft", "rft-b", "grpo"):
        for lr in (0.001, 0.003, 0.01):
            files = glob.glob(f"results/ladder_{setting}_{lr}_seed*.json")
            if not files:
                continue
            ds = [json.load(open(f)) for f in sorted(files)]
            # erank at step 300 and late slope (steps 100..300).
            e300, slopes, o300, accs = [], [], [], []
            for d in ds:
                h = d["history"]
                e300.append(h[-1]["erank_mean"])
                xs = np.array([r["step"] for r in h if r["step"] >= 100],
                              dtype=float)
                ys = np.array([r["erank_mean"] for r in h if r["step"] >= 100])
                slopes.append(float(np.polyfit(xs, ys, 1)[0] * 100))
                o300.append(h[-1]["ovlp_L_vs_step5_mean"])
                accs.append(d["final_acc"])
            print(f"{setting:>6} {lr:>6} {mean(e300):>9.3f} "
                  f"{mean(slopes):>11.4f} {mean(o300):>10.3f} "
                  f"{mean([a['all'] for a in accs]):>6.3f} "
                  f"{mean([a['easy'] for a in accs]):>8.3f} "
                  f"{mean([a['medium'] for a in accs]):>7.3f}")


def opt_table():
    print("\n== optimizer configs (paper Sec. 4) ==")
    for stage in ("sft", "grpo"):
        print(f"-- {stage} stage --")
        print(f"{'config':>6} {'lr':>6} {'acc':>6} {'erank@200':>9} "
              f"{'ovlp5@200':>10}")
        for f in sorted(glob.glob(f"results/opt_{stage}_*.json")):
            d = json.load(open(f))
            cfg = d["config"]
            h = d["history"]
            print(f"{cfg['b_mode']:>6} {cfg['lr']:>6} "
                  f"{d['final_acc']['all']:>6.3f} {h[-1]['erank_mean']:>9.3f} "
                  f"{h[-1]['ovlp_L_vs_step5_mean']:>10.3f}")


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)) + "/..")
    quad_table()
    ladder_table()
    opt_table()
