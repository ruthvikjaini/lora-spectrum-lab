"""Cross-foot every number in README.md against results/*.json.

Recomputes the headline tables from the raw result files and checks that
the exact markdown rows appear in README.md.  Run:
    python scripts/crossfoot.py
Exit code 0 means every README table row recomputes from the shipped results.
"""

import glob
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__)) + "/.."


def mean(xs):
    return float(np.mean(xs))


def check_row(label, row, readme_text):
    ok = row in readme_text
    print(f"  {'OK ' if ok else 'FAIL'} {label}: {row}")
    return ok


def lr_label(x):
    return {1e-5: "1e-5", 3e-5: "3e-5", 1e-4: "1e-4", 3e-4: "3e-4",
            1e-3: "1e-3", 3e-3: "3e-3", 1e-2: "1e-2",
            3e-4: "3e-4", 0.0003: "3e-4"}[x]


def main():
    os.chdir(ROOT)
    readme = open("README.md").read()
    ok = True

    # ---- quadratic model table ----
    print("quadratic model table:")
    d = json.load(open("results/quadratic.json"))
    for eta in sorted({r["eta"] for r in d["runs"]}):
        raw = [r for r in d["runs"] if r["eta"] == eta and not r["baseline"]]
        base = [r for r in d["runs"] if r["eta"] == eta and r["baseline"]]
        sr = mean([r["erank_slope_per_100_steps_post100"] for r in raw])
        sb = mean([r["erank_slope_per_100_steps_post100"] for r in base])
        orr = mean([r["adj_overlap_mean"] for r in raw])
        ob = mean([r["adj_overlap_mean"] for r in base])
        rr = mean([r["risk_end"] for r in raw])
        rb = mean([r["risk_end"] for r in base])
        eta_s = lr_label(eta)
        row = (f"| {eta_s} | {sr:+.3f} | {sb:+.3f} | {orr:.5f} | {ob:.5f} | "
               f"{rr:.3f} | {rb:.3f} |")
        ok &= check_row(f"quad eta={eta}", row, readme)

    # ---- ladder table ----
    print("ladder table:")
    for setting, sname in (("sft", "SFT"), ("rft", "RFT"),
                           ("rft-b", "RFT-B"), ("grpo", "GRPO")):
        for lr in (0.001, 0.003, 0.01):
            files = sorted(glob.glob(f"results/ladder_{setting}_{lr}_seed*.json"))
            assert files, f"missing ladder files for {setting} {lr}"
            ds = [json.load(open(f)) for f in files]

            def at(step, key):
                vals = []
                for x in ds:
                    cand = [r for r in x["history"] if r["step"] <= step]
                    vals.append(cand[-1][key])
                return mean(vals)

            e30 = at(30, "erank_mean")
            e300 = at(300, "erank_mean")
            o300 = at(300, "ovlp_L_vs_step5_mean")
            acc = mean([x["final_acc"]["all"] for x in ds])
            easy = mean([x["final_acc"]["easy"] for x in ds])
            med = mean([x["final_acc"]["medium"] for x in ds])
            lr_s = lr_label(lr)
            row = (f"| {sname} | {lr_s} | {e30:.2f} | {e300:.2f} | "
                   f"{o300:.3f} | {acc:.2f} | {easy:.2f} | {med:.2f} |")
            ok &= check_row(f"ladder {setting} {lr}", row, readme)

    # ---- optimizer tables (best lr per config) ----
    print("optimizer tables:")
    EXPECTED_OPT = {
        # (stage, config, row label): best_lr
        ("sft", "adam", "Adam / Adam"): 0.001,
        ("sft", "M", "Adam / M(Adam)"): 0.01,
        ("sft", "sgdM", "Adam / M(SGD) — paper's SFT config"): 0.0003,
        ("grpo", "adam", "Adam / Adam"): 0.003,
        ("grpo", "M", "Adam / M(Adam)"): 0.003,
        ("grpo", "Mperp", "Adam / M_perp(Adam) — paper's RL config"): 0.003,
    }
    for (stage, cfg, label), blr in EXPECTED_OPT.items():
        files = sorted(glob.glob(f"results/opt_{stage}_{cfg}_{blr}_seed*.json"))
        assert files, f"missing opt files for {stage} {cfg} {blr}"
        ds = [json.load(open(f)) for f in files]
        acc = mean([x["final_acc"]["all"] for x in ds])
        er = mean([x["history"][-1]["erank_mean"] for x in ds])
        ov = mean([x["history"][-1]["ovlp_L_vs_step5_mean"] for x in ds])
        blr_s = lr_label(blr)
        row = f"| {label} | {blr_s} | {acc:.2f} | {er:.2f} | {ov:.3f} |"
        ok &= check_row(f"opt {stage}/{cfg}", row, readme)

    print("CROSSFOOT " + ("PASS" if ok else "FAIL"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
