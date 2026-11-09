#!/usr/bin/env bash
# Full experiment grids for lora-spectrum-lab.
#   1. Ladder grid: 4 settings x 3 learning rates x 2 seeds, 300 steps.
#   2. Optimizer grid (paper Sec. 4): SFT stage and GRPO stage, 3 configs
#      each, 200 steps.  The SGD-based SFT config gets a lower lr range
#      because its B update is not Adam-normalized.
# Run from the project root: bash scripts/run_all_grids.sh
set -e
cd "$(dirname "$0")/.."
PY=../lora-venv/bin/python

echo "== ladder grid =="
for setting in sft rft rft-b grpo; do
  for lr in 0.001 0.003 0.01; do
    for seed in 0 1; do
      out="results/ladder_${setting}_${lr}_seed${seed}.json"
      if [ ! -f "$out" ]; then
        $PY scripts/run_ladder.py --setting "$setting" --lr "$lr" \
          --seed "$seed" --steps 300 --eval-every 100 --out "$out" \
          2>&1 | tail -1
      fi
    done
  done
done

echo "== optimizer grid, SFT stage =="
# configs: adam (Adam/Adam), M (Adam/M(Adam)), sgdM (Adam/M(SGD))
for cfg in adam M sgdM; do
  if [ "$cfg" = "sgdM" ]; then lrs="0.0003 0.001 0.003"; else lrs="0.001 0.003 0.01"; fi
  for lr in $lrs; do
    for seed in 0 1; do
      out="results/opt_sft_${cfg}_${lr}_seed${seed}.json"
      if [ ! -f "$out" ]; then
        if [ "$cfg" = "sgdM" ]; then bmode="sgdM"; bopt="sgd";
        elif [ "$cfg" = "M" ]; then bmode="M"; bopt="adam";
        else bmode="adam"; bopt="adam"; fi
        $PY scripts/run_ladder.py --setting sft --lr "$lr" --seed "$seed" \
          --steps 200 --eval-every 100 --b-mode "$bmode" --b-opt "$bopt" \
          --out "$out" 2>&1 | tail -1
      fi
    done
  done
done

echo "== optimizer grid, GRPO stage =="
for cfg in adam M Mperp; do
  for lr in 0.003 0.01; do
    for seed in 0 1; do
      out="results/opt_grpo_${cfg}_${lr}_seed${seed}.json"
      if [ ! -f "$out" ]; then
        $PY scripts/run_ladder.py --setting grpo --lr "$lr" --seed "$seed" \
          --steps 200 --eval-every 100 --b-mode "$cfg" --b-opt adam \
          --out "$out" 2>&1 | tail -1
      fi
    done
  done
done

echo "all grids done"
