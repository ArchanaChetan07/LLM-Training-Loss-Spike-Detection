#!/usr/bin/env bash
# Small-model instability sweep (own runs arm).
#   controls (lr 1e-3): fit the cluster-aware null rho(s); never used for testing
#   unstable grid: Post-LN, no clipping, high LR -> spikes with and without recovery
cd "$(dirname "$0")/.."
COMMON="--steps 3000 --d 128 --layers 4 --seq 128 --batch 32 --warmup 100 --clip 1e9"
run() { n=$1; shift; [ -f "data/small/$n.parquet" ] || python scripts/train_small.py $COMMON "$@" --out "$n"; }
for seed in 0 1; do
  run ctrl_preln_s$seed  --lr 1e-3 --beta2 0.95  --seed $seed
  run ctrl_postln_s$seed --lr 1e-3 --beta2 0.999 --seed $seed --post-ln
done
run ctrl_postln_s2 --lr 1e-3 --beta2 0.95 --seed 2 --post-ln
for seed in 0 1 2; do
  for lr in 6e-3 1e-2 4e-3 8e-3; do
    for b2 in 0.999 0.95; do
      run "pl_lr${lr}_b2${b2}_s${seed}" --lr $lr --beta2 $b2 --seed $seed --post-ln
    done
  done
done
