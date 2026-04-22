#!/usr/bin/env bash
# M-sweep with generalized teacher-forcing alpha=0.5
set -e
cd "$(dirname "$0")"
mkdir -p results_p_sweep
for M in 4 5 10; do
    LOG="results_p_sweep/run_M${M}_a0p5.log"
    echo "================ starting M=$M (alpha=0.5) ================" | tee "$LOG"
    python experiment_p_sweep.py \
        --epochs 1500 --P_list 2 3 4 --M "$M" \
        --T_gen 5000 --seed 42 \
        --share_B --freeze_B \
        --alpha 0.5 \
        --tag "M${M}_a0p5" \
        2>&1 | tee -a "$LOG"
done
echo "All alpha=0.5 M-sweeps complete."
