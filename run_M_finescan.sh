#!/usr/bin/env bash
# Fine M-scan: M in {10, 11, 14, 15} x P in {2..6}, both vanilla & ortho.
# 1000 epochs, alpha=1.0, shared frozen B per M.  Writes to results_M_finescan/.
set -e
cd "$(dirname "$0")"
mkdir -p results_M_finescan

for M in 10 11 14 15; do
    LOG="results_M_finescan/run_M${M}.log"
    echo "================ starting M=$M ================" | tee "$LOG"
    python experiment_p_sweep.py \
        --epochs 1000 \
        --P_list 2 3 4 5 6 \
        --M "$M" \
        --T_gen 5000 \
        --seed 42 \
        --share_B --freeze_B \
        --tag "M${M}" \
        --out_dir results_M_finescan \
        2>&1 | tee -a "$LOG"
done
echo "M-finescan complete."
