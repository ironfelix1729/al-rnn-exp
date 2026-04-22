#!/usr/bin/env bash
# Chain three M-sweeps (M=4,5,10) back-to-back.  Each run uses its own
# shared, frozen B (shape (3, M) differs per M) and sweeps P=2,3,4.
set -e
cd "$(dirname "$0")"
mkdir -p results_p_sweep

for M in 4 5 10; do
    LOG="results_p_sweep/run_M${M}.log"
    echo "================ starting M=$M ================" | tee -a "$LOG"
    python experiment_p_sweep.py \
        --epochs 1500 \
        --P_list 2 3 4 \
        --M "$M" \
        --T_gen 5000 \
        --seed 42 \
        --share_B --freeze_B \
        --tag "M${M}" \
        2>&1 | tee -a "$LOG"
done
echo "All three M-sweeps complete."
