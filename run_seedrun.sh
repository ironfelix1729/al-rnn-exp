#!/usr/bin/env bash
# Multi-seed sweep: M in {10, 11, 14, 15} x P in {2..6}, 3 seeds, 1000 epochs.
# Skips a cell if its summary file already exists (cheap resume after restarts).

set -e
cd "$(dirname "$0")"
mkdir -p results_seedrun

EPOCHS=1000
P_LIST="2 3 4 5 6"
M_LIST="10 11 14 15"
SEEDS="0 1 2"

for SEED in $SEEDS; do
    for M in $M_LIST; do
        TAG="M${M}_s${SEED}"
        SUMMARY="results_seedrun/summary_${TAG}.json"
        # Skip if summary already has all P values for this (M, seed)
        if [ -f "$SUMMARY" ]; then
            DONE=$(python -c "import json,sys; s=json.load(open('$SUMMARY')); print(','.join(sorted(s['per_P'].keys())))" 2>/dev/null || echo "")
            if [ "$DONE" = "2,3,4,5,6" ]; then
                echo "[skip] $TAG already complete"
                continue
            fi
        fi
        LOG="results_seedrun/run_${TAG}.log"
        echo "================ M=$M seed=$SEED ================" | tee -a "$LOG"
        python experiment_p_sweep.py \
            --epochs "$EPOCHS" \
            --P_list $P_LIST \
            --M "$M" \
            --T_gen 5000 \
            --seed "$SEED" \
            --seed_data 42 \
            --seed_B 42 \
            --share_B --freeze_B \
            --tag "$TAG" \
            --out_dir results_seedrun \
            2>&1 | tee -a "$LOG"
    done
done
echo "All seedrun cells complete."
