#!/usr/bin/env bash
# Resume alpha=0.5 sweep: M=5 P=4 only, then full M=10.
set -e
cd "$(dirname "$0")"

# 1. finish M=5 with just P=4 (P=2,3 already in summary_M5_a0p5.json)
LOG="results_p_sweep/run_M5_a0p5_resume.log"
echo "================ resuming M=5 (P=4 only, alpha=0.5) ================" | tee "$LOG"
python experiment_p_sweep.py \
    --epochs 1500 --P_list 4 --M 5 \
    --T_gen 5000 --seed 42 \
    --share_B --freeze_B \
    --alpha 0.5 \
    --tag "M5_a0p5" \
    2>&1 | tee -a "$LOG"

# 2. run the full M=10 alpha=0.5 sweep
LOG="results_p_sweep/run_M10_a0p5.log"
echo "================ starting M=10 (alpha=0.5) ================" | tee "$LOG"
python experiment_p_sweep.py \
    --epochs 1500 --P_list 2 3 4 --M 10 \
    --T_gen 5000 --seed 42 \
    --share_B --freeze_B \
    --alpha 0.5 \
    --tag "M10_a0p5" \
    2>&1 | tee -a "$LOG"

echo "alpha=0.5 resume complete."
