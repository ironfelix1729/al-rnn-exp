#!/usr/bin/env bash
# seedrun_v2: M in {3, 4, 5, 8, 9} x P <= M x 3 seeds.
# Fixed 20-d master B (Option A, sliced as B[:, :M]); ortho without ReLU bias.
# 1000 epochs, alpha=1.0, shared frozen B per M.
# Resume-aware at the (M, seed) level.

set -e
cd "$(dirname "$0")"
mkdir -p results_seedrun_v2

EPOCHS=1000
SEEDS="0 1 2"

P_for_M() {
    case "$1" in
        3) echo "2 3" ;;
        4) echo "2 3 4" ;;
        5) echo "2 3 4 5" ;;
        8) echo "2 3 4 5 6" ;;
        9) echo "2 3 4 5 6" ;;
        *) echo "" ;;
    esac
}

for SEED in $SEEDS; do
    for M in 3 4 5 8 9; do
        TAG="M${M}_s${SEED}"
        SUMMARY="results_seedrun_v2/summary_${TAG}.json"
        if [ -f "$SUMMARY" ]; then
            EXPECTED=$(P_for_M $M | tr ' ' ',')
            DONE=$(python -c "import json; s=json.load(open('$SUMMARY')); print(','.join(sorted(s['per_P'].keys())))" 2>/dev/null || echo "")
            if [ "$DONE" = "$EXPECTED" ]; then
                echo "[skip] $TAG already complete"
                continue
            fi
        fi
        P_LIST=$(P_for_M $M)
        LOG="results_seedrun_v2/run_${TAG}.log"
        echo "================ M=$M seed=$SEED  P=$P_LIST ================" | tee -a "$LOG"
        python experiment_p_sweep.py \
            --epochs "$EPOCHS" \
            --P_list $P_LIST \
            --M "$M" \
            --T_gen 5000 \
            --seed "$SEED" \
            --seed_data 42 \
            --seed_B 42 \
            --share_B --freeze_B \
            --no_bias_Q \
            --B_full_dim 20 \
            --tag "$TAG" \
            --out_dir results_seedrun_v2 \
            2>&1 | tee -a "$LOG"
    done
done
echo "All seedrun_v2 cells complete."
