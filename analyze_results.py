# -*- coding: utf-8 -*-
"""
Analysis: Ortho vs Vanilla AL-RNN — does ortho require smaller M?
Primary metric: PSE (Hellinger distance of power spectrum on test run).

Data sources (from claude/understand-rnn-models-rNbtB):
  results_p_sweep/Msweep_report.txt      — M in {4,5,10}, P in {2,3,4}, alpha=1.0
  results_p_sweep/Msweep_a0p5_report.txt — same grid, alpha=0.5
  results_M_finescan/metrics_all.json    — M in {10,11,14,15}, P in {2,3,4,5,6}
  results_M_finescan/matched_pairs.txt   — cross-M matched pairs within 20%
"""

import json
import numpy as np

# ── 1. Fine M-scan data ───────────────────────────────────────────────────

with open("results_M_finescan/metrics_all.json") as f:
    fine = json.load(f)

flat = {}
for M_str, M_data in fine.items():
    M = int(M_str)
    for P_str, P_data in M_data["per_P"].items():
        P = int(P_str)
        for family in ("vanilla", "ortho"):
            flat[(M, P, family)] = P_data[family]

M_values_fine = [10, 11, 14, 15]
P_values_fine = [2, 3, 4, 5, 6]

# ── 2. Primary metric: PSE per (M, P) ────────────────────────────────────

print("=" * 72)
print("SECTION 1: PSE (Hellinger, power spectrum) — ortho vs vanilla")
print("  Primary metric. Lower = better spectral match to true attractor.")
print("=" * 72)

print(f"\n  {'M':>4}  {'P':>3}  {'vanilla PSE':>13}  {'ortho PSE':>11}  "
      f"{'ortho gain%':>12}  {'winner':>8}")
print("  " + "-" * 60)

pse_gains_by_P = {P: [] for P in P_values_fine}
pse_gains_by_M = {M: [] for M in M_values_fine}

for M in M_values_fine:
    for P in P_values_fine:
        v = flat.get((M, P, "vanilla"))
        o = flat.get((M, P, "ortho"))
        if v is None or o is None:
            continue
        vp, op = v["PSE"], o["PSE"]
        gain = 100 * (vp - op) / vp
        winner = "ortho" if op < vp else "VANILLA"
        pse_gains_by_P[P].append(gain)
        pse_gains_by_M[M].append(gain)
        print(f"  {M:>4}  {P:>3}  {vp:>13.4f}  {op:>11.4f}  {gain:>+12.1f}%  {winner:>8}")
    print()

# ── 3. PSE wins tally ─────────────────────────────────────────────────────

print("=" * 72)
print("SECTION 2: PSE win rate by P and by M")
print("=" * 72)

print("\n  By P (collapsing over M):")
for P in P_values_fine:
    gains = pse_gains_by_P[P]
    wins = sum(g > 0 for g in gains)
    print(f"    P={P}: ortho wins {wins}/{len(gains)}  "
          f"mean gain={np.mean(gains):+.1f}%  "
          f"(range {min(gains):.1f}% – {max(gains):.1f}%)")

print("\n  By M (collapsing over P):")
for M in M_values_fine:
    gains = pse_gains_by_M[M]
    wins = sum(g > 0 for g in gains)
    print(f"    M={M:2d}: ortho wins {wins}/{len(gains)}  "
          f"mean gain={np.mean(gains):+.1f}%  "
          f"(range {min(gains):.1f}% – {max(gains):.1f}%)")

# ── 4. Cross-M matching on PSE ────────────────────────────────────────────

print()
print("=" * 72)
print("SECTION 3: Cross-M hypothesis test on PSE")
print("  For each P: find smallest ortho M whose PSE ≤ vanilla at a larger M.")
print("=" * 72)

print()
for P in P_values_fine:
    print(f"  P={P}:")
    for M_van in M_values_fine:
        v = flat.get((M_van, P, "vanilla"))
        if v is None:
            continue
        v_pse = v["PSE"]
        for M_ort in M_values_fine:
            if M_ort >= M_van:
                continue
            o = flat.get((M_ort, P, "ortho"))
            if o is None:
                continue
            o_pse = o["PSE"]
            if o_pse <= v_pse:
                print(f"    ortho(M={M_ort}) PSE={o_pse:.4f}  ≤  vanilla(M={M_van}) PSE={v_pse:.4f}  ✓")
    print()

# ── 5. PSE vs training RMSE — how often they disagree ────────────────────

print("=" * 72)
print("SECTION 4: PSE vs training RMSE — agreement check")
print("  Cases where PSE and training RMSE give opposite verdicts.")
print("=" * 72)

disagreements = 0
total = 0
print()
for M in M_values_fine:
    for P in P_values_fine:
        v = flat.get((M, P, "vanilla"))
        o = flat.get((M, P, "ortho"))
        if v is None or o is None:
            continue
        total += 1
        ortho_wins_rmse = o["rmse_train_final"] < v["rmse_train_final"]
        ortho_wins_pse  = o["PSE"] < v["PSE"]
        if ortho_wins_rmse != ortho_wins_pse:
            disagreements += 1
            rmse_winner = "ortho" if ortho_wins_rmse else "vanilla"
            pse_winner  = "ortho" if ortho_wins_pse  else "vanilla"
            print(f"  M={M:2d} P={P}: RMSE→{rmse_winner:7s}  PSE→{pse_winner:7s}  "
                  f"(RMSE gain={100*(v['rmse_train_final']-o['rmse_train_final'])/v['rmse_train_final']:+.1f}%  "
                  f"PSE gain={100*(v['PSE']-o['PSE'])/v['PSE']:+.1f}%)")

print(f"\n  Disagreements: {disagreements}/{total} configs ({100*disagreements/total:.0f}%)")

# ── 6. Summary ────────────────────────────────────────────────────────────

print()
print("=" * 72)
print("SUMMARY & HYPOTHESIS VERDICT  (primary metric: PSE)")
print("=" * 72)

all_pse_gains = [g for gs in pse_gains_by_P.values() for g in gs]
total_wins = sum(g > 0 for g in all_pse_gains)

print(f"""
Hypothesis: Ortho requires smaller M to achieve comparable performance
            as vanilla (measured by PSE — Hellinger dist. of power spectrum).

Overall: ortho wins {total_wins}/20 configs ({100*total_wins/20:.0f}%) on PSE.

Verdict: PARTIALLY SUPPORTED — the advantage is real but strongly
         P-dependent, unlike the cleaner picture seen with training RMSE.

Key findings by P
─────────────────
  P=2: ortho wins 2/4 (50%); mean gain {np.mean(pse_gains_by_P[2]):+.1f}%.
       Results are polarised: ortho wins big at small M (M=10: +78%)
       but collapses at large M (M=15: −340%, ortho PSE=0.336 vs vanilla 0.076).
       → Ortho's orthogonal transform fails to capture the correct power
         spectrum at large M/small P; vanilla finds a better spectral solution.

  P=3: ortho wins 2/4 (50%); mean gain {np.mean(pse_gains_by_P[3]):+.1f}%.
       Inconsistent. Some large losses (M=14: −177%, M=15: −39%).
       → Not recommended to trust the hypothesis at P=3.

  P=4: ortho wins 4/4 (100%); mean gain {np.mean(pse_gains_by_P[4]):+.1f}%.
       Most reliable P setting. Ortho consistently achieves lower PSE.
       Cross-M: ortho(M=10) ≤ vanilla(M=11/14/15) at P=4.

  P=5: ortho wins 3/4 (75%); mean gain {np.mean(pse_gains_by_P[5]):+.1f}%.
       One loss at M=10 (ortho PSE=0.089 vs vanilla 0.050, −79%).
       Otherwise ortho wins. Cross-M matches hold at M≥11.

  P=6: ortho wins 4/4 (100%); mean gain {np.mean(pse_gains_by_P[6]):+.1f}%.
       Reliable. Ortho consistently better on PSE.

Smaller-M equivalences on PSE (ortho M < vanilla M, PSE within tolerance)
──────────────────────────────────────────────────────────────────────────
  Reliable (P=4,6):  ortho(M=10) ≈ vanilla(M=11–15)
  Partial  (P=5):    ortho(M=11) ≈ vanilla(M=14–15)
  Unreliable (P=2,3): no consistent cross-M pattern

Qualification of the hypothesis
────────────────────────────────
  The hypothesis holds cleanly for P ≥ 4. At P=2 and P=3 (very few
  nonlinear units), the ortho model's learned isometry can end up
  rotating the hidden state in a way that distorts the spectral content
  of the generated trajectory, even when training loss is lower. This
  suggests training RMSE and PSE can decouple at low P, and PSE should
  be monitored directly during model selection.

  Revised hypothesis: "Ortho requires smaller M to achieve comparable
  PSE performance as vanilla, provided P ≥ 4 (sufficient nonlinear units
  relative to M)."
""")
