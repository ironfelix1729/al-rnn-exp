# -*- coding: utf-8 -*-
"""Compute Dstsp + PSE + RMSE for every (M, P, family) cell of the
M-finescan sweep.  Reads results_M_finescan/P{P}_M{M}_arrays.npz and writes
results_M_finescan/metrics_all.json.
"""
import os, json, math, sys
import numpy as np
import torch as tc

from metrics import state_space_divergence_binning, power_spectrum_error

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_M_finescan")
M_LIST = [10, 11, 14, 15]
P_LIST = [2, 3, 4, 5, 6]


def per_cell(orbit, truth):
    """Return dict with Dstsp, PSE, free-run RMSE on a 100-step window."""
    n = min(len(orbit), len(truth), 5000)
    orb = np.asarray(orbit)[:n]
    tru = np.asarray(truth)[:n]
    # observation coords are the first 3 entries of the hidden state
    orb3 = orb[:, :3]
    tru3 = tru[:, :3]
    dstsp = state_space_divergence_binning(
        tc.tensor(orb3, dtype=tc.float32),
        tc.tensor(tru3, dtype=tc.float32),
        n_bins=30,
    )
    pse = power_spectrum_error(tru3, orb3, smoothing=20)
    # short-horizon free-run RMSE: how fast does the orbit diverge from truth?
    h = min(100, n)
    rmse_h = float(np.sqrt(np.mean((orb3[:h] - tru3[:h]) ** 2)))
    return {"Dstsp": float(dstsp), "PSE": float(pse), "rmse_h100": rmse_h}


def main():
    out = {}
    missing = []
    for M in M_LIST:
        s_path = os.path.join(OUT, f"summary_M{M}.json")
        summary = json.load(open(s_path)) if os.path.exists(s_path) else {"per_P": {}}
        out[str(M)] = {"per_P": {}}
        for P in P_LIST:
            arr_path = os.path.join(OUT, f"P{P}_M{M}_arrays.npz")
            if not os.path.exists(arr_path):
                missing.append((M, P))
                continue
            d = np.load(arr_path)
            truth = d["X_test_trans"]
            cell = {
                "vanilla": per_cell(d["orbit_orig"], truth),
                "ortho":   per_cell(d["orbit_ortho"], truth),
            }
            # add training-final MSE / RMSE (one-step) from summary if present
            if str(P) in summary["per_P"]:
                pp = summary["per_P"][str(P)]
                cell["vanilla"]["mse_train_final"] = pp["final_loss_orig"]
                cell["vanilla"]["rmse_train_final"] = math.sqrt(pp["final_loss_orig"])
                cell["ortho"]["mse_train_final"] = pp["final_loss_ortho"]
                cell["ortho"]["rmse_train_final"] = math.sqrt(pp["final_loss_ortho"])
                cell["Q_analysis"] = pp.get("Q_analysis", {})
            out[str(M)]["per_P"][str(P)] = cell
    with open(os.path.join(OUT, "metrics_all.json"), "w") as f:
        json.dump(out, f, indent=2)
    if missing:
        print(f"missing: {missing}")
    # also dump a flat text table
    lines = [
        f"{'M':>3} {'P':>3}  {'family':>8}  {'rmse_train':>10}  {'rmse_h100':>9}  "
        f"{'Dstsp':>8}  {'PSE':>8}",
        "-" * 64,
    ]
    for M in M_LIST:
        for P in P_LIST:
            cell = out[str(M)].get("per_P", {}).get(str(P))
            if not cell:
                continue
            for fam in ("vanilla", "ortho"):
                c = cell[fam]
                lines.append(
                    f"{M:>3} {P:>3}  {fam:>8}  "
                    f"{c.get('rmse_train_final', float('nan')):>10.5f}  "
                    f"{c['rmse_h100']:>9.4f}  {c['Dstsp']:>8.3f}  {c['PSE']:>8.4f}"
                )
            lines.append("")
    with open(os.path.join(OUT, "metrics_all.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
