# -*- coding: utf-8 -*-
"""Aggregate seedrun results: mean +- std across seeds for each (M, P, family).

Reads results_seedrun/summary_M{M}_s{seed}.json and the corresponding
P{P}_M{M}_s{seed}_arrays.npz, computes per-cell metrics
(rmse_train, rmse_h100, Dstsp, PSE, D_H_state) for each seed, and
writes results_seedrun/metrics_seedrun.json with per-seed values plus
mean/std summaries.
"""
import os, json, math
import numpy as np
import torch as tc

from metrics import (
    state_space_divergence_binning,
    state_space_hellinger_binning,
    power_spectrum_error,
)

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_seedrun")
M_LIST = [10, 11, 14, 15]
P_LIST = [2, 3, 4, 5, 6]
SEEDS = [0, 1, 2]


def per_cell(orbit, truth):
    n = min(len(orbit), len(truth), 5000)
    orb3 = np.asarray(orbit)[:n, :3]
    tru3 = np.asarray(truth)[:n, :3]
    dstsp = state_space_divergence_binning(
        tc.tensor(orb3, dtype=tc.float32),
        tc.tensor(tru3, dtype=tc.float32),
        n_bins=30,
    )
    d_h = state_space_hellinger_binning(
        tc.tensor(orb3, dtype=tc.float32),
        tc.tensor(tru3, dtype=tc.float32),
        n_bins=30,
    )
    pse = power_spectrum_error(tru3, orb3, smoothing=20)
    h = min(100, n)
    rmse_h = float(np.sqrt(np.mean((orb3[:h] - tru3[:h]) ** 2)))
    return {"Dstsp": float(dstsp), "PSE": float(pse), "D_H_state": float(d_h),
            "rmse_h100": rmse_h}


def main():
    out = {"per_M": {}}
    seed_table = {}
    for M in M_LIST:
        out["per_M"][str(M)] = {"per_P": {}}
        for P in P_LIST:
            cell = {"vanilla": {}, "ortho": {}}
            for seed in SEEDS:
                tag = f"M{M}_s{seed}"
                arr_path = os.path.join(OUT, f"P{P}_{tag}_arrays.npz")
                summary_path = os.path.join(OUT, f"summary_{tag}.json")
                if not (os.path.exists(arr_path) and os.path.exists(summary_path)):
                    continue
                d = np.load(arr_path)
                truth = d["X_test_trans"]
                with open(summary_path) as f:
                    s = json.load(f)
                if str(P) not in s["per_P"]:
                    continue
                pp = s["per_P"][str(P)]
                m_v = per_cell(d["orbit_orig"], truth)
                m_o = per_cell(d["orbit_ortho"], truth)
                m_v["mse_train_final"] = pp["final_loss_orig"]
                m_v["rmse_train_final"] = math.sqrt(pp["final_loss_orig"])
                m_o["mse_train_final"] = pp["final_loss_ortho"]
                m_o["rmse_train_final"] = math.sqrt(pp["final_loss_ortho"])
                cell["vanilla"][str(seed)] = m_v
                cell["ortho"][str(seed)] = m_o
            # mean / std per metric
            cell_summary = {"vanilla": {}, "ortho": {}}
            for fam in ("vanilla", "ortho"):
                runs = list(cell[fam].values())
                if not runs:
                    continue
                for key in ("rmse_train_final", "rmse_h100",
                            "Dstsp", "PSE", "D_H_state"):
                    vals = np.array([r.get(key, np.nan) for r in runs])
                    cell_summary[fam][key] = {
                        "mean": float(np.nanmean(vals)),
                        "std": float(np.nanstd(vals, ddof=1) if len(vals) > 1 else 0.0),
                        "n": int(np.sum(~np.isnan(vals))),
                        "values": vals.tolist(),
                    }
            out["per_M"][str(M)]["per_P"][str(P)] = {
                "per_seed": cell, "summary": cell_summary,
            }
    with open(os.path.join(OUT, "metrics_seedrun.json"), "w") as f:
        json.dump(out, f, indent=2)

    # text table: mean +- std per metric per cell
    lines = []
    add = lines.append
    for key in ("rmse_train_final", "Dstsp", "PSE", "D_H_state"):
        add(f"\n=== {key}  (mean +- std over seeds) ===")
        add(f"{'M':>3} {'P':>3}  {'vanilla':>20}  {'ortho':>20}  {'gap_mean%':>9}")
        add("-" * 72)
        for M in M_LIST:
            for P in P_LIST:
                cell = out["per_M"].get(str(M), {}).get("per_P", {}).get(str(P))
                if not cell or "summary" not in cell:
                    continue
                v = cell["summary"].get("vanilla", {}).get(key)
                o = cell["summary"].get("ortho", {}).get(key)
                if v is None or o is None:
                    continue
                gap = (o["mean"] - v["mean"]) / v["mean"] * 100 if v["mean"] != 0 else 0.0
                add(f"{M:>3} {P:>3}  {v['mean']:>10.5f} +- {v['std']:>5.4f}  "
                    f"{o['mean']:>10.5f} +- {o['std']:>5.4f}  {gap:>+8.1f}%")
            add("")
    txt = "\n".join(lines)
    with open(os.path.join(OUT, "metrics_seedrun.txt"), "w") as f:
        f.write(txt)
    print(txt[:3000])


if __name__ == "__main__":
    main()
