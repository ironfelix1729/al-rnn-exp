# -*- coding: utf-8 -*-
"""Generate a high-resolution PDF of regenerated trajectories (M-sweep) and
matched-pair comparisons (ortho at small M vs vanilla at larger M).

One big PDF with each page a large, crisp figure.  Uses a vector-friendly
backend so lines stay sharp at any zoom.
"""
import os, json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_p_sweep")

M_LIST = [4, 5, 10]
P_LIST = [2, 3, 4]
LABELS = ["x", "y", "z"]
PAIRS = [(0, 1), (0, 2), (1, 2)]


# ---------------------------------------------------------------------------
def load_arrays(M, P, suffix=""):
    path = os.path.join(OUT, f"P{P}_M{M}{suffix}_arrays.npz")
    if not os.path.exists(path):
        return None
    return np.load(path)


def _params_vanilla(M): return M * M + 2 * M
def _params_ortho(M):   return M * M + 3 * M + M * (M - 1) // 2


# ---------------------------------------------------------------------------
def draw_3d(ax, orb, color, title, elev=25, azim=-60, lw=0.9):
    orb = np.asarray(orb)
    ax.plot(orb[:, 0], orb[:, 1], orb[:, 2], color=color, lw=lw, alpha=0.95)
    ax.set_title(title, fontsize=11)
    ax.view_init(elev=elev, azim=azim)
    rng = float(max(np.ptp(orb[:, 0]), np.ptp(orb[:, 1]), np.ptp(orb[:, 2])))
    cx, cy, cz = orb[:, 0].mean(), orb[:, 1].mean(), orb[:, 2].mean()
    ax.set_xlim(cx - rng / 2 - 0.3, cx + rng / 2 + 0.3)
    ax.set_ylim(cy - rng / 2 - 0.3, cy + rng / 2 + 0.3)
    ax.set_zlim(cz - rng / 2 - 0.3, cz + rng / 2 + 0.3)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])


def draw_2d(ax, orb, color, title, ij, lw=0.6):
    i, j = ij
    ax.plot(orb[:, i], orb[:, j], color=color, lw=lw, alpha=0.92)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel(LABELS[i], fontsize=9); ax.set_ylabel(LABELS[j], fontsize=9)
    ax.set_aspect("equal", adjustable="datalim")
    ax.tick_params(labelsize=7)


# ---------------------------------------------------------------------------
def trajectory_pages(pdf, suffix, label_alpha, num_steps=5000):
    """One page per (M, P): 3D on top row, 2D projections on bottom row."""
    for M in M_LIST:
        for P in P_LIST:
            arr = load_arrays(M, P, suffix)
            if arr is None:
                continue
            truth = arr["X_test_trans"][:num_steps]
            orb_v = arr["orbit_orig"][:num_steps]
            orb_o = arr["orbit_ortho"][:num_steps]

            fig = plt.figure(figsize=(17, 11))
            fig.suptitle(
                f"Free-run Lorenz reconstruction   ({label_alpha}, M = {M}, P = {P})",
                fontsize=15, fontweight="bold")

            # 3D row
            ax = fig.add_subplot(2, 3, 1, projection="3d")
            draw_3d(ax, truth, "black", "Ground truth (rotated)")
            ax = fig.add_subplot(2, 3, 2, projection="3d")
            draw_3d(ax, orb_v, "#c0392b", f"Vanilla AL-RNN  ({_params_vanilla(M)} params)")
            ax = fig.add_subplot(2, 3, 3, projection="3d")
            draw_3d(ax, orb_o, "#2c5fb0", f"Orthogonal AL-RNN  ({_params_ortho(M)} params)")

            # 2D row - show x-z projection only (most informative) across the 3
            ij = (0, 2)
            for col, (orb, color, name) in enumerate([
                    (truth, "black",    "Ground truth"),
                    (orb_v, "#c0392b",  "Vanilla"),
                    (orb_o, "#2c5fb0",  "Orthogonal")]):
                ax = fig.add_subplot(2, 3, col + 4)
                draw_2d(ax, orb, color, f"{name}  {LABELS[ij[0]]}-{LABELS[ij[1]]} projection", ij, lw=0.55)

            fig.tight_layout(rect=[0, 0, 1, 0.96])
            pdf.savefig(fig, dpi=220)
            plt.close(fig)


def matched_pairs_pages(pdf, num_steps=5000):
    """One page per matched pair: ortho (M_small) vs vanilla (M_big) at ~equal MSE."""
    # Build from both alpha runs
    sources = [("", "alpha = 1.0"), ("_a0p5", "alpha = 0.5")]
    # extra alpha=1.0 source: M=20 shared-B
    extra = os.path.join(OUT, "summary_sharedB.json")
    summaries = {}
    for suffix, tag in sources:
        for M in M_LIST:
            p = os.path.join(OUT, f"summary_M{M}{suffix}.json")
            if os.path.exists(p):
                with open(p) as f:
                    summaries[(tag, M, suffix)] = json.load(f)
    if os.path.exists(extra):
        with open(extra) as f:
            summaries[("alpha = 1.0", 20, "")] = json.load(f)

    # Harvest candidate pairs: for each alpha, for each ortho(M_small, P), find
    # vanilla(M_big, P, same alpha) with |delta MSE| / MSE <= 0.2 and strictly larger M.
    rows = []
    for (tag, M, suffix), s in summaries.items():
        for P in P_LIST:
            if str(P) not in s["per_P"]:
                continue
            pp = s["per_P"][str(P)]
            rows.append({"tag": tag, "M": M, "P": P, "suffix": suffix,
                         "family": "vanilla", "mse": pp["final_loss_orig"],
                         "params": _params_vanilla(M)})
            rows.append({"tag": tag, "M": M, "P": P, "suffix": suffix,
                         "family": "ortho",   "mse": pp["final_loss_ortho"],
                         "params": _params_ortho(M)})
    matches = []
    for r in rows:
        if r["family"] != "ortho":
            continue
        cands = [v for v in rows
                 if v["family"] == "vanilla" and v["tag"] == r["tag"]
                 and v["M"] > r["M"]
                 and v["P"] == r["P"]
                 and abs(v["mse"] - r["mse"]) / r["mse"] <= 0.20]
        if not cands:
            continue
        best = min(cands, key=lambda v: v["params"])
        matches.append({"ortho": r, "vanilla": best})

    # Keep just one pair per (tag, P) choosing the biggest param-ratio win
    best_per = {}
    for m in matches:
        key = (m["ortho"]["tag"], m["ortho"]["P"])
        ratio = m["vanilla"]["params"] / m["ortho"]["params"]
        if key not in best_per or ratio > best_per[key][0]:
            best_per[key] = (ratio, m)
    matches = [t[1] for t in best_per.values()]

    for m in sorted(matches, key=lambda x: (x["ortho"]["tag"], x["ortho"]["P"])):
        o = m["ortho"]; v = m["vanilla"]
        # need trajectory arrays
        arr_o = load_arrays(o["M"], o["P"], o["suffix"])
        # vanilla's arrays may come from the sharedB (M=20) run where files are named bare
        if v["M"] == 20 and v["suffix"] == "":
            # M=20 shared-B did not tag its P*.npz - those files were overwritten.
            # So we can only show this match if an alternate source exists.  Skip if missing.
            continue
        arr_v = load_arrays(v["M"], v["P"], v["suffix"])
        if arr_o is None or arr_v is None:
            continue

        truth = arr_o["X_test_trans"][:num_steps]
        orb_o = arr_o["orbit_ortho"][:num_steps]
        orb_v = arr_v["orbit_orig"][:num_steps]

        fig = plt.figure(figsize=(17, 11))
        ratio = v["params"] / o["params"]
        fig.suptitle(
            f"Matched pair ({o['tag']}, P = {o['P']}):  "
            f"ortho M={o['M']} ({o['params']} params, MSE {o['mse']:.4f})  "
            f"≈  vanilla M={v['M']} ({v['params']} params, MSE {v['mse']:.4f})  "
            f"[vanilla uses {ratio:.2f}× params]",
            fontsize=13, fontweight="bold")

        ax = fig.add_subplot(2, 3, 1, projection="3d")
        draw_3d(ax, truth, "black", "Ground truth (rotated)")
        ax = fig.add_subplot(2, 3, 2, projection="3d")
        draw_3d(ax, orb_v, "#c0392b", f"Vanilla  M={v['M']}  P={v['P']}")
        ax = fig.add_subplot(2, 3, 3, projection="3d")
        draw_3d(ax, orb_o, "#2c5fb0", f"Orthogonal  M={o['M']}  P={o['P']}")

        ij = (0, 2)
        for col, (orb, color, name) in enumerate([
                (truth, "black",   "Ground truth"),
                (orb_v, "#c0392b", f"Vanilla  M={v['M']}"),
                (orb_o, "#2c5fb0", f"Orthogonal  M={o['M']}")]):
            ax = fig.add_subplot(2, 3, col + 4)
            draw_2d(ax, orb, color, f"{name}  {LABELS[ij[0]]}-{LABELS[ij[1]]}", ij, lw=0.55)

        fig.tight_layout(rect=[0, 0, 1, 0.96])
        pdf.savefig(fig, dpi=220)
        plt.close(fig)


# ---------------------------------------------------------------------------
def main():
    pdf_path = os.path.join(OUT, "highres_visualisations.pdf")
    with PdfPages(pdf_path) as pdf:
        # section 1: regenerated trajectories at alpha=1.0
        trajectory_pages(pdf, "",     "alpha = 1.0")
        # section 2: regenerated trajectories at alpha=0.5
        trajectory_pages(pdf, "_a0p5", "alpha = 0.5")
        # section 3: matched-pair comparisons
        matched_pairs_pages(pdf)
    print(f"wrote {pdf_path} ({os.path.getsize(pdf_path)/1024/1024:.1f} MB)")


if __name__ == "__main__":
    main()
