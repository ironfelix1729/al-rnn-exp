# -*- coding: utf-8 -*-
"""High-resolution generated-trajectory comparison for the multi-seed sweep.

For each (M, P) cell we pick the median-PSE seed (so the picture represents
the typical, not the best/worst, run) and render a page with:
    * 3-D views of truth, vanilla, ortho   (top row)
    * x-z 2-D projections of the same      (bottom row)

A leading "showcase" page shows the cells where ortho wins biggest on
state-space Hellinger D_H_state.
"""
import os, json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_seedrun")
M_LIST = [10, 11, 14, 15]
P_LIST = [2, 3, 4, 5, 6]
SEEDS = [0, 1, 2]


def median_seed(metrics, M, P):
    """Return the seed whose PSE value is the median of the three."""
    cell = metrics["per_M"][str(M)]["per_P"][str(P)]
    pse = cell["summary"]["ortho"]["PSE"]["values"]  # list of 3
    order = sorted(range(len(pse)), key=lambda i: pse[i])
    median_idx = order[len(order) // 2]
    return SEEDS[median_idx]


def stats_for(metrics, M, P, key, fam):
    s = metrics["per_M"][str(M)]["per_P"][str(P)]["summary"][fam][key]
    return s["mean"], s["std"]


def draw_3d(ax, orb, color, title, lw=0.85):
    orb = np.asarray(orb)
    ax.plot(orb[:, 0], orb[:, 1], orb[:, 2], color=color, lw=lw, alpha=0.95)
    ax.set_title(title, fontsize=11)
    ax.view_init(elev=25, azim=-60)
    rng = float(max(np.ptp(orb[:, 0]), np.ptp(orb[:, 1]), np.ptp(orb[:, 2])))
    cx, cy, cz = orb[:, 0].mean(), orb[:, 1].mean(), orb[:, 2].mean()
    ax.set_xlim(cx - rng / 2 - .3, cx + rng / 2 + .3)
    ax.set_ylim(cy - rng / 2 - .3, cy + rng / 2 + .3)
    ax.set_zlim(cz - rng / 2 - .3, cz + rng / 2 + .3)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])


def draw_2d_xz(ax, orb, color, title, lw=0.55):
    orb = np.asarray(orb)
    ax.plot(orb[:, 0], orb[:, 2], color=color, lw=lw, alpha=0.95)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("x", fontsize=8); ax.set_ylabel("z", fontsize=8)
    ax.set_aspect("equal", adjustable="datalim")
    ax.tick_params(labelsize=7)


def page_for_cell(pdf, metrics, M, P, num_steps=5000):
    seed = median_seed(metrics, M, P)
    arr_path = os.path.join(OUT, f"P{P}_M{M}_s{seed}_arrays.npz")
    if not os.path.exists(arr_path):
        return False
    d = np.load(arr_path)
    truth = d["X_test_trans"][:num_steps]
    orb_v = d["orbit_orig"][:num_steps]
    orb_o = d["orbit_ortho"][:num_steps]

    fig = plt.figure(figsize=(17, 11))
    pse_v_m, pse_v_s = stats_for(metrics, M, P, "PSE", "vanilla")
    pse_o_m, pse_o_s = stats_for(metrics, M, P, "PSE", "ortho")
    dh_v_m, dh_v_s = stats_for(metrics, M, P, "D_H_state", "vanilla")
    dh_o_m, dh_o_s = stats_for(metrics, M, P, "D_H_state", "ortho")
    rel_pse = (pse_o_m - pse_v_m) / pse_v_m * 100 if pse_v_m else 0.0
    rel_dh  = (dh_o_m  - dh_v_m)  / dh_v_m  * 100 if dh_v_m  else 0.0
    fig.suptitle(
        f"M = {M}   P = {P}   (median-PSE seed = {seed} of {SEEDS})\n"
        f"PSE       vanilla {pse_v_m:.4f}±{pse_v_s:.4f}   ortho {pse_o_m:.4f}±{pse_o_s:.4f}   "
        f"mean rel gap {rel_pse:+.1f}%      "
        f"D_H_state vanilla {dh_v_m:.3f}±{dh_v_s:.3f}   ortho {dh_o_m:.3f}±{dh_o_s:.3f}   "
        f"mean rel gap {rel_dh:+.1f}%",
        fontsize=11, fontweight="bold")

    ax1 = fig.add_subplot(2, 3, 1, projection="3d")
    draw_3d(ax1, truth, "black", "Ground truth (rotated)")
    ax2 = fig.add_subplot(2, 3, 2, projection="3d")
    draw_3d(ax2, orb_v, "#c0392b", f"Vanilla AL-RNN  (seed {seed})")
    ax3 = fig.add_subplot(2, 3, 3, projection="3d")
    draw_3d(ax3, orb_o, "#2c5fb0", f"Orthogonal AL-RNN  (seed {seed})")

    ax4 = fig.add_subplot(2, 3, 4)
    draw_2d_xz(ax4, truth, "black", "Ground truth   x-z")
    ax5 = fig.add_subplot(2, 3, 5)
    draw_2d_xz(ax5, orb_v, "#c0392b", f"Vanilla   x-z  (seed {seed})")
    ax6 = fig.add_subplot(2, 3, 6)
    draw_2d_xz(ax6, orb_o, "#2c5fb0", f"Orthogonal   x-z  (seed {seed})")

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    pdf.savefig(fig, dpi=220); plt.close(fig)
    return True


def showcase_page(pdf, metrics, num_steps=5000):
    """Top 4 cells where ortho wins biggest on D_H_state (mean), as a single page."""
    rows = []
    for M in M_LIST:
        for P in P_LIST:
            cell = metrics["per_M"][str(M)]["per_P"][str(P)]["summary"]
            v = cell["vanilla"]["D_H_state"]["mean"]
            o = cell["ortho"]["D_H_state"]["mean"]
            rel = (o - v) / v * 100 if v else 0.0
            rows.append((rel, M, P))
    rows.sort()
    top4 = rows[:4]

    fig = plt.figure(figsize=(17, 11))
    fig.suptitle("Top 4 ortho wins on D_H_state (state-space Hellinger) - median-PSE seed",
                 fontsize=14, fontweight="bold")
    for r, (rel, M, P) in enumerate(top4):
        seed = median_seed(metrics, M, P)
        arr_path = os.path.join(OUT, f"P{P}_M{M}_s{seed}_arrays.npz")
        if not os.path.exists(arr_path):
            continue
        d = np.load(arr_path)
        truth = d["X_test_trans"][:num_steps]
        orb_v = d["orbit_orig"][:num_steps]
        orb_o = d["orbit_ortho"][:num_steps]
        for c, (orb, color, name) in enumerate([
                (truth, "black",   "truth"),
                (orb_v, "#c0392b", "vanilla"),
                (orb_o, "#2c5fb0", "ortho")]):
            ax = fig.add_subplot(4, 3, r * 3 + c + 1)
            draw_2d_xz(ax, orb, color,
                       f"M={M} P={P} {name}  (D_H gap {rel:+.0f}%)" if c == 0
                       else f"M={M} P={P} {name}", lw=0.55)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    pdf.savefig(fig, dpi=220); plt.close(fig)


def main():
    metrics = json.load(open(os.path.join(OUT, "metrics_seedrun.json")))
    pdf_path = os.path.join(OUT, "seedrun_highres_trajectories.pdf")
    with PdfPages(pdf_path) as pdf:
        # cover
        fig = plt.figure(figsize=(8.5, 11))
        ax = fig.add_axes([0.07, 0.04, 0.86, 0.92]); ax.axis("off")
        ax.text(0.0, 0.97,
                "Multi-seed M-finescan: high-resolution trajectories",
                fontsize=15, fontweight="bold", transform=ax.transAxes, va="top")
        for i, ln in enumerate([
            "",
            "For each (M, P) cell we pick the seed whose ORTHO PSE is the median",
            "of the three seeds, so the figure represents the typical run rather",
            "than the best or worst.  Each page: 3-D views (top) and x-z 2-D",
            "projections (bottom) for ground truth / vanilla / orthogonal.",
            "",
            "Header gives PSE and D_H_state mean +/- std over the 3 seeds and",
            "the relative ortho-vs-vanilla gap on the mean.",
        ]):
            ax.text(0.0, 0.92 - 0.03 * i, ln, fontsize=11, family="monospace",
                    transform=ax.transAxes, va="top")
        pdf.savefig(fig); plt.close(fig)
        showcase_page(pdf, metrics)
        for M in M_LIST:
            for P in P_LIST:
                page_for_cell(pdf, metrics, M, P)
    print(f"wrote {pdf_path} ({os.path.getsize(pdf_path) / 1024 / 1024:.1f} MB)")


if __name__ == "__main__":
    main()
