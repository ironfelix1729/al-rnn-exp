# -*- coding: utf-8 -*-
"""Higher-resolution trajectory comparison images for the multi-seed sweep.

Outputs (under results_seedrun/trajectories_hires/):

  cell_M{M}_P{P}.png        - one large image per (M, P) cell, all 3 seeds shown.
                              3 rows (truth / vanilla / ortho) x 6 cols
                              (3 seeds 3-D + 3 seeds x-z 2-D).
                              ~24 x 14 inches at 200 DPI.

  wall_M{M}.png             - all 5 P values for one M, with the median-PSE
                              seed only, in a single wall-sized image.

  wall_all.png              - 4 (M) x 5 (P) wall, median-PSE seed, x-z only.

These are easier to inspect than a PDF: GitHub renders PNGs at native
resolution and the lines stay crisp when you zoom in.
"""
import os, json
import numpy as np
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_seedrun")
DEST = os.path.join(OUT, "trajectories_hires")
os.makedirs(DEST, exist_ok=True)
M_LIST = [10, 11, 14, 15]
P_LIST = [2, 3, 4, 5, 6]
SEEDS = [0, 1, 2]


def median_seed(metrics, M, P):
    pse = metrics["per_M"][str(M)]["per_P"][str(P)]["summary"]["ortho"]["PSE"]["values"]
    order = sorted(range(len(pse)), key=lambda i: pse[i])
    return SEEDS[order[len(order) // 2]]


def stats_for(metrics, M, P, key, fam):
    s = metrics["per_M"][str(M)]["per_P"][str(P)]["summary"][fam][key]
    return s["mean"], s["std"]


def load_arr(M, P, seed):
    p = os.path.join(OUT, f"P{P}_M{M}_s{seed}_arrays.npz")
    if not os.path.exists(p):
        return None
    return np.load(p)


def draw_3d(ax, orb, color, title, lw=0.8, elev=25, azim=-60):
    orb = np.asarray(orb)
    ax.plot(orb[:, 0], orb[:, 1], orb[:, 2], color=color, lw=lw, alpha=0.95)
    ax.set_title(title, fontsize=10)
    ax.view_init(elev=elev, azim=azim)
    rng = float(max(np.ptp(orb[:, 0]), np.ptp(orb[:, 1]), np.ptp(orb[:, 2])))
    cx, cy, cz = orb[:, 0].mean(), orb[:, 1].mean(), orb[:, 2].mean()
    ax.set_xlim(cx - rng / 2 - 0.3, cx + rng / 2 + 0.3)
    ax.set_ylim(cy - rng / 2 - 0.3, cy + rng / 2 + 0.3)
    ax.set_zlim(cz - rng / 2 - 0.3, cz + rng / 2 + 0.3)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])


def draw_2d(ax, orb, color, title, ij=(0, 2), lw=0.5):
    orb = np.asarray(orb)
    i, j = ij
    ax.plot(orb[:, i], orb[:, j], color=color, lw=lw, alpha=0.95)
    ax.set_title(title, fontsize=9)
    ax.set_xlabel(["x", "y", "z"][i], fontsize=8)
    ax.set_ylabel(["x", "y", "z"][j], fontsize=8)
    ax.set_aspect("equal", adjustable="datalim")
    ax.tick_params(labelsize=6)


def make_cell_image(metrics, M, P, num_steps=5000, dpi=200):
    """One file per (M, P) cell showing all 3 seeds in two banks (3-D and 2-D)."""
    rows, cols = 3, 6
    fig, axes = plt.subplots(rows, cols, figsize=(24, 14),
                             gridspec_kw={"hspace": 0.35, "wspace": 0.25})
    pse_v_m, pse_v_s = stats_for(metrics, M, P, "PSE", "vanilla")
    pse_o_m, pse_o_s = stats_for(metrics, M, P, "PSE", "ortho")
    dh_v_m, dh_v_s   = stats_for(metrics, M, P, "D_H_state", "vanilla")
    dh_o_m, dh_o_s   = stats_for(metrics, M, P, "D_H_state", "ortho")
    rel_pse = (pse_o_m - pse_v_m) / pse_v_m * 100 if pse_v_m else 0.0
    rel_dh  = (dh_o_m  - dh_v_m)  / dh_v_m  * 100 if dh_v_m  else 0.0
    fig.suptitle(
        f"M = {M}    P = {P}    (all 3 seeds; cols 1-3: 3-D, cols 4-6: x-z 2-D)\n"
        f"PSE        vanilla {pse_v_m:.4f} ± {pse_v_s:.4f}    ortho {pse_o_m:.4f} ± {pse_o_s:.4f}    "
        f"mean rel gap {rel_pse:+.1f}%      "
        f"D_H_state vanilla {dh_v_m:.3f} ± {dh_v_s:.3f}    ortho {dh_o_m:.3f} ± {dh_o_s:.3f}    "
        f"mean rel gap {rel_dh:+.1f}%",
        fontsize=14, fontweight="bold")

    # convert axes to 3-D where needed: cols 0-2 are 3-D, cols 3-5 are 2-D
    for ax in axes[:, :3].ravel():
        ax.remove()
    for r in range(3):
        for c in range(3):
            ax3 = fig.add_subplot(rows, cols, r * cols + c + 1, projection="3d")
            axes[r, c] = ax3

    rownames = ["Truth", "Vanilla", "Orthogonal"]
    colors = ["black", "#c0392b", "#2c5fb0"]
    for c, seed in enumerate(SEEDS):
        d = load_arr(M, P, seed)
        if d is None:
            continue
        truth = d["X_test_trans"][:num_steps]
        ov = d["orbit_orig"][:num_steps]
        oo = d["orbit_ortho"][:num_steps]
        triples = [(truth, "black"), (ov, "#c0392b"), (oo, "#2c5fb0")]
        for r, (orb, color) in enumerate(triples):
            draw_3d(axes[r, c], orb, color,
                    f"{rownames[r]}  seed {seed}  3-D")
            draw_2d(axes[r, c + 3], orb, color,
                    f"{rownames[r]}  seed {seed}  x-z")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    out = os.path.join(DEST, f"cell_M{M}_P{P}.png")
    fig.savefig(out, dpi=dpi); plt.close(fig)
    return out


def make_wall_M(metrics, M, num_steps=5000, dpi=180):
    """For one M, all 5 P values with median-PSE seed, x-z projection only."""
    rows, cols = 3, len(P_LIST)
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows),
                             gridspec_kw={"hspace": 0.3, "wspace": 0.18})
    fig.suptitle(f"M = {M}    median-PSE seed per cell    x-z projections",
                 fontsize=14, fontweight="bold")
    rownames = ["Truth", "Vanilla", "Orthogonal"]
    colors = ["black", "#c0392b", "#2c5fb0"]
    for c, P in enumerate(P_LIST):
        seed = median_seed(metrics, M, P)
        d = load_arr(M, P, seed)
        if d is None:
            continue
        triples = [(d["X_test_trans"][:num_steps], "black", "Truth"),
                   (d["orbit_orig"][:num_steps], "#c0392b", "Vanilla"),
                   (d["orbit_ortho"][:num_steps], "#2c5fb0", "Orthogonal")]
        for r, (orb, color, name) in enumerate(triples):
            draw_2d(axes[r, c], orb, color,
                    f"{name}  P={P}  seed {seed}")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = os.path.join(DEST, f"wall_M{M}.png")
    fig.savefig(out, dpi=dpi); plt.close(fig)
    return out


def make_wall_all(metrics, num_steps=5000, dpi=160):
    """Single wall: 12 rows (4 M x 3 sources) x 5 P columns."""
    rows = len(M_LIST) * 3
    cols = len(P_LIST)
    fig, axes = plt.subplots(rows, cols, figsize=(3.4 * cols, 2.6 * rows),
                             gridspec_kw={"hspace": 0.45, "wspace": 0.25})
    fig.suptitle("Trajectory wall: all (M, P) cells, median-PSE seed, x-z projections",
                 fontsize=15, fontweight="bold")
    for mi, M in enumerate(M_LIST):
        for c, P in enumerate(P_LIST):
            seed = median_seed(metrics, M, P)
            d = load_arr(M, P, seed)
            if d is None:
                continue
            triples = [(d["X_test_trans"][:num_steps], "black", "Truth"),
                       (d["orbit_orig"][:num_steps], "#c0392b", "Vanilla"),
                       (d["orbit_ortho"][:num_steps], "#2c5fb0", "Orthogonal")]
            for r, (orb, color, name) in enumerate(triples):
                draw_2d(axes[3 * mi + r, c], orb, color,
                        f"M={M} P={P}  {name}  seed {seed}", lw=0.45)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = os.path.join(DEST, "wall_all.png")
    fig.savefig(out, dpi=dpi); plt.close(fig)
    return out


def main():
    metrics = json.load(open(os.path.join(OUT, "metrics_seedrun.json")))
    paths = []
    for M in M_LIST:
        for P in P_LIST:
            p = make_cell_image(metrics, M, P)
            paths.append(p)
            print(f"wrote {p}  ({os.path.getsize(p)/1024:.0f} KB)")
    for M in M_LIST:
        p = make_wall_M(metrics, M)
        paths.append(p)
        print(f"wrote {p}  ({os.path.getsize(p)/1024:.0f} KB)")
    p = make_wall_all(metrics)
    paths.append(p)
    print(f"wrote {p}  ({os.path.getsize(p)/1024:.0f} KB)")


if __name__ == "__main__":
    main()
