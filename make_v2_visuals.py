# -*- coding: utf-8 -*-
"""High-res visuals + 3-D (M, P, metric) analysis for seedrun_v2.

Produces under results_seedrun_v2/viz_v2/:

  per_cell/cell_M{M}_P{P}.png         large per-cell trajectory triple
                                       (truth | vanilla | ortho), all 3 seeds
  walls/wall_all.png                  4 M x 5 P wall, median-PSE seed, x-z 2-D
  surface_<metric>.png                3-D surface  M x P x metric for both models
  bars3d_<metric>.png                 3-D bar plot M x P x metric
  lines_M_<metric>.png                metric vs M, one curve per P
  lines_P_<metric>.png                metric vs P, one curve per M
  ridge_<metric>.png                  ridge plot of seed-level distributions
  v2_visuals_report.pdf               stitched PDF
"""
import os, json, math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_seedrun_v2")
DEST = os.path.join(OUT, "viz_v2")
PER_CELL = os.path.join(DEST, "per_cell")
WALLS = os.path.join(DEST, "walls")
SURFACES = os.path.join(DEST, "surfaces")
os.makedirs(PER_CELL, exist_ok=True)
os.makedirs(WALLS, exist_ok=True)
os.makedirs(SURFACES, exist_ok=True)

M_LIST = [3, 4, 5, 8, 9]
SEEDS = [0, 1, 2]
P_FOR_M = {3: [2, 3], 4: [2, 3, 4], 5: [2, 3, 4, 5],
           8: [2, 3, 4, 5, 6], 9: [2, 3, 4, 5, 6]}
ALL_P = sorted({P for ps in P_FOR_M.values() for P in ps})
HELLINGER = [("PSE", "PSE  (Hellinger of power spectrum)"),
             ("D_H_state", "D_H state-space (Hellinger)")]
ALL_KEYS = HELLINGER + [("Dstsp", "Dstsp (KL state-space)"),
                        ("rmse_train_final", "training RMSE")]


def load_metrics():
    return json.load(open(os.path.join(OUT, "metrics_seedrun_v2.json")))


def median_seed(metrics, M, P):
    pse = metrics["per_M"][str(M)]["per_P"][str(P)]["summary"]["ortho"]["PSE"]["values"]
    pse = [p for p in pse if not (isinstance(p, float) and (p != p))]
    if not pse:
        return 0
    pse_idx = sorted(range(len(pse)), key=lambda i: pse[i])
    return SEEDS[pse_idx[len(pse_idx) // 2]]


def stats_for(metrics, M, P, key, fam):
    s = metrics["per_M"][str(M)]["per_P"][str(P)]["summary"][fam].get(key)
    if s is None:
        return float("nan"), float("nan")
    return s["mean"], s["std"]


def load_arr(M, P, seed):
    p = os.path.join(OUT, f"P{P}_M{M}_s{seed}_arrays.npz")
    if not os.path.exists(p):
        return None
    return np.load(p)


# ---------------------------------------------------------------------------
def draw_3d(ax, orb, color, title, lw=0.85, elev=25, azim=-60):
    orb = np.asarray(orb)
    ok = np.isfinite(orb).all(axis=1)
    if ok.sum() < 5:
        ax.text(0.5, 0.5, 0.5, "diverged", transform=ax.transAxes,
                ha="center", va="center", color="red")
        ax.set_title(title, fontsize=10)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
        return
    orb = orb[ok]
    ax.plot(orb[:, 0], orb[:, 1], orb[:, 2], color=color, lw=lw, alpha=0.95)
    ax.set_title(title, fontsize=10)
    ax.view_init(elev=elev, azim=azim)
    rng = float(max(np.ptp(orb[:, 0]), np.ptp(orb[:, 1]), np.ptp(orb[:, 2])))
    cx, cy, cz = orb[:, 0].mean(), orb[:, 1].mean(), orb[:, 2].mean()
    ax.set_xlim(cx - rng/2 - 0.3, cx + rng/2 + 0.3)
    ax.set_ylim(cy - rng/2 - 0.3, cy + rng/2 + 0.3)
    ax.set_zlim(cz - rng/2 - 0.3, cz + rng/2 + 0.3)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])


def draw_2d(ax, orb, color, title, ij=(0, 2), lw=0.5):
    orb = np.asarray(orb)
    ok = np.isfinite(orb).all(axis=1)
    if ok.sum() < 5:
        ax.text(0.5, 0.5, "diverged", transform=ax.transAxes,
                ha="center", va="center", color="red")
        ax.set_title(title, fontsize=9); return
    orb = orb[ok]; i, j = ij
    ax.plot(orb[:, i], orb[:, j], color=color, lw=lw, alpha=0.95)
    ax.set_title(title, fontsize=9)
    ax.set_xlabel(["x", "y", "z"][i], fontsize=8)
    ax.set_ylabel(["x", "y", "z"][j], fontsize=8)
    ax.set_aspect("equal", adjustable="datalim"); ax.tick_params(labelsize=7)


# ---------------------------------------------------------------------------
def make_per_cell_image(metrics, M, P, num_steps=5000, dpi=180):
    fig, axes = plt.subplots(3, 6, figsize=(22, 14),
                             gridspec_kw={"hspace": 0.35, "wspace": 0.25})
    pse_v_m, pse_v_s = stats_for(metrics, M, P, "PSE", "vanilla")
    pse_o_m, pse_o_s = stats_for(metrics, M, P, "PSE", "ortho")
    dh_v_m, dh_v_s = stats_for(metrics, M, P, "D_H_state", "vanilla")
    dh_o_m, dh_o_s = stats_for(metrics, M, P, "D_H_state", "ortho")
    rel_pse = (pse_o_m - pse_v_m) / pse_v_m * 100 if pse_v_m else 0.0
    rel_dh = (dh_o_m - dh_v_m) / dh_v_m * 100 if dh_v_m else 0.0
    fig.suptitle(
        f"M = {M}    P = {P}    (all 3 seeds; cols 1-3: 3-D, cols 4-6: x-z 2-D)\n"
        f"PSE        van {pse_v_m:.4f}+/-{pse_v_s:.4f}   ortho {pse_o_m:.4f}+/-{pse_o_s:.4f}    "
        f"rel gap {rel_pse:+.1f}%      "
        f"D_H_state  van {dh_v_m:.3f}+/-{dh_v_s:.3f}   ortho {dh_o_m:.3f}+/-{dh_o_s:.3f}    "
        f"rel gap {rel_dh:+.1f}%",
        fontsize=14, fontweight="bold")
    for ax in axes[:, :3].ravel():
        ax.remove()
    for r in range(3):
        for c in range(3):
            axes[r, c] = fig.add_subplot(3, 6, r * 6 + c + 1, projection="3d")

    rownames = ["Truth", "Vanilla", "Orthogonal"]
    for c, seed in enumerate(SEEDS):
        d = load_arr(M, P, seed)
        if d is None:
            continue
        triples = [(d["X_test_trans"][:num_steps], "black"),
                   (d["orbit_orig"][:num_steps], "#c0392b"),
                   (d["orbit_ortho"][:num_steps], "#2c5fb0")]
        for r, (orb, color) in enumerate(triples):
            draw_3d(axes[r, c], orb, color, f"{rownames[r]} seed {seed} 3-D")
            draw_2d(axes[r, c + 3], orb, color, f"{rownames[r]} seed {seed} x-z")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    out = os.path.join(PER_CELL, f"cell_M{M}_P{P}.png")
    fig.savefig(out, dpi=dpi); plt.close(fig)
    return out


def make_wall_all(metrics, num_steps=5000, dpi=160):
    fig, axes = plt.subplots(len(M_LIST) * 3, len(ALL_P),
                             figsize=(3.2 * len(ALL_P), 2.4 * len(M_LIST) * 3),
                             gridspec_kw={"hspace": 0.45, "wspace": 0.25})
    fig.suptitle("Trajectory wall (v2): all (M, P) cells, median-PSE seed, x-z projection",
                 fontsize=15, fontweight="bold")
    for mi, M in enumerate(M_LIST):
        for c, P in enumerate(ALL_P):
            for r in range(3):
                ax = axes[3 * mi + r, c]
                if P not in P_FOR_M[M]:
                    ax.axis("off"); continue
                seed = median_seed(metrics, M, P)
                d = load_arr(M, P, seed)
                if d is None:
                    ax.axis("off"); continue
                triples = [(d["X_test_trans"][:num_steps], "black", "Truth"),
                           (d["orbit_orig"][:num_steps], "#c0392b", "Vanilla"),
                           (d["orbit_ortho"][:num_steps], "#2c5fb0", "Ortho")]
                orb, color, name = triples[r]
                draw_2d(ax, orb, color,
                        f"M={M} P={P} {name} seed{seed}", lw=0.45)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = os.path.join(WALLS, "wall_all.png")
    fig.savefig(out, dpi=dpi); plt.close(fig)
    return out


# ---------------------------------------------------------------------------
def grid(metrics, family, key):
    G = np.full((len(M_LIST), len(ALL_P)), np.nan)
    for i, M in enumerate(M_LIST):
        for j, P in enumerate(ALL_P):
            if P not in P_FOR_M[M]:
                continue
            s = metrics["per_M"][str(M)]["per_P"][str(P)]["summary"].get(family, {}).get(key)
            if s is not None and not (isinstance(s["mean"], float) and s["mean"] != s["mean"]):
                G[i, j] = s["mean"]
    return G


def make_surface(metrics, key, label, savepath):
    """3-D surface plot M x P x metric, one surface per family."""
    Gv = grid(metrics, "vanilla", key)
    Go = grid(metrics, "ortho", key)
    Mg, Pg = np.meshgrid(M_LIST, ALL_P, indexing="ij")
    fig = plt.figure(figsize=(14, 6.5))
    for col, (G, color, name) in enumerate(
            [(Gv, "Reds", "Vanilla"), (Go, "Blues", "Orthogonal")]):
        ax = fig.add_subplot(1, 2, col + 1, projection="3d")
        # mask NaN cells
        mask = np.isnan(G)
        Gp = np.where(mask, np.nanmin(G), G)
        ax.plot_surface(Mg.astype(float), Pg.astype(float), Gp,
                        cmap=color, alpha=0.85, edgecolor="k",
                        linewidth=0.3, rcount=len(M_LIST), ccount=len(ALL_P))
        # scatter the actual valid data points
        valid = ~mask
        ax.scatter(Mg[valid], Pg[valid], G[valid],
                   color="black", s=20, alpha=0.8)
        ax.set_xlabel("M"); ax.set_ylabel("P"); ax.set_zlabel(label, fontsize=9)
        ax.set_title(f"{name}: {label}")
        ax.view_init(elev=30, azim=-65)
    fig.suptitle(f"3-D performance surface: {label}",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(savepath, dpi=140); plt.close(fig)


def make_bars3d(metrics, key, label, savepath):
    """3-D bar plot (M, P, metric) with vanilla and ortho side by side."""
    Gv = grid(metrics, "vanilla", key)
    Go = grid(metrics, "ortho", key)
    fig = plt.figure(figsize=(13, 7))
    ax = fig.add_subplot(111, projection="3d")
    width = 0.35
    for i, M in enumerate(M_LIST):
        for j, P in enumerate(ALL_P):
            if P not in P_FOR_M[M]:
                continue
            v = Gv[i, j]; o = Go[i, j]
            if not np.isnan(v):
                ax.bar3d(M - width/2, P - width/2, 0, width, width, v,
                         color="#c0392b", alpha=0.7, edgecolor="black", linewidth=0.4)
            if not np.isnan(o):
                ax.bar3d(M + width/2, P - width/2, 0, width, width, o,
                         color="#2c5fb0", alpha=0.7, edgecolor="black", linewidth=0.4)
    ax.set_xlabel("M"); ax.set_ylabel("P"); ax.set_zlabel(label, fontsize=10)
    ax.set_xticks(M_LIST); ax.set_yticks(ALL_P)
    ax.set_title(f"3-D bars: {label}  (red = vanilla, blue = ortho)")
    ax.view_init(elev=22, azim=-60)
    fig.tight_layout()
    fig.savefig(savepath, dpi=140); plt.close(fig)


def make_lines_M(metrics, key, label, savepath):
    fig, ax = plt.subplots(1, 1, figsize=(9, 5.5))
    cmap = plt.cm.viridis(np.linspace(0.05, 0.85, len(ALL_P)))
    for j, P in enumerate(ALL_P):
        Mxs_v = []; ys_v = []; es_v = []
        Mxs_o = []; ys_o = []; es_o = []
        for M in M_LIST:
            if P not in P_FOR_M[M]:
                continue
            sv = metrics["per_M"][str(M)]["per_P"][str(P)]["summary"].get("vanilla", {}).get(key)
            so = metrics["per_M"][str(M)]["per_P"][str(P)]["summary"].get("ortho", {}).get(key)
            if sv:
                Mxs_v.append(M); ys_v.append(sv["mean"]); es_v.append(sv["std"])
            if so:
                Mxs_o.append(M); ys_o.append(so["mean"]); es_o.append(so["std"])
        if Mxs_v:
            ax.errorbar(Mxs_v, ys_v, yerr=es_v, fmt="--o", color=cmap[j], alpha=0.85,
                        label=f"vanilla P={P}", markersize=5, capsize=2)
        if Mxs_o:
            ax.errorbar(Mxs_o, ys_o, yerr=es_o, fmt="-s", color=cmap[j], alpha=0.85,
                        label=f"ortho  P={P}", markersize=5, capsize=2)
    ax.set_xlabel("M"); ax.set_ylabel(label)
    ax.set_title(f"{label} vs M  (vanilla dashed, ortho solid; mean +/- std over 3 seeds)")
    if key != "Dstsp":
        ax.set_yscale("log")
    ax.grid(True, alpha=0.3); ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(savepath, dpi=140); plt.close(fig)


def make_lines_P(metrics, key, label, savepath):
    fig, ax = plt.subplots(1, 1, figsize=(9, 5.5))
    cmap = plt.cm.plasma(np.linspace(0.05, 0.85, len(M_LIST)))
    for i, M in enumerate(M_LIST):
        Ps_v = []; ys_v = []; es_v = []
        Ps_o = []; ys_o = []; es_o = []
        for P in P_FOR_M[M]:
            sv = metrics["per_M"][str(M)]["per_P"][str(P)]["summary"].get("vanilla", {}).get(key)
            so = metrics["per_M"][str(M)]["per_P"][str(P)]["summary"].get("ortho", {}).get(key)
            if sv:
                Ps_v.append(P); ys_v.append(sv["mean"]); es_v.append(sv["std"])
            if so:
                Ps_o.append(P); ys_o.append(so["mean"]); es_o.append(so["std"])
        if Ps_v:
            ax.errorbar(Ps_v, ys_v, yerr=es_v, fmt="--o", color=cmap[i], alpha=0.85,
                        label=f"vanilla M={M}", markersize=5, capsize=2)
        if Ps_o:
            ax.errorbar(Ps_o, ys_o, yerr=es_o, fmt="-s", color=cmap[i], alpha=0.85,
                        label=f"ortho  M={M}", markersize=5, capsize=2)
    ax.set_xlabel("P"); ax.set_ylabel(label)
    ax.set_title(f"{label} vs P  (vanilla dashed, ortho solid; mean +/- std over 3 seeds)")
    if key != "Dstsp":
        ax.set_yscale("log")
    ax.grid(True, alpha=0.3); ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(savepath, dpi=140); plt.close(fig)


def make_gap_surface(metrics, key, label, savepath):
    """A single relative-gap surface (ortho / vanilla - 1) over (M, P)."""
    Gv = grid(metrics, "vanilla", key); Go = grid(metrics, "ortho", key)
    diff = (Go - Gv) / Gv * 100
    Mg, Pg = np.meshgrid(M_LIST, ALL_P, indexing="ij")
    fig = plt.figure(figsize=(9, 6.5))
    ax = fig.add_subplot(111, projection="3d")
    mask = np.isnan(diff)
    Z = np.where(mask, 0, diff)
    surf = ax.plot_surface(Mg.astype(float), Pg.astype(float), Z,
                           cmap="RdBu_r", alpha=0.85, edgecolor="k",
                           linewidth=0.3, rcount=len(M_LIST), ccount=len(ALL_P),
                           vmin=-max(abs(np.nanmin(diff)), abs(np.nanmax(diff))),
                           vmax=+max(abs(np.nanmin(diff)), abs(np.nanmax(diff))))
    valid = ~mask
    ax.scatter(Mg[valid], Pg[valid], diff[valid], color="black", s=20, alpha=0.8)
    ax.set_xlabel("M"); ax.set_ylabel("P")
    ax.set_zlabel("rel gap % (ortho − vanilla)/vanilla", fontsize=9)
    ax.set_title(f"3-D gap surface: {label}\nblue = ortho wins, red = vanilla wins")
    ax.view_init(elev=30, azim=-65)
    fig.colorbar(surf, ax=ax, fraction=0.045, label="rel gap %")
    fig.tight_layout()
    fig.savefig(savepath, dpi=140); plt.close(fig)


# ---------------------------------------------------------------------------
def main():
    metrics = load_metrics()

    print("Per-cell trajectory images ...")
    cell_paths = []
    for M in M_LIST:
        for P in P_FOR_M[M]:
            try:
                p = make_per_cell_image(metrics, M, P)
                cell_paths.append((M, P, p))
                print(f"  cell M={M} P={P} -> {p}")
            except Exception as e:
                print(f"  failed M={M} P={P}: {e}")

    print("Wall image ...")
    wall = make_wall_all(metrics)
    print(f"  {wall}")

    print("3-D surfaces, bars, line plots ...")
    surf_files = []
    for slug, label in ALL_KEYS:
        p = os.path.join(SURFACES, f"surface_{slug}.png"); make_surface(metrics, slug, label, p); surf_files.append(p)
        p = os.path.join(SURFACES, f"bars3d_{slug}.png"); make_bars3d(metrics, slug, label, p); surf_files.append(p)
        p = os.path.join(SURFACES, f"gapsurf_{slug}.png"); make_gap_surface(metrics, slug, label, p); surf_files.append(p)
        p = os.path.join(SURFACES, f"lines_M_{slug}.png"); make_lines_M(metrics, slug, label, p); surf_files.append(p)
        p = os.path.join(SURFACES, f"lines_P_{slug}.png"); make_lines_P(metrics, slug, label, p); surf_files.append(p)

    print("Stitching PDF ...")
    pdf_path = os.path.join(DEST, "v2_visuals_report.pdf")
    with PdfPages(pdf_path) as pdf:
        # cover
        fig = plt.figure(figsize=(8.5, 11))
        ax = fig.add_axes([0.07, 0.04, 0.86, 0.92]); ax.axis("off")
        ax.text(0.0, 0.97, "Seedrun v2 visuals: trajectories + 3-D (M, P, metric)",
                fontsize=14, fontweight="bold", transform=ax.transAxes, va="top")
        for i, ln in enumerate([
            "",
            "This file collects the trajectory comparison and the (M, P, metric)",
            "visualisations for the seedrun_v2 sweep.",
            "",
            "Sections:",
            "  1. Wall image: every (M, P) cell at the median-PSE seed.",
            "  2. 3-D performance surfaces (M x P x metric) for both models.",
            "  3. 3-D bar plots (M x P x metric) and gap surfaces.",
            "  4. M-cross-sections: metric vs M, one curve per P.",
            "  5. P-cross-sections: metric vs P, one curve per M.",
            "  6. Per-cell trajectories at high resolution: 19 pages",
            "     showing all 3 seeds in 3-D and x-z 2-D.",
        ]):
            ax.text(0.0, 0.92 - 0.03 * i, ln, fontsize=10, family="monospace",
                    transform=ax.transAxes, va="top")
        pdf.savefig(fig); plt.close(fig)

        # wall
        img = plt.imread(wall)
        fig = plt.figure(figsize=(11, 14))
        ax = fig.add_axes([0.02, 0.02, 0.96, 0.96])
        ax.imshow(img); ax.axis("off"); pdf.savefig(fig); plt.close(fig)

        for path in surf_files:
            if not os.path.exists(path): continue
            img = plt.imread(path)
            # landscape for the 3-D plots
            fig = plt.figure(figsize=(11, 8.5))
            ax = fig.add_axes([0.02, 0.02, 0.96, 0.96])
            ax.imshow(img); ax.axis("off")
            pdf.savefig(fig); plt.close(fig)

        # per-cell pages, ordered by (M, P)
        for M, P, path in cell_paths:
            img = plt.imread(path)
            fig = plt.figure(figsize=(11, 8.5))
            ax = fig.add_axes([0.02, 0.02, 0.96, 0.96])
            ax.imshow(img); ax.axis("off")
            pdf.savefig(fig); plt.close(fig)

    print(f"wrote {pdf_path}  ({os.path.getsize(pdf_path)/1024/1024:.1f} MB)")


if __name__ == "__main__":
    main()
