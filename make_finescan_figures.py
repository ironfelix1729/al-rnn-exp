# -*- coding: utf-8 -*-
"""Figures and a PDF report for the M-finescan sweep.

Reads results_M_finescan/metrics_all.json (produced by compute_finescan_metrics.py)
and renders:

  finescan_heatmap_<metric>.png   - vanilla vs ortho heatmaps over (M, P)
  finescan_gap_<metric>.png       - ortho - vanilla gap heatmap (negative = ortho wins)
  finescan_trends.png             - line plots: each metric vs M for each P
  finescan_trajectories.png       - 2-D x-z projection grid: rows (M, P), cols (truth/vanilla/ortho)
  finescan_matched_pairs.png      - for each metric, ortho(M_small) ≈ vanilla(M_big)
  finescan_report.pdf             - all of the above stitched

Tolerates missing cells so it can be re-run while the sweep is still going.
"""
import os, json, math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_M_finescan")
M_LIST = [10, 11, 14, 15]
P_LIST = [2, 3, 4, 5, 6]
METRICS = [
    ("rmse_train_final", "training MSE",   "lower"),
    ("rmse_h100",        "100-step RMSE",  "lower"),
    ("Dstsp",            "Dstsp (KL)",     "lower"),
    ("PSE",              "PSE (Hellinger of power spectrum)", "lower"),
]
LABELS_3D = ["x", "y", "z"]
PAIRS_2D = [(0, 2)]  # x-z only for compactness


# ---------------------------------------------------------------------------
def load_metrics():
    with open(os.path.join(OUT, "metrics_all.json")) as f:
        return json.load(f)


def grid(metrics, family, key):
    G = np.full((len(M_LIST), len(P_LIST)), np.nan)
    for i, M in enumerate(M_LIST):
        per_P = metrics.get(str(M), {}).get("per_P", {})
        for j, P in enumerate(P_LIST):
            cell = per_P.get(str(P))
            if cell and family in cell and key in cell[family]:
                G[i, j] = cell[family][key]
    return G


# ---------------------------------------------------------------------------
def heatmap_pair(metrics, key, label, savepath):
    Gv = grid(metrics, "vanilla", key)
    Go = grid(metrics, "ortho", key)
    vmin = float(np.nanmin([Gv, Go]))
    vmax = float(np.nanmax([Gv, Go]))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for ax, G, name in zip(axes, [Gv, Go], ["Vanilla", "Orthogonal"]):
        im = ax.imshow(G, cmap="viridis", vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_xticks(range(len(P_LIST))); ax.set_xticklabels([f"P={P}" for P in P_LIST])
        ax.set_yticks(range(len(M_LIST))); ax.set_yticklabels([f"M={M}" for M in M_LIST])
        ax.set_title(f"{name}: {label}", fontsize=11)
        for i in range(len(M_LIST)):
            for j in range(len(P_LIST)):
                if not np.isnan(G[i, j]):
                    ax.text(j, i, f"{G[i, j]:.3g}", ha="center", va="center",
                            color="white" if G[i, j] < (vmin + vmax) / 2 else "black",
                            fontsize=8)
        plt.colorbar(im, ax=ax, fraction=0.045)
    fig.suptitle(f"{label}  (lower is better)", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(savepath, dpi=140)
    plt.close(fig)


def heatmap_gap(metrics, key, label, savepath):
    Gv = grid(metrics, "vanilla", key)
    Go = grid(metrics, "ortho", key)
    diff = (Go - Gv) / Gv * 100  # percentage; negative = ortho wins
    vmax = float(np.nanmax(np.abs(diff)))
    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    im = ax.imshow(diff, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(P_LIST))); ax.set_xticklabels([f"P={P}" for P in P_LIST])
    ax.set_yticks(range(len(M_LIST))); ax.set_yticklabels([f"M={M}" for M in M_LIST])
    for i in range(len(M_LIST)):
        for j in range(len(P_LIST)):
            if not np.isnan(diff[i, j]):
                ax.text(j, i, f"{diff[i, j]:+.0f}%", ha="center", va="center",
                        fontsize=9, color="black")
    ax.set_title(f"Gap (ortho - vanilla) on {label}\nblue = ortho wins, red = vanilla wins")
    plt.colorbar(im, ax=ax, fraction=0.045)
    fig.tight_layout()
    fig.savefig(savepath, dpi=140)
    plt.close(fig)


def trends_plot(metrics, savepath):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    cmap = plt.cm.viridis(np.linspace(0.05, 0.85, len(P_LIST)))
    for ax, (key, label, _) in zip(axes.flat, METRICS):
        Gv = grid(metrics, "vanilla", key)
        Go = grid(metrics, "ortho", key)
        for j, P in enumerate(P_LIST):
            ax.plot(M_LIST, Gv[:, j], "--o", color=cmap[j], alpha=0.85,
                    label=f"vanilla P={P}", markersize=6)
            ax.plot(M_LIST, Go[:, j], "-s", color=cmap[j], alpha=0.85,
                    label=f"ortho  P={P}", markersize=6)
        ax.set_xlabel("M")
        ax.set_ylabel(label)
        ax.set_title(label)
        if key in ("Dstsp",):
            pass  # linear; KL has its own scale
        else:
            ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
    axes[0, 0].legend(fontsize=7, ncol=2, loc="upper right")
    fig.suptitle("Metrics vs M, by P  (vanilla dashed, ortho solid)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(savepath, dpi=140)
    plt.close(fig)


# ---------------------------------------------------------------------------
def trajectory_grid(metrics, savepath, num_steps=5000):
    """One row per (M, P) cell; columns = truth / vanilla / ortho (x-z projection)."""
    rows = []
    for M in M_LIST:
        for P in P_LIST:
            arr_path = os.path.join(OUT, f"P{P}_M{M}_arrays.npz")
            if os.path.exists(arr_path):
                rows.append((M, P, arr_path))
    fig, axes = plt.subplots(len(rows), 3, figsize=(13, 2.5 * len(rows)))
    if len(rows) == 1:
        axes = np.array([axes])
    for r, (M, P, path) in enumerate(rows):
        d = np.load(path)
        sources = [(d["X_test_trans"][:num_steps], "black",   f"truth      M={M} P={P}"),
                   (d["orbit_orig"][:num_steps],  "#c0392b", f"vanilla    M={M} P={P}"),
                   (d["orbit_ortho"][:num_steps], "#2c5fb0", f"orthogonal M={M} P={P}")]
        for c, (orb, color, name) in enumerate(sources):
            ax = axes[r, c]
            ax.plot(orb[:, 0], orb[:, 2], color=color, lw=0.45, alpha=0.92)
            ax.set_title(name, fontsize=8)
            ax.set_aspect("equal", adjustable="datalim")
            ax.tick_params(labelsize=6)
            if c == 0:
                ax.set_ylabel("z", fontsize=7)
            if r == len(rows) - 1:
                ax.set_xlabel("x", fontsize=7)
    fig.suptitle("M-finescan: free-run x-z projections", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.99])
    fig.savefig(savepath, dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------
def matched_pairs_table(metrics, key, label):
    """For each ortho cell, find the smallest vanilla M' >= ortho's M+1 with similar value."""
    rows = []
    for i_o, M_o in enumerate(M_LIST):
        for j_o, P in enumerate(P_LIST):
            cell_o = metrics.get(str(M_o), {}).get("per_P", {}).get(str(P))
            if not cell_o or "ortho" not in cell_o or key not in cell_o["ortho"]:
                continue
            v_o = cell_o["ortho"][key]
            best = None
            for M_v in M_LIST:
                if M_v <= M_o:
                    continue
                cell_v = metrics.get(str(M_v), {}).get("per_P", {}).get(str(P))
                if not cell_v or "vanilla" not in cell_v or key not in cell_v["vanilla"]:
                    continue
                v_v = cell_v["vanilla"][key]
                if v_v <= v_o * 1.20 and v_v >= v_o * 0.80:  # within +/- 20 %
                    if best is None or M_v < best["M"]:
                        best = {"M": M_v, "val": v_v}
            if best:
                rows.append({"P": P, "ortho_M": M_o, "ortho_val": v_o,
                             "vanilla_M": best["M"], "vanilla_val": best["val"]})
    return rows


def matched_pairs_text(metrics):
    lines = ["Matched (M_ortho < M_vanilla) pairs at same P, |Δ value| ≤ 20 %", "=" * 76]
    for key, label, _ in METRICS:
        rows = matched_pairs_table(metrics, key, label)
        if not rows:
            continue
        lines.append("")
        lines.append(f"--- on {label} ---")
        lines.append(f"  P  ortho M -> val   |   vanilla M -> val")
        for r in rows:
            lines.append(f"  {r['P']}  M={r['ortho_M']:>3} -> {r['ortho_val']:.4g}   "
                         f"|   M={r['vanilla_M']:>3} -> {r['vanilla_val']:.4g}")
    txt = "\n".join(lines)
    with open(os.path.join(OUT, "matched_pairs.txt"), "w") as f:
        f.write(txt + "\n")
    return txt


# ---------------------------------------------------------------------------
def make_pdf(metrics, paths):
    pdf_path = os.path.join(OUT, "finescan_report.pdf")
    with PdfPages(pdf_path) as pdf:
        # Page 1: title + brief
        fig = plt.figure(figsize=(8.5, 11))
        ax = fig.add_axes([0.07, 0.04, 0.86, 0.92]); ax.axis("off")
        y = 0.97
        ax.text(0.0, y, "M-finescan: vanilla vs orthogonal AL-RNN",
                fontsize=15, fontweight="bold", transform=ax.transAxes, va="top")
        y -= 0.06
        for ln in [
            "Sweep:  M in {10, 11, 14, 15}  x  P in {2, 3, 4, 5, 6}",
            "Epochs: 1000   alpha: 1.0   shared frozen B per M   seed: 42",
            "",
            "Metrics:",
            "  rmse_train_final - sqrt(final training MSE), one-step prediction",
            "  rmse_h100        - free-run RMSE on the first 100 steps after t=0",
            "  Dstsp            - KL(p_true || p_gen) on a 30^3 binned 3-D density",
            "  PSE              - mean Hellinger distance between smoothed power spectra",
            "                     (the AL-RNN paper's attractor-quality metrics)",
        ]:
            ax.text(0.0, y, ln, fontsize=10, family="monospace",
                    transform=ax.transAxes, va="top")
            y -= 0.022
        pdf.savefig(fig); plt.close(fig)
        # Each metric: heatmap + gap heatmap
        for png in paths:
            img = plt.imread(png)
            fig = plt.figure(figsize=(8.5, 11))
            ax = fig.add_axes([0.02, 0.02, 0.96, 0.96])
            ax.imshow(img); ax.axis("off")
            pdf.savefig(fig); plt.close(fig)
    print(f"wrote {pdf_path}")


def main():
    metrics = load_metrics()

    paths = []
    for key, label, _ in METRICS:
        slug = key.replace(" ", "_")
        p1 = os.path.join(OUT, f"finescan_heatmap_{slug}.png")
        p2 = os.path.join(OUT, f"finescan_gap_{slug}.png")
        heatmap_pair(metrics, key, label, p1); paths.append(p1)
        heatmap_gap(metrics, key, label, p2); paths.append(p2)
    p3 = os.path.join(OUT, "finescan_trends.png")
    trends_plot(metrics, p3); paths.append(p3)
    p4 = os.path.join(OUT, "finescan_trajectories.png")
    trajectory_grid(metrics, p4); paths.append(p4)

    txt = matched_pairs_text(metrics)
    print(txt)

    make_pdf(metrics, paths)


if __name__ == "__main__":
    main()
