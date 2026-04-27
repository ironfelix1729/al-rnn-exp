# -*- coding: utf-8 -*-
"""Figures + PDF for the multi-seed M-finescan sweep.

Reads results_seedrun/metrics_seedrun.json (produced by aggregate_seedrun.py).

Outputs into results_seedrun/:
  seed_heatmap_<key>_mean.png        - vanilla vs ortho mean over seeds
  seed_heatmap_<key>_gap.png         - relative gap (ortho - vanilla) on the mean
  seed_winrate_<key>.png             - fraction of seeds where ortho beats vanilla
  seed_box_<key>.png                 - per-cell box plots over seeds
  seed_trends.png                    - line plot mean+-std vs M, by P, all metrics
  seed_matched_pairs_<key>.png       - ortho(small M) vs vanilla(big M) within 20% on means
  seed_report.pdf                    - all of the above
"""
import os, json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_seedrun")
M_LIST = [10, 11, 14, 15]
P_LIST = [2, 3, 4, 5, 6]
SEEDS = [0, 1, 2]
METRICS = [
    ("rmse_train_final", "training RMSE"),
    ("Dstsp",            "Dstsp (KL)"),
    ("PSE",              "PSE (Hellinger of power spectrum)"),
    ("D_H_state",        "D_H state-space (Hellinger)"),
]


def load():
    return json.load(open(os.path.join(OUT, "metrics_seedrun.json")))


def grid_mean(metrics, family, key):
    G = np.full((len(M_LIST), len(P_LIST)), np.nan)
    GS = np.full((len(M_LIST), len(P_LIST)), np.nan)
    for i, M in enumerate(M_LIST):
        per_P = metrics.get("per_M", {}).get(str(M), {}).get("per_P", {})
        for j, P in enumerate(P_LIST):
            cell = per_P.get(str(P), {})
            s = cell.get("summary", {}).get(family, {}).get(key)
            if s is not None:
                G[i, j] = s["mean"]
                GS[i, j] = s["std"]
    return G, GS


def winrate(metrics, key):
    """Fraction of seeds where ortho < vanilla (for "lower is better" metrics)."""
    W = np.full((len(M_LIST), len(P_LIST)), np.nan)
    for i, M in enumerate(M_LIST):
        per_P = metrics.get("per_M", {}).get(str(M), {}).get("per_P", {})
        for j, P in enumerate(P_LIST):
            cell = per_P.get(str(P), {})
            v = cell.get("summary", {}).get("vanilla", {}).get(key, {}).get("values", [])
            o = cell.get("summary", {}).get("ortho", {}).get(key, {}).get("values", [])
            if not v or not o or len(v) != len(o):
                continue
            wins = sum(1 for a, b in zip(o, v) if a < b)
            W[i, j] = wins / len(v)
    return W


def heatmap_mean(metrics, key, label, savepath):
    Gv, _ = grid_mean(metrics, "vanilla", key)
    Go, _ = grid_mean(metrics, "ortho", key)
    vmin = float(np.nanmin([Gv, Go])); vmax = float(np.nanmax([Gv, Go]))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for ax, G, name in zip(axes, [Gv, Go], ["Vanilla mean", "Orthogonal mean"]):
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
    fig.suptitle(f"{label}  (mean over {len(SEEDS)} seeds, lower is better)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(savepath, dpi=140); plt.close(fig)


def heatmap_gap(metrics, key, label, savepath):
    Gv, _ = grid_mean(metrics, "vanilla", key)
    Go, _ = grid_mean(metrics, "ortho", key)
    diff = (Go - Gv) / Gv * 100  # negative = ortho wins on the mean
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
    ax.set_title(f"Gap on mean {label}\nblue = ortho wins, red = vanilla wins")
    plt.colorbar(im, ax=ax, fraction=0.045)
    fig.tight_layout()
    fig.savefig(savepath, dpi=140); plt.close(fig)


def heatmap_winrate(metrics, key, label, savepath):
    W = winrate(metrics, key)
    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    im = ax.imshow(W, cmap="RdBu", vmin=0.0, vmax=1.0, aspect="auto")
    ax.set_xticks(range(len(P_LIST))); ax.set_xticklabels([f"P={P}" for P in P_LIST])
    ax.set_yticks(range(len(M_LIST))); ax.set_yticklabels([f"M={M}" for M in M_LIST])
    for i in range(len(M_LIST)):
        for j in range(len(P_LIST)):
            if not np.isnan(W[i, j]):
                ax.text(j, i, f"{W[i, j]:.2f}", ha="center", va="center",
                        fontsize=10, color="black")
    ax.set_title(f"Ortho win rate over seeds on {label}")
    plt.colorbar(im, ax=ax, fraction=0.045)
    fig.tight_layout()
    fig.savefig(savepath, dpi=140); plt.close(fig)


def trends_plot(metrics, savepath):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    cmap = plt.cm.viridis(np.linspace(0.05, 0.85, len(P_LIST)))
    for ax, (key, label) in zip(axes.flat, METRICS):
        Gv, GvS = grid_mean(metrics, "vanilla", key)
        Go, GoS = grid_mean(metrics, "ortho", key)
        for j, P in enumerate(P_LIST):
            ax.errorbar(M_LIST, Gv[:, j], yerr=GvS[:, j], fmt="--o",
                        color=cmap[j], alpha=0.85, label=f"vanilla P={P}",
                        markersize=5, capsize=2)
            ax.errorbar(M_LIST, Go[:, j], yerr=GoS[:, j], fmt="-s",
                        color=cmap[j], alpha=0.85, label=f"ortho  P={P}",
                        markersize=5, capsize=2)
        ax.set_xlabel("M"); ax.set_ylabel(label); ax.set_title(label)
        if key != "Dstsp":
            ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
    axes[0, 0].legend(fontsize=7, ncol=2, loc="upper right")
    fig.suptitle(f"Metrics vs M, mean +- std over {len(SEEDS)} seeds  (vanilla dashed, ortho solid)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(savepath, dpi=140); plt.close(fig)


def boxplot(metrics, key, label, savepath):
    """Box plot per (M, P) of vanilla & ortho values across seeds."""
    fig, axes = plt.subplots(len(M_LIST), len(P_LIST),
                             figsize=(3.0 * len(P_LIST), 2.8 * len(M_LIST)),
                             sharey="row")
    for i, M in enumerate(M_LIST):
        for j, P in enumerate(P_LIST):
            ax = axes[i, j]
            cell = metrics.get("per_M", {}).get(str(M), {}).get("per_P", {}).get(str(P), {})
            v = cell.get("summary", {}).get("vanilla", {}).get(key, {}).get("values", [])
            o = cell.get("summary", {}).get("ortho", {}).get(key, {}).get("values", [])
            if v and o:
                bp = ax.boxplot([v, o], positions=[1, 2], widths=0.6,
                                patch_artist=True, showfliers=False)
                for patch, color in zip(bp['boxes'], ["#c0392b", "#2c5fb0"]):
                    patch.set_facecolor(color); patch.set_alpha(0.6)
                ax.scatter([1] * len(v), v, s=12, color="darkred",  alpha=0.8)
                ax.scatter([2] * len(o), o, s=12, color="darkblue", alpha=0.8)
            ax.set_xticks([1, 2]); ax.set_xticklabels(["van", "ortho"], fontsize=7)
            ax.tick_params(labelsize=7)
            if j == 0:
                ax.set_ylabel(f"M={M}", fontsize=10)
            if i == 0:
                ax.set_title(f"P={P}", fontsize=10)
    fig.suptitle(f"{label}  (per-seed values; box = IQR, line = median)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(savepath, dpi=140); plt.close(fig)


def matched_pairs_text(metrics):
    lines = ["Matched (M_ortho < M_vanilla, same P) pairs at 20% mean tolerance",
             "=" * 76]
    for key, label in METRICS:
        Gv, _ = grid_mean(metrics, "vanilla", key)
        Go, _ = grid_mean(metrics, "ortho", key)
        lines.append("")
        lines.append(f"--- on {label} ---")
        lines.append(f"  P  ortho M -> mean   |   vanilla M -> mean")
        for i_o, M_o in enumerate(M_LIST):
            for j, P in enumerate(P_LIST):
                vo = Go[i_o, j]
                if np.isnan(vo): continue
                best = None
                for i_v, M_v in enumerate(M_LIST):
                    if M_v <= M_o: continue
                    vv = Gv[i_v, j]
                    if np.isnan(vv): continue
                    if 0.80 <= vv / vo <= 1.20:
                        if best is None or M_v < best[0]:
                            best = (M_v, vv)
                if best:
                    lines.append(f"  {P}  M={M_o:>3} -> {vo:.4g}   |   M={best[0]:>3} -> {best[1]:.4g}")
    txt = "\n".join(lines)
    with open(os.path.join(OUT, "matched_pairs_seed.txt"), "w") as f:
        f.write(txt + "\n")
    return txt


def make_pdf(metrics, image_paths):
    pdf_path = os.path.join(OUT, "seed_report.pdf")
    with PdfPages(pdf_path) as pdf:
        # title
        fig = plt.figure(figsize=(8.5, 11))
        ax = fig.add_axes([0.07, 0.04, 0.86, 0.92]); ax.axis("off")
        ax.text(0.0, 0.97, "Multi-seed M-finescan: vanilla vs orthogonal AL-RNN",
                fontsize=15, fontweight="bold", transform=ax.transAxes, va="top")
        y = 0.88
        for ln in [
            f"Sweep:  M in {M_LIST}  x  P in {P_LIST}  x  seeds in {SEEDS}  x  "
            f"{{vanilla, ortho}}",
            "Epochs: 1000   alpha: 1.0   shared frozen B per M (seed_B = 42)",
            "                          fixed Q_dataset rotation (seed_data = 42)",
            "Per-seed seed_run controls model init + Python random batch sampling.",
            "",
            "Metrics (lower is better):",
            "  rmse_train_final - sqrt(final training MSE), one-step prediction",
            "  Dstsp            - KL(p_true || p_gen) on a 30^3 binned 3-D density",
            "  PSE              - mean Hellinger distance between smoothed power spectra",
            "  D_H_state        - Hellinger distance on the same state-space density",
            "",
            "Per-cell summary uses mean +- std over the seeds; the 'win rate' heatmap",
            "shows the fraction of seeds where ortho < vanilla on that metric.",
        ]:
            ax.text(0.0, y, ln, fontsize=10, family="monospace",
                    transform=ax.transAxes, va="top")
            y -= 0.022
        pdf.savefig(fig); plt.close(fig)
        for png in image_paths:
            img = plt.imread(png)
            fig = plt.figure(figsize=(8.5, 11))
            ax = fig.add_axes([0.02, 0.02, 0.96, 0.96])
            ax.imshow(img); ax.axis("off")
            pdf.savefig(fig); plt.close(fig)
    print(f"wrote {pdf_path}")


def main():
    metrics = load()
    paths = []
    for key, label in METRICS:
        slug = key
        p1 = os.path.join(OUT, f"seed_heatmap_{slug}_mean.png"); paths.append(p1)
        heatmap_mean(metrics, key, label, p1)
        p2 = os.path.join(OUT, f"seed_heatmap_{slug}_gap.png"); paths.append(p2)
        heatmap_gap(metrics, key, label, p2)
        p3 = os.path.join(OUT, f"seed_winrate_{slug}.png"); paths.append(p3)
        heatmap_winrate(metrics, key, label, p3)
        p4 = os.path.join(OUT, f"seed_box_{slug}.png"); paths.append(p4)
        boxplot(metrics, key, label, p4)
    p5 = os.path.join(OUT, "seed_trends.png"); paths.append(p5)
    trends_plot(metrics, p5)
    txt = matched_pairs_text(metrics); print(txt[:3000])
    make_pdf(metrics, paths)


if __name__ == "__main__":
    main()
