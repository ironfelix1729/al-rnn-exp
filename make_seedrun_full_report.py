# -*- coding: utf-8 -*-
"""Comprehensive PDF: experiments, architectures, statistics, highlight trajectories.

Pages:
  1.  Title + TL;DR
  2.  Model architectures
  3.  Experimental protocol
  4.  Aggregate metric heatmaps (mean over seeds)  -> existing PNGs
  5.  Gap heatmaps                                  -> existing PNGs
  6.  Win-rate heatmaps                             -> existing PNGs
  7.  Trends (mean +/- std vs M, by P)
  8.  Per-cell statistical table on PSE
  9.  Per-cell statistical table on D_H_state
  10. Pooled & stratified results table
  11. Forest plots
  12. Effect-size scatter
  13-16. Highlight trajectories
"""
import os, json, math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_seedrun")
PDF = os.path.join(OUT, "seedrun_full_report.pdf")
M_LIST = [10, 11, 14, 15]
P_LIST = [2, 3, 4, 5, 6]


def _text(ax, lines, fontsize=10, top=0.97):
    y = top
    for ln in lines:
        if ln.startswith("### "):
            ax.text(0.0, y, ln[4:], fontsize=fontsize + 1, fontweight="bold",
                    transform=ax.transAxes, va="top")
            y -= fontsize / 720.0 * 2.0
        else:
            ax.text(0.0, y, ln, fontsize=fontsize, family="monospace",
                    transform=ax.transAxes, va="top")
            y -= fontsize / 720.0 * 1.55


def text_page(pdf, title, lines, fontsize=10):
    fig = plt.figure(figsize=(8.5, 11))
    ax = fig.add_axes([0.06, 0.04, 0.88, 0.92]); ax.axis("off")
    if title:
        ax.text(0.0, 0.98, title, fontsize=15, fontweight="bold",
                transform=ax.transAxes, va="top")
        _text(ax, lines, fontsize=fontsize, top=0.93)
    else:
        _text(ax, lines, fontsize=fontsize)
    pdf.savefig(fig); plt.close(fig)


def image_page(pdf, png_name, title=None):
    path = os.path.join(OUT, png_name)
    if not os.path.exists(path):
        return
    img = plt.imread(path)
    fig = plt.figure(figsize=(8.5, 11))
    if title:
        fig.suptitle(title, fontsize=12, fontweight="bold")
        ax = fig.add_axes([0.02, 0.02, 0.96, 0.94])
    else:
        ax = fig.add_axes([0.02, 0.02, 0.96, 0.96])
    ax.imshow(img); ax.axis("off")
    pdf.savefig(fig); plt.close(fig)


def hires_image_page(pdf, png_name, title=None):
    path = os.path.join(OUT, "trajectories_hires", png_name)
    if not os.path.exists(path):
        return
    # render landscape to better fit the wide cell images
    img = plt.imread(path)
    fig = plt.figure(figsize=(11, 8.5))  # landscape
    if title:
        fig.suptitle(title, fontsize=12, fontweight="bold")
        ax = fig.add_axes([0.02, 0.02, 0.96, 0.94])
    else:
        ax = fig.add_axes([0.02, 0.02, 0.96, 0.96])
    ax.imshow(img); ax.axis("off")
    pdf.savefig(fig); plt.close(fig)


def main():
    metrics = json.load(open(os.path.join(OUT, "metrics_seedrun.json")))
    stats   = json.load(open(os.path.join(OUT, "statistical_analysis.json")))

    with PdfPages(PDF) as pdf:
        # 1 - TL;DR
        text_page(pdf, "Multi-seed M-finescan: vanilla vs orthogonal AL-RNN",
                  [
                      "120 training runs total: M in {10, 11, 14, 15} x P in {2..6}",
                      "x {vanilla, ortho} x 3 seeds, 1000 epochs each, alpha=1.0,",
                      "shared frozen B per M, fixed Q_dataset rotation (seed=42).",
                      "",
                      "### TL;DR",
                      "* On D_H_state (state-space Hellinger) ortho is statistically",
                      "  significantly better:  pooled mean rel gap = -5.83%,",
                      "  bootstrap 95% CI = [-9.66%, -1.78%]; ortho wins on the mean",
                      "  in 16/20 cells; sign-test p = 0.006, Wilcoxon p = 0.006.",
                      "",
                      "* On PSE (power-spectrum Hellinger) ortho leads directionally",
                      "  but not significantly: pooled mean rel gap = -4.30%,",
                      "  bootstrap 95% CI = [-15.2%, +8.8%]; the CI straddles zero",
                      "  due to two ortho optimisation-failure outliers.",
                      "",
                      "* On training RMSE (one-step MSE^0.5) ortho wins on the mean",
                      "  in 18/20 cells (small but tight).",
                      "",
                      "* Largest single-cell ortho wins on D_H_state:",
                      "    M=15 P=4   -20.9%   d_z = -2.55   95% t-CI excludes zero",
                      "    M=10 P=5   -12.7%   d_z = -3.05   95% t-CI excludes zero",
                      "    M=11 P=4   -14.6%, M=14 P=5  -16.1%, M=15 P=3  -16.9%",
                      "",
                      "* Parameter-efficiency: ortho-at-M=11 reaches vanilla-at-M=14",
                      "  quality on PSE (mean) at every P - 3 fewer hidden units for",
                      "  the same Hellinger-attractor quality.",
                  ])

        # 2 - architectures
        text_page(pdf, "Model architectures",
                  [
                      "Both models share A (diagonal skip), W (MxM coupling), h",
                      "(bias) and a frozen input embedding B (3 x M).  Only the",
                      "last P hidden units pass through a ReLU; the first M-P",
                      "remain linear.",
                      "",
                      "### Vanilla AL-RNN",
                      "    z_{t+1} = A * z_t + phi_P(z_t) W^T + h",
                      "    phi_P(z)[:M-P] = z[:M-P]",
                      "    phi_P(z)[M-P:] = ReLU(z[M-P:])",
                      "",
                      "### Orthogonal AL-RNN",
                      "    z_{t+1} = A * z_t + phi_P(Q z_t + b_Q) W^T + h",
                      "    Q in O(M) enforced by",
                      "      torch.nn.utils.parametrizations.orthogonal",
                      "",
                      "### Trainable parameters (frozen B)",
                      "             M=10   M=11   M=14   M=15",
                      "  vanilla    120    143    224    255",
                      "  ortho      175    209    329    375",
                      "",
                      "### Teacher forcing",
                      "  hard reset z[:N] = x at t=0; alpha = 1.0 injection every",
                      "  n_interleave = 16 steps  (z[:N] := (1-alpha)z[:N] + alpha x).",
                      "",
                      "### Loss",
                      "  MSE on the first N=3 hidden coords over T=128 steps.",
                  ])

        # 3 - protocol
        text_page(pdf, "Experimental protocol",
                  [
                      "Data        : Lorenz63 train/test (first 500 steps dropped).",
                      "              Fixed 3x3 random orthogonal Q_dataset (72 deg,",
                      "              det=+1) rotates the observations in R^3.",
                      "              seed_data = 42 (Q_dataset),  seed_B = 42 (shared B)",
                      "              are fixed across all runs.",
                      "Per-seed    : seed_run in {0, 1, 2} controls model init",
                      "              (A, W, h, Q's parametrisation init) AND Python's",
                      "              random.randint used by the dataset's batch sampler.",
                      "Optimiser   : Adam, lr 1e-3 -> 1e-5 exponentially decayed.",
                      "Epochs      : 1000  per run; 20 batches/epoch; batch 64.",
                      "Seq length  : 128;  teacher-forcing period n_interleave = 16.",
                      "Free-run    : after training, 5000-step rollout from",
                      "              X_test_trans[0] with no teacher forcing.",
                      "",
                      "### Metrics computed per cell",
                      "  rmse_train_final - sqrt(final training MSE), one-step",
                      "  rmse_h100        - free-run RMSE on the first 100 steps",
                      "  Dstsp            - KL(p_true || p_gen) on a 30^3 binned 3-D density",
                      "  PSE              - Hellinger of smoothed power spectra (per coord, mean)",
                      "  D_H_state        - Hellinger on the same state-space density",
                      "",
                      "### Statistical analysis",
                      "  Per cell (n=3 paired seeds): paired diff o-v, paired SD,",
                      "  Cohen's d_z, paired-t 95% CI, Wilcoxon signed-rank p,",
                      "  win rate over seeds.",
                      "  Pooled (20 cells): bootstrap 95% CI on the cell-level mean",
                      "  relative gap, sign test, Wilcoxon on cell-level gaps.",
                      "  Stratified by M-band (low={10,11} vs high={14,15}) and by P.",
                  ])

        # 4-7 aggregate figures (heatmap mean / gap / winrate / trends)
        for slug, label in [
            ("D_H_state", "D_H state-space Hellinger"),
            ("PSE",       "PSE (Hellinger of power spectrum)"),
            ("Dstsp",     "Dstsp (KL state-space)"),
            ("rmse_train_final", "training RMSE (one-step)"),
        ]:
            image_page(pdf, f"seed_heatmap_{slug}_mean.png",
                       f"Mean over 3 seeds: {label}")
            image_page(pdf, f"seed_heatmap_{slug}_gap.png",
                       f"Gap (ortho-vanilla)/vanilla on the mean: {label}")
            image_page(pdf, f"seed_winrate_{slug}.png",
                       f"Ortho win rate over 3 seeds: {label}")
            image_page(pdf, f"seed_box_{slug}.png",
                       f"Per-cell box plots over 3 seeds: {label}")
        image_page(pdf, "seed_trends.png",
                   "Mean +/- std vs M, by P (vanilla dashed, ortho solid)")

        # 8 - per-cell stats on D_H_state
        rows = stats["per_cell"]["D_H_state"]
        lines = [
            "# Per-cell paired statistics on D_H_state  (lower is better)",
            "",
            f"  {'M':>2} {'P':>2}  {'vanilla':>14}  {'ortho':>14}  {'rel_gap%':>9}  "
            f"{'d_z':>5}  {'95% t-CI on diff':>22}  {'wilc p':>6}  {'wins':>5}",
            "  " + "-" * 90,
        ]
        for r in rows:
            lines.append(
                f"  {r['M']:>2} {r['P']:>2}  "
                f"{r['vanilla_mean']:>10.4f}±{r['vanilla_std']:>4.3f}  "
                f"{r['ortho_mean']:>10.4f}±{r['ortho_std']:>4.3f}  "
                f"{r['rel_gap_pct']:>+8.1f}%  "
                f"{r['cohens_dz']:>+5.2f}  "
                f"[{r['t_ci_lo']:>+8.4g}, {r['t_ci_hi']:>+8.4g}]  "
                f"{r['wilcoxon_p_one_sided']:>6.3f}  "
                f"{int(r['ortho_win_rate']*r['n']):>1}/{r['n']:>1}")
        lines += [""]
        p = stats["pooled"]["D_H_state"]
        lines += [
            "## Pooled across 20 cells",
            f"   mean rel gap                  : {p['cell_mean_rel_gap_pct']:+.2f}%",
            f"   bootstrap 95% CI on mean rel gap: "
            f"[{p['cell_rel_gap_bootstrap_95CI_pct'][0]:+.2f}%, "
            f"{p['cell_rel_gap_bootstrap_95CI_pct'][1]:+.2f}%]",
            f"   ortho wins on mean            : {p['n_cells_ortho_wins_on_mean']}/{p['n_cells']}",
            f"   sign-test p (one-sided H1)    : {p['sign_test_p_one_sided']:.4f}",
            f"   Wilcoxon on cell gaps p (H1)  : {p['wilcoxon_cell_means_p_one_sided']:.4f}",
            "",
            "## Stratified mean rel gaps",
        ]
        for grp_key, grp_val in stats["stratified"]["D_H_state"].items():
            lines.append(f"   {grp_key:<22s} = {grp_val['cell_mean_rel_gap_pct']:+7.2f}%   "
                         f"({grp_val['n_cells_ortho_wins_on_mean']}/{grp_val['n_cells']} cells)")
        text_page(pdf, "Statistics: D_H_state (priority Hellinger metric)", lines, fontsize=8)

        # 9 - per-cell stats on PSE
        rows = stats["per_cell"]["PSE"]
        lines = [
            "# Per-cell paired statistics on PSE  (lower is better)",
            "",
            f"  {'M':>2} {'P':>2}  {'vanilla':>14}  {'ortho':>14}  {'rel_gap%':>9}  "
            f"{'d_z':>5}  {'95% t-CI on diff':>22}  {'wilc p':>6}  {'wins':>5}",
            "  " + "-" * 90,
        ]
        for r in rows:
            lines.append(
                f"  {r['M']:>2} {r['P']:>2}  "
                f"{r['vanilla_mean']:>10.4f}±{r['vanilla_std']:>4.3f}  "
                f"{r['ortho_mean']:>10.4f}±{r['ortho_std']:>4.3f}  "
                f"{r['rel_gap_pct']:>+8.1f}%  "
                f"{r['cohens_dz']:>+5.2f}  "
                f"[{r['t_ci_lo']:>+8.4g}, {r['t_ci_hi']:>+8.4g}]  "
                f"{r['wilcoxon_p_one_sided']:>6.3f}  "
                f"{int(r['ortho_win_rate']*r['n']):>1}/{r['n']:>1}")
        lines += [""]
        p = stats["pooled"]["PSE"]
        lines += [
            "## Pooled across 20 cells",
            f"   mean rel gap                  : {p['cell_mean_rel_gap_pct']:+.2f}%",
            f"   bootstrap 95% CI on mean rel gap: "
            f"[{p['cell_rel_gap_bootstrap_95CI_pct'][0]:+.2f}%, "
            f"{p['cell_rel_gap_bootstrap_95CI_pct'][1]:+.2f}%]",
            f"   ortho wins on mean            : {p['n_cells_ortho_wins_on_mean']}/{p['n_cells']}",
            f"   sign-test p (one-sided H1)    : {p['sign_test_p_one_sided']:.4f}",
            f"   Wilcoxon on cell gaps p (H1)  : {p['wilcoxon_cell_means_p_one_sided']:.4f}",
            "",
            "## Stratified mean rel gaps",
        ]
        for grp_key, grp_val in stats["stratified"]["PSE"].items():
            lines.append(f"   {grp_key:<22s} = {grp_val['cell_mean_rel_gap_pct']:+7.2f}%   "
                         f"({grp_val['n_cells_ortho_wins_on_mean']}/{grp_val['n_cells']} cells)")
        text_page(pdf, "Statistics: PSE (Hellinger of power spectrum)", lines, fontsize=8)

        # 10 - pooled & stratified table for all 4 metrics
        lines = [
            "Pooled and stratified results for each metric.",
            "",
            "Pooled across 20 cells (mean of cell-level relative gaps):",
            "",
            f"  {'metric':<18s} {'mean rel gap':>14s} {'95% bootstrap CI':>26s} {'wins':>6s} {'sign p':>7s} {'wilc p':>7s}",
            "  " + "-" * 86,
        ]
        for slug, label in [("D_H_state", "D_H_state"),
                            ("PSE",       "PSE"),
                            ("Dstsp",     "Dstsp (KL)"),
                            ("rmse_train_final", "rmse_train")]:
            p = stats["pooled"][slug]
            ci = p["cell_rel_gap_bootstrap_95CI_pct"]
            lines.append(
                f"  {label:<18s} {p['cell_mean_rel_gap_pct']:>+13.2f}% "
                f"[{ci[0]:>+7.2f}%, {ci[1]:>+7.2f}%]   "
                f"{p['n_cells_ortho_wins_on_mean']:>2d}/{p['n_cells']:>2d}  "
                f"{p['sign_test_p_one_sided']:>6.3f}  "
                f"{p['wilcoxon_cell_means_p_one_sided']:>6.3f}")
        lines += ["", "Stratified mean rel gaps (% on mean):", ""]
        groups = list(stats["stratified"]["D_H_state"].keys())
        header = "  " + " ".join([f"{g:>20s}" for g in groups])
        lines.append(f"  {'metric':<14s}" + header)
        lines.append("  " + "-" * (16 + 21 * len(groups)))
        for slug, label in [("D_H_state", "D_H_state"),
                            ("PSE",       "PSE"),
                            ("Dstsp",     "Dstsp"),
                            ("rmse_train_final", "rmse_train")]:
            cells = " ".join([f"{stats['stratified'][slug][g]['cell_mean_rel_gap_pct']:>+19.2f}%"
                              for g in groups])
            lines.append(f"  {label:<14s} {cells}")
        text_page(pdf, "Pooled & stratified results", lines, fontsize=7)

        # 11 - forest plots
        image_page(pdf, "stat_forest_D_H_state.png",
                   "Forest plot: per-cell gap on D_H_state (95% t-CI)")
        image_page(pdf, "stat_forest_PSE.png",
                   "Forest plot: per-cell gap on PSE (95% t-CI)")

        # 12 - effect size scatter
        image_page(pdf, "stat_effectsize_hellinger.png",
                   "Effect-size scatter: rel gap vs Cohen's d_z (Hellinger metrics)")

        # 13-19 - highlight trajectories
        # rank cells by D_H_state gain (most negative first)
        rows = stats["per_cell"]["D_H_state"]
        ranked = sorted(rows, key=lambda r: r["rel_gap_pct"])
        text_page(pdf, "Highlight trajectories",
                  [
                      "Each of the next pages shows one (M, P) cell at high",
                      "resolution: 3 columns of 3-D views (one per seed) plus 3",
                      "columns of x-z 2-D projections (one per seed).  Rows are",
                      "ground truth, vanilla AL-RNN, orthogonal AL-RNN.",
                      "",
                      "Ordered by D_H_state gain (largest ortho wins first):",
                      "",
                  ] + [
                      f"  M={r['M']:<3} P={r['P']:<3}  D_H_state gap = {r['rel_gap_pct']:+6.1f}%   "
                      f"PSE gap = {next(x['rel_gap_pct'] for x in stats['per_cell']['PSE'] if x['M']==r['M'] and x['P']==r['P']):+6.1f}%"
                      for r in ranked[:6]
                  ] + [
                      "",
                      "Plus one ortho-failure cell:",
                      f"  M={ranked[-1]['M']:<3} P={ranked[-1]['P']:<3}  D_H_state gap = "
                      f"{ranked[-1]['rel_gap_pct']:+6.1f}%   "
                      f"(largest ortho regression on D_H_state)",
                  ])
        for r in ranked[:6]:
            hires_image_page(pdf, f"cell_M{r['M']}_P{r['P']}.png",
                             f"M={r['M']} P={r['P']}: ortho wins by {r['rel_gap_pct']:+.1f}% on D_H_state")
        # one failure cell
        rfail = ranked[-1]
        hires_image_page(pdf, f"cell_M{rfail['M']}_P{rfail['P']}.png",
                         f"M={rfail['M']} P={rfail['P']}: ortho regression "
                         f"({rfail['rel_gap_pct']:+.1f}% on D_H_state)")

        # 20 - wall_all
        hires_image_page(pdf, "wall_all.png",
                         "Trajectory wall: all 20 cells, median-PSE seed, x-z projection")

    print(f"wrote {PDF}  ({os.path.getsize(PDF)/1024/1024:.1f} MB)")


if __name__ == "__main__":
    main()
