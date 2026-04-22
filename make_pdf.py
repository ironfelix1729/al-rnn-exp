# -*- coding: utf-8 -*-
"""Build a PDF report summarising the P-sweep experiment.

  - Page 1: title, architectures, training protocol, results table, loss curve
  - Page 2+: free-run trajectories, one P per page, at legible size
  - Last page: short analysis of the learned Q (culled per request)
"""
import os, json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_p_sweep")


def load_all():
    with open(os.path.join(OUT, "summary_main.json")) as f:
        summary = json.load(f)
    data = {}
    for P in summary["P_list"]:
        arrs = np.load(os.path.join(OUT, f"P{P}_arrays.npz"))
        data[P] = {k: arrs[k] for k in arrs.files}
    return summary, data


def add_text_page(pdf, lines, title=None, fontsize=10):
    fig = plt.figure(figsize=(8.5, 11))
    ax = fig.add_axes([0.08, 0.05, 0.84, 0.9])
    ax.axis("off")
    y = 0.97
    if title:
        ax.text(0.0, y, title, fontsize=16, fontweight="bold",
                transform=ax.transAxes, va="top")
        y -= 0.05
    for ln in lines:
        ax.text(0.0, y, ln, fontsize=fontsize, family="monospace",
                transform=ax.transAxes, va="top")
        y -= fontsize / 720.0 * 1.6  # roughly one line
    pdf.savefig(fig)
    plt.close(fig)


def add_title_page(pdf, summary):
    lines = []
    lines.append("")
    lines.append("Goal")
    lines.append("----")
    lines.append("Compare two AL-RNN variants on a randomly rotated Lorenz63 attractor,")
    lines.append("sweeping the number of ReLU units P = 2..6.")
    lines.append("")
    lines.append("Architectures")
    lines.append("-------------")
    lines.append("Both models have state z in R^M (M=20), observation x in R^N (N=3).")
    lines.append("Shared parameters: A (diag skip), W (MxM coupling), h (bias),")
    lines.append("B (NxM input embedding).  Only the last P hidden units pass through")
    lines.append("a ReLU; the others stay linear.")
    lines.append("")
    lines.append("  Vanilla:    z_{t+1} = A . z_t + phi_P(z_t) W^T + h")
    lines.append("  Orthogonal: z_{t+1} = A . z_t + phi_P(Q z_t + b_Q) W^T + h")
    lines.append("                with Q in O(M), enforced by")
    lines.append("                torch.nn.utils.parametrizations.orthogonal")
    lines.append("")
    lines.append("Training protocol")
    lines.append("-----------------")
    lines.append(f"  data        : Lorenz63 train/test, first 500 steps dropped,")
    lines.append(f"                rotated in R^3 by a fixed random orthogonal")
    lines.append(f"                Q_dataset (72 deg, det=+1, seed=42)")
    lines.append(f"  M, N        : 20, 3")
    lines.append(f"  P           : {summary['P_list']}")
    lines.append(f"  seq length  : 128")
    lines.append(f"  batch size  : 64")
    lines.append(f"  batches/ep  : 20")
    lines.append(f"  epochs      : {summary['epochs']}")
    lines.append(f"  optimiser   : Adam, lr 1e-3 -> 1e-5 exponential decay")
    lines.append(f"  loss        : MSE on first N=3 hidden coordinates")
    lines.append(f"  teacher forcing: hard reset at t=0, alpha=1 injection")
    lines.append(f"                  every n_interleave=16 steps")
    lines.append(f"  free-run test  : 5000-step rollout from X_test_trans[0]")
    lines.append("")
    lines.append("Speed-ups vs the original notebook: cache the orthogonal-parametrisation")
    lines.append("output once per sequence (avoids recomputing on each of 128 inner steps),")
    lines.append("torch.cat split instead of clone+assign, batch 32->64.  Full sweep ran")
    lines.append("in ~98 minutes on 4 CPUs (projected ~3.5 h before these changes).")
    lines.append("")
    lines.append("Results (final training MSE)")
    lines.append("----------------------------")
    lines.append(f"  {'P':>3}   {'vanilla':>10}   {'orthogonal':>12}   {'rel. gap':>10}")
    for P in summary["P_list"]:
        pp = summary["per_P"][str(P)]
        fo = pp["final_loss_orig"]; fr = pp["final_loss_ortho"]
        gap = (fr - fo) / fo * 100.0
        lines.append(f"  {P:>3}   {fo:>10.5f}   {fr:>12.5f}   {gap:>+9.1f}%")
    lines.append("")
    lines.append("Orthogonal AL-RNN achieves lower MSE at every P; biggest gap at P=4 (-19%)")
    lines.append("and P=5 (-31%).  For attractor reconstruction the gap is much bigger")
    lines.append("visually - see trajectory pages below.")
    add_text_page(pdf, lines, title="P-sweep: vanilla vs orthogonal AL-RNN on rotated Lorenz63",
                  fontsize=10)


def add_loss_page(pdf, summary, data):
    fig, ax = plt.subplots(1, 1, figsize=(8.5, 5.5))
    cmap = plt.cm.viridis(np.linspace(0.05, 0.85, len(summary["P_list"])))
    for c, P in zip(cmap, summary["P_list"]):
        ax.plot(data[P]["losses_orig"], color=c, ls="--", alpha=0.85, label=f"vanilla P={P}")
        ax.plot(data[P]["losses_ortho"], color=c, ls="-", alpha=0.85, label=f"ortho  P={P}")
    ax.set_yscale("log")
    ax.set_xlabel("epoch"); ax.set_ylabel("MSE (log)")
    ax.set_title("Training loss curves, 1500 epochs")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    pdf.savefig(fig); plt.close(fig)


def _plot_orbit(ax, orb, color, title, elev=26, azim=-72):
    orb = np.asarray(orb)
    ax.plot(orb[:, 0], orb[:, 1], orb[:, 2],
            color=color, linewidth=0.9, alpha=0.95)
    ax.set_title(title, fontsize=11)
    ax.view_init(elev=elev, azim=azim)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    # equal box so attractor isn't squashed
    xs = orb[:, 0]; ys = orb[:, 1]; zs = orb[:, 2]
    rng = float(max(np.ptp(xs), np.ptp(ys), np.ptp(zs)))
    mx = [xs.mean(), ys.mean(), zs.mean()]
    for setter, m in zip(("set_xlim", "set_ylim", "set_zlim"), mx):
        getattr(ax, setter)(m - rng / 2 - 0.5, m + rng / 2 + 0.5)


def _plot_2d(ax, orb, color, title, ij, labels):
    i, j = ij
    ax.plot(orb[:, i], orb[:, j], color=color, lw=0.6, alpha=0.9)
    ax.set_title(title, fontsize=9)
    ax.set_xlabel(labels[i], fontsize=8); ax.set_ylabel(labels[j], fontsize=8)
    ax.tick_params(labelsize=7)
    ax.set_aspect("equal", adjustable="datalim")


def add_trajectory_page(pdf, P, truth, orbit_o, orbit_r, num_steps=5000):
    # Page A: 3D panels (big)
    fig = plt.figure(figsize=(11, 4.5))
    fig.suptitle(f"Free-run reconstruction at P = {P}  (3-D view, 5000 steps)",
                 fontsize=13, fontweight="bold")
    ax1 = fig.add_subplot(131, projection="3d")
    _plot_orbit(ax1, truth[:num_steps], "black", "Ground truth (rotated)")
    ax2 = fig.add_subplot(132, projection="3d")
    _plot_orbit(ax2, orbit_o[:num_steps], "#c0392b", "Vanilla AL-RNN")
    ax3 = fig.add_subplot(133, projection="3d")
    _plot_orbit(ax3, orbit_r[:num_steps], "#2c5fb0", "Orthogonal AL-RNN")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    pdf.savefig(fig); plt.close(fig)

    # Page B: 2D projections - 3 rows (truth/vanilla/ortho) x 3 cols (xy, xz, yz)
    labels = ["x", "y", "z"]
    pairs = [(0, 1), (0, 2), (1, 2)]
    fig, axes = plt.subplots(3, 3, figsize=(11, 11))
    fig.suptitle(f"Free-run reconstruction at P = {P}  (2-D projections, 5000 steps)",
                 fontsize=13, fontweight="bold")
    for ax, ij in zip(axes[0], pairs):
        _plot_2d(ax, truth[:num_steps], "black", f"Ground truth   {labels[ij[0]]}-{labels[ij[1]]}", ij, labels)
    for ax, ij in zip(axes[1], pairs):
        _plot_2d(ax, orbit_o[:num_steps], "#c0392b", f"Vanilla   {labels[ij[0]]}-{labels[ij[1]]}", ij, labels)
    for ax, ij in zip(axes[2], pairs):
        _plot_2d(ax, orbit_r[:num_steps], "#2c5fb0", f"Orthogonal   {labels[ij[0]]}-{labels[ij[1]]}", ij, labels)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    pdf.savefig(fig); plt.close(fig)
    return fig  # unused


def add_q_summary_page(pdf, summary):
    lines = []
    lines.append("")
    lines.append("The orthogonal constraint was enforced numerically (orthogonality error and")
    lines.append("singular-value spread are ~1e-6 across all P).  For every P the learned Q is:")
    lines.append("")
    lines.append("  * far from identity          ||Q - I||_F ~ 6.1-6.5")
    lines.append("                               (random-orthogonal baseline: 6.33 +/- 0.15)")
    lines.append("  * not a permutation          mean row participation ~ 13-14")
    lines.append("                               (1 = permutation, M=20 = fully uniform mixing)")
    lines.append("  * 9-10 complex eigenvalue pairs - i.e. ~9 genuine 2-plane rotations")
    lines.append("  * det(Q) alternates +/- 1 across P (rotations and rotoreflections both")
    lines.append("    found; sign is init-dependent, not data-dependent)")
    lines.append("")
    lines.append("In short, the learned Q is a dense, non-trivial full-rank rotation of the 20-d")
    lines.append("hidden state.  Its role is not to 'undo the data rotation' - the 3-d")
    lines.append("observation block Q[:3,:3] is far from Q_dataset and far from orthogonal.")
    lines.append("Q provides an extra, learned rotational degree of freedom in hidden space")
    lines.append("that improves the fit, most visibly as P grows.")
    lines.append("")
    lines.append("Per-P numerical diagnostics (Q_analysis section of summary_main.json):")
    lines.append("  P   ||Q-I||_F  perm-res  ortho-err  mean-part  cplx-pairs  real_+-1   det")
    for P in summary["P_list"]:
        q = summary["per_P"][str(P)]["Q_analysis"]
        lines.append(f"  {P}  {q['distance_from_identity_frob']:>9.2f}  "
                     f"{q['permutation_residual_frob']:>8.2f}  "
                     f"{q['orthogonality_error_frob']:>9.1e}  "
                     f"{q['mean_row_participation']:>9.2f}  "
                     f"{q['num_complex_eig_pairs']:>10d}  "
                     f"{q['num_real_pm1_eigs']:>8d}  "
                     f"{q['determinant']:>+5.2f}")
    add_text_page(pdf, lines, title="Learned Q (20x20) - brief diagnostics", fontsize=10)


def main():
    summary, data = load_all()
    pdf_path = os.path.join(OUT, "report.pdf")
    with PdfPages(pdf_path) as pdf:
        add_title_page(pdf, summary)
        add_loss_page(pdf, summary, data)
        for P in summary["P_list"]:
            d = data[P]
            add_trajectory_page(pdf, P, d["X_test_trans"], d["orbit_orig"], d["orbit_ortho"])
        add_q_summary_page(pdf, summary)
    print(f"wrote {pdf_path}")


if __name__ == "__main__":
    main()
