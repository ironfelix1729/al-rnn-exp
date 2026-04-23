# -*- coding: utf-8 -*-
"""Build figures and a PDF for the M-sweep (M=4,5,10; P=2,3,4).

Reads:
  results_p_sweep/summary_M{M}.json
  results_p_sweep/P{P}_M{M}_arrays.npz     (trajectories + losses)
  results_p_sweep/shared_B_M{M}.npy        (the shared, frozen embedding)
  results_p_sweep/Q_dataset_M{M}.npy       (the dataset rotation, same every run)

Writes into results_p_sweep/:
  Msweep_trajectories_2d.png      - grid: rows=(M,P) pairs, cols=(truth/vanilla/ortho)x3 projs
  Msweep_losses.png               - loss curves
  Msweep_gap.png                  - final-loss gap ortho-vs-vanilla across M and P
  Msweep_q_analysis.png           - Q diagnostics by M (for M=10 this is still 10x10)
  Msweep_report.txt               - tabular text summary
  Msweep_report.pdf               - one-PDF version of everything (title+tables+figures)
"""
import os, json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_p_sweep")
M_LIST = [4, 5, 10]
P_LIST = [2, 3, 4]
SUFFIX = ""  # set by main()


def _tag(M):
    return f"M{M}{SUFFIX}"


def load_M(M):
    with open(os.path.join(OUT, f"summary_{_tag(M)}.json")) as f:
        summary = json.load(f)
    data = {}
    for P in summary["P_list"]:
        arrs = np.load(os.path.join(OUT, f"P{P}_{_tag(M)}_arrays.npz"))
        data[P] = {k: arrs[k] for k in arrs.files}
    return summary, data


def plot_trajectories_2d(savepath, num_steps=5000):
    labels = ["x", "y", "z"]
    pairs = [(0, 1), (0, 2), (1, 2)]
    rows = len(M_LIST) * len(P_LIST)
    fig, axes = plt.subplots(rows, 9, figsize=(22, 2.5 * rows))
    r = 0
    for M in M_LIST:
        _, data = load_M(M)
        for P in P_LIST:
            d = data[P]
            srcs = [(d["X_test_trans"][:num_steps], "black", "truth"),
                    (d["orbit_orig"][:num_steps], "#c0392b", "vanilla"),
                    (d["orbit_ortho"][:num_steps], "#2c5fb0", "ortho")]
            col = 0
            for orb, color, name in srcs:
                for i, j in pairs:
                    ax = axes[r, col]
                    ax.plot(orb[:, i], orb[:, j], color=color, lw=0.5, alpha=0.9)
                    ax.set_title(f"M={M} P={P} {name} {labels[i]}-{labels[j]}", fontsize=8)
                    ax.set_aspect("equal", adjustable="datalim")
                    ax.tick_params(labelsize=6)
                    col += 1
            r += 1
    fig.suptitle("M-sweep: free-run Lorenz reconstruction - 2D projections", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(savepath, dpi=120)
    plt.close(fig)


def plot_trajectories_3d(savepath, num_steps=5000, elev=25, azim=-60):
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    rows = len(M_LIST) * len(P_LIST)
    fig = plt.figure(figsize=(11, 3.3 * rows))
    r = 0
    for M in M_LIST:
        _, data = load_M(M)
        for P in P_LIST:
            d = data[P]
            triples = [(d["X_test_trans"][:num_steps], "black", f"Ground truth M={M} P={P}"),
                       (d["orbit_orig"][:num_steps], "#c0392b", f"Vanilla M={M} P={P}"),
                       (d["orbit_ortho"][:num_steps], "#2c5fb0", f"Orthogonal M={M} P={P}")]
            for c, (orb, color, title) in enumerate(triples):
                ax = fig.add_subplot(rows, 3, r * 3 + c + 1, projection="3d")
                orb = np.asarray(orb)
                ax.plot(orb[:, 0], orb[:, 1], orb[:, 2], color=color, lw=0.6, alpha=0.95)
                ax.set_title(title, fontsize=10)
                ax.view_init(elev=elev, azim=azim)
                rng = float(max(np.ptp(orb[:, 0]), np.ptp(orb[:, 1]), np.ptp(orb[:, 2])))
                cx, cy, cz = orb[:, 0].mean(), orb[:, 1].mean(), orb[:, 2].mean()
                ax.set_xlim(cx - rng / 2 - .3, cx + rng / 2 + .3)
                ax.set_ylim(cy - rng / 2 - .3, cy + rng / 2 + .3)
                ax.set_zlim(cz - rng / 2 - .3, cz + rng / 2 + .3)
                ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
            r += 1
    fig.suptitle("M-sweep: free-run Lorenz reconstruction - 3D views", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(savepath, dpi=120)
    plt.close(fig)


def plot_losses(savepath):
    fig, axes = plt.subplots(1, len(M_LIST), figsize=(16, 4.5), sharey=True)
    for ax, M in zip(axes, M_LIST):
        summary, data = load_M(M)
        cmap = plt.cm.viridis(np.linspace(0.05, 0.85, len(P_LIST)))
        for c, P in zip(cmap, P_LIST):
            ax.plot(data[P]["losses_orig"], color=c, ls="--", alpha=0.85, label=f"vanilla P={P}")
            ax.plot(data[P]["losses_ortho"], color=c, ls="-", alpha=0.85, label=f"ortho  P={P}")
        ax.set_yscale("log")
        ax.set_title(f"M = {M}")
        ax.set_xlabel("epoch")
        ax.legend(fontsize=7, ncol=2)
    axes[0].set_ylabel("MSE (log)")
    fig.suptitle("Training loss curves, M-sweep", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(savepath, dpi=140)
    plt.close(fig)


def plot_gap(savepath):
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    markers = {4: "o", 5: "s", 10: "^"}
    for M in M_LIST:
        summary, _ = load_M(M)
        gaps = []
        for P in P_LIST:
            pp = summary["per_P"][str(P)]
            fo = pp["final_loss_orig"]
            fr = pp["final_loss_ortho"]
            gaps.append(100 * (fr - fo) / fo)
        ax.plot(P_LIST, gaps, "-" + markers[M], label=f"M={M}", markersize=10)
    ax.axhline(0, color="k", lw=0.5, ls=":")
    ax.set_xlabel("P (# ReLU units)")
    ax.set_ylabel("MSE gap ortho vs vanilla (%)   (negative = ortho wins)")
    ax.set_title("Orthogonal advantage grows as slack shrinks (smaller M)")
    ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(savepath, dpi=140)
    plt.close(fig)


def plot_q_analysis(savepath):
    fig, axes = plt.subplots(2, len(M_LIST), figsize=(16, 9))
    for col, M in enumerate(M_LIST):
        summary, _ = load_M(M)
        # eigenvalues on the unit circle
        ax = axes[0, col]
        theta = np.linspace(0, 2 * np.pi, 256)
        ax.plot(np.cos(theta), np.sin(theta), "k-", lw=0.5, alpha=0.5)
        cmap = plt.cm.viridis(np.linspace(0.05, 0.85, len(P_LIST)))
        for c, P in zip(cmap, P_LIST):
            angles = np.array(summary["per_P"][str(P)]["Q_analysis"]["rotation_angles_deg_sorted"])
            ax.scatter(np.cos(np.deg2rad(angles)), np.sin(np.deg2rad(angles)),
                       s=40, color=c, alpha=0.8, label=f"P={P}")
        ax.set_aspect("equal")
        ax.set_title(f"Q eigenvalues on unit circle  (M={M})")
        ax.set_xlabel("Re"); ax.set_ylabel("Im"); ax.legend(fontsize=8)

        # structural scalars
        ax = axes[1, col]
        keys = ["distance_from_identity_frob", "permutation_residual_frob",
                "orthogonality_error_frob", "singular_value_spread",
                "mean_row_participation"]
        Ps = np.array(P_LIST)
        for k in keys:
            ys = [summary["per_P"][str(P)]["Q_analysis"][k] for P in P_LIST]
            ax.plot(Ps, ys, "-o", label=k)
        ax.set_yscale("symlog", linthresh=1e-4)
        ax.set_xlabel("P"); ax.set_title(f"Q structure (M={M})")
        ax.legend(fontsize=7)
    fig.suptitle("Learned Q diagnostics across M", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(savepath, dpi=140)
    plt.close(fig)


def write_report(savepath):
    lines = []
    add = lines.append
    add("=" * 80)
    add("M-sweep: vanilla vs orthogonal AL-RNN, shared frozen B per M")
    add(f"  M  in {M_LIST}    P in {P_LIST}    epochs=1500    N=3 (Lorenz)")
    add("=" * 80)
    add("")
    for M in M_LIST:
        summary, _ = load_M(M)
        add(f"--- M = {M} ---")
        add(f"  {'P':>3}  {'vanilla':>10}  {'ortho':>10}  {'rel_gap':>10}   {'Q rot planes':>12}  {'||Q-I||_F':>10}  det  ")
        for P in P_LIST:
            pp = summary["per_P"][str(P)]
            fo = pp["final_loss_orig"]; fr = pp["final_loss_ortho"]
            gap = 100 * (fr - fo) / fo
            q = pp["Q_analysis"]
            add(f"  {P:>3}  {fo:>10.5f}  {fr:>10.5f}  {gap:>+9.1f}%   "
                f"{q['num_complex_eig_pairs']:>12d}  {q['distance_from_identity_frob']:>10.3f}  "
                f"{q['determinant']:>+.2f}")
        add("")

    add("Interpretation")
    add("--------------")
    add("With M=20 (lots of slack) the ortho edge is small (5-16 %).  Shrinking M to 4")
    add("removes the slack dimensions: every hidden dim matters for the dynamics, W's")
    add("gauge redundancy with Q shrinks, and the ortho edge grows sharply.")
    with open(savepath, "w") as f:
        f.write("\n".join(lines) + "\n")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--suffix", default="",
                    help="tag suffix like '_a0p5' (empty = alpha=1.0 run)")
    ap.add_argument("--label", default="Msweep",
                    help="output filename prefix (default Msweep)")
    args = ap.parse_args()
    global SUFFIX
    SUFFIX = args.suffix
    prefix = args.label

    have_all = all(os.path.exists(os.path.join(OUT, f"summary_M{M}{SUFFIX}.json")) for M in M_LIST)
    if not have_all:
        present = [M for M in M_LIST if os.path.exists(os.path.join(OUT, f"summary_M{M}{SUFFIX}.json"))]
        print(f"Only have summaries for M={present}; need all three before drawing figures.")
        return

    plot_trajectories_2d(os.path.join(OUT, f"{prefix}_trajectories_2d.png"))
    plot_losses(os.path.join(OUT, f"{prefix}_losses.png"))
    plot_gap(os.path.join(OUT, f"{prefix}_gap.png"))
    plot_q_analysis(os.path.join(OUT, f"{prefix}_q_analysis.png"))
    write_report(os.path.join(OUT, f"{prefix}_report.txt"))

    plot_trajectories_3d(os.path.join(OUT, f"{prefix}_trajectories_3d.png"))

    pdf_path = os.path.join(OUT, f"{prefix}_report.pdf")
    with PdfPages(pdf_path) as pdf:
        for png in [f"{prefix}_gap.png", f"{prefix}_losses.png",
                    f"{prefix}_q_analysis.png",
                    f"{prefix}_trajectories_3d.png",
                    f"{prefix}_trajectories_2d.png"]:
            img = plt.imread(os.path.join(OUT, png))
            fig = plt.figure(figsize=(8.5, 11))
            ax = fig.add_axes([0.02, 0.02, 0.96, 0.96])
            ax.imshow(img); ax.axis("off")
            pdf.savefig(fig); plt.close(fig)
    print(f"wrote {pdf_path}")


if __name__ == "__main__":
    main()
