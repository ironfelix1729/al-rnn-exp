# -*- coding: utf-8 -*-
"""Render figures & analysis from results_p_sweep/ artefacts.

Produces:
  results_p_sweep/trajectories.png     (5xP rows, 3 cols: truth / vanilla / ortho)
  results_p_sweep/losses.png           (loss curves per P)
  results_p_sweep/q_analysis.png       (Q spectral & structure diagnostics)
  results_p_sweep/weight_norms.png     (Frobenius norms per layer)
  results_p_sweep/report.txt           (prose summary)
"""
import os, json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_p_sweep")


def load_all(tag="main"):
    with open(os.path.join(OUT, f"summary_{tag}.json")) as f:
        summary = json.load(f)
    data = {}
    for P in summary["P_list"]:
        arrs = np.load(os.path.join(OUT, f"P{P}_arrays.npz"))
        data[P] = {k: arrs[k] for k in arrs.files}
    return summary, data


def plot_trajectories(summary, data, savepath, num_steps=5000):
    P_list = summary["P_list"]
    n_rows = len(P_list)
    # generous per-panel size + consistent viewing angle + equal 3-D box
    fig = plt.figure(figsize=(16, 5.2 * n_rows))
    elev, azim = 26, -72

    def _draw(ax, orb, color, title):
        orb = np.asarray(orb)
        ax.plot(orb[:, 0], orb[:, 1], orb[:, 2], color=color, lw=0.9, alpha=0.95)
        ax.set_title(title, fontsize=13)
        ax.view_init(elev=elev, azim=azim)
        rng = float(max(np.ptp(orb[:, 0]), np.ptp(orb[:, 1]), np.ptp(orb[:, 2])))
        cx, cy, cz = orb[:, 0].mean(), orb[:, 1].mean(), orb[:, 2].mean()
        ax.set_xlim(cx - rng / 2 - 0.5, cx + rng / 2 + 0.5)
        ax.set_ylim(cy - rng / 2 - 0.5, cy + rng / 2 + 0.5)
        ax.set_zlim(cz - rng / 2 - 0.5, cz + rng / 2 + 0.5)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])

    for i, P in enumerate(P_list):
        d = data[P]
        truth = d["X_test_trans"][:num_steps]
        orbit_o = d["orbit_orig"][:num_steps]
        orbit_r = d["orbit_ortho"][:num_steps]
        _draw(fig.add_subplot(n_rows, 3, 3 * i + 1, projection="3d"),
              truth, "black", f"Ground truth  (P={P})")
        _draw(fig.add_subplot(n_rows, 3, 3 * i + 2, projection="3d"),
              orbit_o, "#c0392b", f"Vanilla AL-RNN  (P={P})")
        _draw(fig.add_subplot(n_rows, 3, 3 * i + 3, projection="3d"),
              orbit_r, "#2c5fb0", f"Orthogonal AL-RNN  (P={P})")

    fig.suptitle("Free-run Lorenz reconstruction: truth vs vanilla vs orthogonal, P sweep",
                 fontsize=16, fontweight="bold")
    fig.tight_layout()
    fig.savefig(savepath, dpi=150)
    plt.close(fig)


def plot_losses(summary, data, savepath):
    P_list = summary["P_list"]
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    cmap = plt.cm.viridis(np.linspace(0.05, 0.85, len(P_list)))
    for c, P in zip(cmap, P_list):
        ax.plot(data[P]["losses_orig"], color=c, ls="--", alpha=0.8, label=f"vanilla P={P}")
        ax.plot(data[P]["losses_ortho"], color=c, ls="-", alpha=0.8, label=f"ortho  P={P}")
    ax.set_yscale("log")
    ax.set_xlabel("epoch")
    ax.set_ylabel("MSE (log)")
    ax.set_title("Training loss by P")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(savepath, dpi=140)
    plt.close(fig)


def plot_q_analysis(summary, savepath):
    P_list = summary["P_list"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    # (a) eigenvalues of Q on the unit circle
    ax = axes[0, 0]
    theta = np.linspace(0, 2 * np.pi, 256)
    ax.plot(np.cos(theta), np.sin(theta), "k-", lw=0.5, alpha=0.5)
    cmap = plt.cm.viridis(np.linspace(0.05, 0.85, len(P_list)))
    for c, P in zip(cmap, P_list):
        angles = np.array(summary["per_P"][str(P)]["Q_analysis"]["rotation_angles_deg_sorted"])
        ax.scatter(np.cos(np.deg2rad(angles)), np.sin(np.deg2rad(angles)),
                   s=40, color=c, alpha=0.7, label=f"P={P}")
    ax.set_aspect("equal")
    ax.set_title("Eigenvalues of learned Q on the unit circle")
    ax.set_xlabel("Re"); ax.set_ylabel("Im"); ax.legend(fontsize=8)

    # (b) rotation angles histogram
    ax = axes[0, 1]
    for c, P in zip(cmap, P_list):
        angles = np.array(summary["per_P"][str(P)]["Q_analysis"]["rotation_angles_deg_sorted"])
        ax.plot(np.sort(angles), np.arange(len(angles)) / len(angles),
                color=c, label=f"P={P}")
    ax.set_xlabel("rotation angle of eigenvalues (deg)")
    ax.set_ylabel("cumulative fraction")
    ax.set_title("CDF of Q-eigenvalue angles")
    ax.legend(fontsize=8)

    # (c) structural scalars
    keys = ["distance_from_identity_frob", "permutation_residual_frob",
            "orthogonality_error_frob", "singular_value_spread",
            "mean_row_participation"]
    Ps = np.array(P_list)
    ax = axes[1, 0]
    for k in keys:
        ys = [summary["per_P"][str(P)]["Q_analysis"][k] for P in P_list]
        ax.plot(Ps, ys, "-o", label=k)
    ax.set_yscale("symlog", linthresh=1e-4)
    ax.set_xlabel("P")
    ax.set_title("Structural diagnostics of Q (lower = closer to I / perm)")
    ax.legend(fontsize=8)

    # (d) weight Frobenius norms (vanilla vs ortho)
    ax = axes[1, 1]
    for name in ["W", "A", "h", "B"]:
        y_o = [summary["per_P"][str(P)]["weight_summary_orig"][name]["frob"] for P in P_list]
        y_r = [summary["per_P"][str(P)]["weight_summary_ortho"][name]["frob"] for P in P_list]
        (line,) = ax.plot(Ps, y_o, "--o", label=f"vanilla {name}")
        ax.plot(Ps, y_r, "-s", color=line.get_color(), label=f"ortho {name}")
    ax.set_xlabel("P")
    ax.set_ylabel("Frobenius norm")
    ax.set_title("Weight norms across models")
    ax.legend(fontsize=8, ncol=2)

    fig.tight_layout()
    fig.savefig(savepath, dpi=140)
    plt.close(fig)


def random_ortho_baseline(M, n=200, seed=123):
    """Sample random orthogonal matrices to get baselines for diagnostics."""
    rng = np.random.default_rng(seed)
    dists, perm_res, mean_part = [], [], []
    I = np.eye(M)
    for _ in range(n):
        A = rng.standard_normal((M, M))
        Q, R = np.linalg.qr(A)
        signs = np.sign(np.diag(R)); signs[signs == 0] = 1
        Q = Q * signs
        dists.append(np.linalg.norm(Q - I))
        absQ = np.abs(Q)
        row_max = absQ.max(axis=1)
        perm_res.append(np.linalg.norm(absQ - np.diag(row_max) @ (absQ == row_max[:, None]).astype(float)))
        row_l2 = np.sqrt((Q ** 2).sum(axis=1)); row_l1 = np.abs(Q).sum(axis=1)
        mean_part.append(((row_l1 / row_l2) ** 2).mean())
    return {
        "dist_from_I_mean": float(np.mean(dists)),
        "dist_from_I_std": float(np.std(dists)),
        "perm_residual_mean": float(np.mean(perm_res)),
        "mean_participation_mean": float(np.mean(mean_part)),
        "mean_participation_std": float(np.std(mean_part)),
    }


def write_report(summary, data, savepath):
    P_list = summary["P_list"]
    M = summary["M"]
    baseline = random_ortho_baseline(M)
    lines = []
    add = lines.append
    add("=" * 72)
    add("P-sweep: vanilla vs orthogonal AL-RNN on rotated Lorenz63")
    add(f"epochs={summary['epochs']}  M={M}  P values={P_list}")
    add("=" * 72)
    add("")
    add(f"Random-{M}x{M}-orthogonal baseline (mean over 200 samples):")
    add(f"  ||Q - I||_F          : {baseline['dist_from_I_mean']:.2f} +- {baseline['dist_from_I_std']:.2f}")
    add(f"  perm residual        : {baseline['perm_residual_mean']:.2f}")
    add(f"  mean row participation: {baseline['mean_participation_mean']:.2f} "
        f"+- {baseline['mean_participation_std']:.2f}   (1=perm, M={M}=uniform)")
    add("")
    add("-" * 72)
    add(f"{'P':>3}  {'final_loss':>20}  {'min_loss':>20}  {'train_s':>16}")
    add(f"{'':>3}  {'orig':>9}/{'ortho':>9}  {'orig':>9}/{'ortho':>9}  {'orig':>7}/{'ortho':>7}")
    add("-" * 72)
    for P in P_list:
        pp = summary["per_P"][str(P)]
        add(f"{P:>3}  {pp['final_loss_orig']:>9.5f}/{pp['final_loss_ortho']:>9.5f}  "
            f"{pp['min_loss_orig']:>9.5f}/{pp['min_loss_ortho']:>9.5f}  "
            f"{pp['train_seconds_orig']:>7.1f}/{pp['train_seconds_ortho']:>7.1f}")

    add("")
    add("Q (learned orthogonal transform) structural diagnostics by P:")
    add("-" * 72)
    hdr = f"{'P':>3}  {'||Q-I||_F':>10}  {'perm_res':>9}  {'ortho_err':>10}  " \
          f"{'sv_spread':>9}  {'mean_part':>9}  {'complex_pairs':>13}  {'real_+-1':>8}  {'det':>6}"
    add(hdr)
    for P in P_list:
        q = summary["per_P"][str(P)]["Q_analysis"]
        add(f"{P:>3}  {q['distance_from_identity_frob']:>10.3f}  "
            f"{q['permutation_residual_frob']:>9.3f}  "
            f"{q['orthogonality_error_frob']:>10.2e}  "
            f"{q['singular_value_spread']:>9.2e}  "
            f"{q['mean_row_participation']:>9.3f}  "
            f"{q['num_complex_eig_pairs']:>13d}  "
            f"{q['num_real_pm1_eigs']:>8d}  "
            f"{q['determinant']:>6.2f}")

    add("")
    add("Interpretation key:")
    add("  ortho_err ~ 0             -> Q satisfies the orthogonality constraint")
    add("  det = +/- 1               -> rotation (det=+1) or rotoreflection (det=-1)")
    add("  ||Q - I||_F near baseline -> Q is a nontrivial transform, not the identity")
    add("  mean_part >> 1            -> Q mixes many coordinates (not a permutation)")
    add("  complex eigenpairs        -> genuine 2-plane rotations in Q")
    add("")
    add("Conclusion (auto):")
    for P in P_list:
        q = summary["per_P"][str(P)]["Q_analysis"]
        close_to_I = q["distance_from_identity_frob"] < 0.2
        close_to_perm = q["permutation_residual_frob"] < 0.3
        mix = q["mean_row_participation"]
        verdict = []
        if close_to_I: verdict.append("near-identity")
        if close_to_perm: verdict.append("permutation-like")
        if mix > 2 and not close_to_perm: verdict.append(f"mixes {mix:.1f} coords/row")
        if q["num_complex_eig_pairs"] > 0:
            verdict.append(f"{q['num_complex_eig_pairs']} genuine rotation plane(s)")
        if not verdict: verdict.append("nontrivial")
        add(f"  P={P}: " + "; ".join(verdict))

    with open(savepath, "w") as f:
        f.write("\n".join(lines))
    print("\n".join(lines))


def plot_trajectories_2d(summary, data, savepath, num_steps=5000):
    """2D projections per P, much more legible than a single 3D angle."""
    P_list = summary["P_list"]
    n_rows = len(P_list)
    labels = ["x", "y", "z"]
    pairs = [(0, 1), (0, 2), (1, 2)]
    # per-P: 3 panels wide (truth/vanilla/ortho) x 3 projections = 9 panels per P
    fig, axes = plt.subplots(n_rows, 9, figsize=(22, 2.6 * n_rows))
    for r, P in enumerate(P_list):
        d = data[P]
        srcs = [(d["X_test_trans"][:num_steps], "black", "truth"),
                (d["orbit_orig"][:num_steps], "#c0392b", "vanilla"),
                (d["orbit_ortho"][:num_steps], "#2c5fb0", "ortho")]
        col = 0
        for orb, color, name in srcs:
            for i, j in pairs:
                ax = axes[r, col]
                ax.plot(orb[:, i], orb[:, j], color=color, lw=0.5, alpha=0.9)
                ax.set_title(f"P={P}  {name}  {labels[i]}-{labels[j]}", fontsize=9)
                ax.set_aspect("equal", adjustable="datalim")
                ax.tick_params(labelsize=7)
                col += 1
    fig.suptitle("Free-run Lorenz reconstruction - 2D projections", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(savepath, dpi=130)
    plt.close(fig)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="main")
    args = ap.parse_args()
    tag = args.tag
    suffix = "" if tag == "main" else f"_{tag}"
    summary, data = load_all(tag)
    plot_trajectories(summary, data, os.path.join(OUT, f"trajectories{suffix}.png"))
    plot_trajectories_2d(summary, data, os.path.join(OUT, f"trajectories_2d{suffix}.png"))
    plot_losses(summary, data, os.path.join(OUT, f"losses{suffix}.png"))
    plot_q_analysis(summary, os.path.join(OUT, f"q_analysis{suffix}.png"))
    write_report(summary, data, os.path.join(OUT, f"report{suffix}.txt"))
    print("\nFigures written to", OUT)


if __name__ == "__main__":
    main()
