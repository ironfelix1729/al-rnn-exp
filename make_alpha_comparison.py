# -*- coding: utf-8 -*-
"""Overlay of alpha=1.0 vs alpha=0.5 M-sweep results.

Produces:
  results_p_sweep/alpha_compare_gap.png    # MSE gap ortho-vs-vanilla
  results_p_sweep/alpha_compare_losses.png # final-loss bars
  results_p_sweep/alpha_compare_report.txt # text table
"""
import os, json
import numpy as np
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_p_sweep")
M_LIST = [4, 5, 10]
P_LIST = [2, 3, 4]


def load(summary_path):
    with open(summary_path) as f:
        return json.load(f)


def fetch(alpha_suffix):
    out = {}
    for M in M_LIST:
        out[M] = load(os.path.join(OUT, f"summary_M{M}{alpha_suffix}.json"))
    return out


def main():
    a1 = fetch("")
    a5 = fetch("_a0p5")

    # text report
    lines = []
    add = lines.append
    add("=" * 96)
    add("M-sweep: alpha=1.0 vs alpha=0.5 (shared frozen B, 1500 epochs)")
    add("=" * 96)
    add("")
    hdr = f"{'M':>3} {'P':>3}  " \
          f"{'vanilla@1.0':>12} {'ortho@1.0':>12} {'gap@1.0':>8}   " \
          f"{'vanilla@0.5':>12} {'ortho@0.5':>12} {'gap@0.5':>8}"
    add(hdr)
    add("-" * len(hdr))
    for M in M_LIST:
        for P in P_LIST:
            pp1 = a1[M]["per_P"][str(P)]
            pp5 = a5[M]["per_P"][str(P)]
            g1 = 100 * (pp1["final_loss_ortho"] - pp1["final_loss_orig"]) / pp1["final_loss_orig"]
            g5 = 100 * (pp5["final_loss_ortho"] - pp5["final_loss_orig"]) / pp5["final_loss_orig"]
            add(f"{M:>3} {P:>3}  "
                f"{pp1['final_loss_orig']:>12.5f} {pp1['final_loss_ortho']:>12.5f} {g1:>+7.1f}%   "
                f"{pp5['final_loss_orig']:>12.5f} {pp5['final_loss_ortho']:>12.5f} {g5:>+7.1f}%")
        add("")
    rep = "\n".join(lines) + "\n"
    with open(os.path.join(OUT, "alpha_compare_report.txt"), "w") as f:
        f.write(rep)
    print(rep)

    # gap figure: two subplots, one per alpha, same axes
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    markers = {4: "o", 5: "s", 10: "^"}
    for ax, (label, src) in zip(axes, [(r"$\alpha = 1.0$", a1), (r"$\alpha = 0.5$", a5)]):
        for M in M_LIST:
            gaps = []
            for P in P_LIST:
                pp = src[M]["per_P"][str(P)]
                gaps.append(100 * (pp["final_loss_ortho"] - pp["final_loss_orig"]) / pp["final_loss_orig"])
            ax.plot(P_LIST, gaps, "-" + markers[M], label=f"M={M}", markersize=9)
        ax.axhline(0, color="k", lw=0.5, ls=":")
        ax.set_xlabel("P"); ax.set_title(label)
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("MSE gap ortho vs vanilla (%)   (negative = ortho wins)")
    axes[1].legend(fontsize=10)
    fig.suptitle("Orthogonal advantage across M, P and teacher-forcing strength alpha",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(OUT, "alpha_compare_gap.png"), dpi=140)
    plt.close(fig)

    # grouped-bar: absolute final losses at each (M, P, model, alpha)
    fig, axes = plt.subplots(1, len(M_LIST), figsize=(16, 5), sharey=False)
    width = 0.18
    x = np.arange(len(P_LIST))
    for ax, M in zip(axes, M_LIST):
        v1 = [a1[M]["per_P"][str(P)]["final_loss_orig"] for P in P_LIST]
        o1 = [a1[M]["per_P"][str(P)]["final_loss_ortho"] for P in P_LIST]
        v5 = [a5[M]["per_P"][str(P)]["final_loss_orig"] for P in P_LIST]
        o5 = [a5[M]["per_P"][str(P)]["final_loss_ortho"] for P in P_LIST]
        ax.bar(x - 1.5 * width, v1, width, label=r"vanilla $\alpha$=1.0", color="#e89a93")
        ax.bar(x - 0.5 * width, o1, width, label=r"ortho  $\alpha$=1.0", color="#8ea8cf")
        ax.bar(x + 0.5 * width, v5, width, label=r"vanilla $\alpha$=0.5", color="#c0392b")
        ax.bar(x + 1.5 * width, o5, width, label=r"ortho  $\alpha$=0.5", color="#2c5fb0")
        ax.set_xticks(x); ax.set_xticklabels([f"P={P}" for P in P_LIST])
        ax.set_title(f"M = {M}")
        ax.set_yscale("log")
        ax.grid(axis="y", which="both", alpha=0.3)
    axes[0].set_ylabel("final MSE (log)")
    axes[-1].legend(fontsize=8, loc="upper right")
    fig.suptitle("Final training MSE - alpha=1.0 vs alpha=0.5", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(OUT, "alpha_compare_losses.png"), dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    main()
