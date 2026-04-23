# -*- coding: utf-8 -*-
"""Comprehensive PDF summary: architectures, protocol, results, insights.

Renders results_p_sweep/final_report.pdf by stitching:
  - title + TL;DR  (text)
  - model architectures (text with math)
  - training protocol (text)
  - main results table (text)
  - alpha=1.0 vs alpha=0.5 gap plot
  - parameter-efficiency pareto
  - trajectories (α=1.0 and α=0.5)
  - matched-pair trajectories
  - learned Q diagnostics
  - insights / conclusion (text)
"""
import os, json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_p_sweep")
PDF = os.path.join(OUT, "final_report.pdf")


# --- page helpers ------------------------------------------------------------
def text_page(pdf, title, lines, fontsize=10):
    fig = plt.figure(figsize=(8.5, 11))
    ax = fig.add_axes([0.07, 0.04, 0.86, 0.92]); ax.axis("off")
    y = 0.97
    if title:
        ax.text(0.0, y, title, fontsize=15, fontweight="bold",
                transform=ax.transAxes, va="top")
        y -= 0.045
    for ln in lines:
        # headings: if line starts with '### '
        if ln.startswith("### "):
            ax.text(0.0, y, ln[4:], fontsize=11, fontweight="bold",
                    transform=ax.transAxes, va="top")
            y -= fontsize / 720.0 * 2.0
        else:
            ax.text(0.0, y, ln, fontsize=fontsize, family="monospace",
                    transform=ax.transAxes, va="top")
            y -= fontsize / 720.0 * 1.55
    pdf.savefig(fig); plt.close(fig)


def image_page(pdf, png_name, title=None):
    path = os.path.join(OUT, png_name)
    if not os.path.exists(path):
        return
    img = plt.imread(path)
    fig = plt.figure(figsize=(8.5, 11))
    if title:
        fig.suptitle(title, fontsize=13, fontweight="bold")
        ax = fig.add_axes([0.02, 0.02, 0.96, 0.93])
    else:
        ax = fig.add_axes([0.02, 0.02, 0.96, 0.96])
    ax.imshow(img); ax.axis("off")
    pdf.savefig(fig); plt.close(fig)


# --- data loading ------------------------------------------------------------
def load_all():
    data = {}
    for alpha, suf in [("1.0", ""), ("0.5", "_a0p5")]:
        data[alpha] = {}
        for M in [4, 5, 10]:
            p = os.path.join(OUT, f"summary_M{M}{suf}.json")
            if os.path.exists(p):
                data[alpha][M] = json.load(open(p))
        # M=20 shared-B (only alpha=1.0)
        if alpha == "1.0":
            p = os.path.join(OUT, "summary_sharedB.json")
            if os.path.exists(p):
                data[alpha][20] = json.load(open(p))
    return data


# --- assemble ---------------------------------------------------------------
def main():
    data = load_all()

    with PdfPages(PDF) as pdf:
        # Page 1 - title + TL;DR
        text_page(pdf, "Orthogonal AL-RNN on Lorenz63: summary of experiments",
                  [
                      "A comparison of two Almost-Linear RNN variants on a randomly rotated",
                      "Lorenz63 attractor.  The ortho variant adds a learned orthogonal",
                      "transform Q before the ReLU branch; we sweep the hidden width M,",
                      "the number of ReLU units P, and the teacher-forcing blend alpha.",
                      "",
                      "### TL;DR",
                      "* Ortho matches vanilla when M is much larger than the effective",
                      "  dim of the dynamics (M=20, 3-D Lorenz): large gauge redundancy",
                      "  with W absorbs Q's extra capacity.",
                      "* Shrinking M removes that slack and the ortho edge explodes:",
                      "    - M=10: ortho typically 10-30 % lower MSE",
                      "    - M=5 : ortho typically 20-75 % lower MSE",
                      "    - M=4 : ortho typically 50-73 % lower MSE",
                      "* Under weaker teacher forcing (alpha=0.5) the gap is if anything",
                      "  larger and more monotone in P.",
                      "* Parameter efficiency: ortho M=5 matches vanilla M=20 at P=3",
                      "  using 9x fewer trainable parameters.",
                      "* Learned Q is numerically exactly orthogonal, far from identity,",
                      "  with ~M/2 genuine 2-plane rotations - non-trivial expressivity,",
                      "  not a gauge artefact.",
                  ])

        # Page 2 - Architectures
        text_page(pdf, "Model architectures",
                  [
                      "Both models share A (diag skip), W (MxM coupling), h (bias) and",
                      "a frozen input embedding B (3 x M) used only to initialise the",
                      "hidden state.  The ReLU is applied only to the LAST P coordinates;",
                      "the first M-P coordinates are linear.",
                      "",
                      "### Vanilla AL-RNN",
                      "   z_{t+1} = A . z_t + phi_P(z_t) W^T + h",
                      "   phi_P(z)[:M-P] = z[:M-P]",
                      "   phi_P(z)[M-P:] = ReLU(z[M-P:])",
                      "",
                      "### Orthogonal AL-RNN",
                      "   z_{t+1} = A . z_t + phi_P(Q z_t + b_Q) W^T + h",
                      "   with Q in O(M) enforced via",
                      "        torch.nn.utils.parametrizations.orthogonal",
                      "",
                      "Trainable-parameter counts (B is frozen in the M-sweeps):",
                      "",
                      "                  M=4     M=5     M=10     M=20",
                      "   vanilla         24      35      120      440",
                      "   ortho           34      50      175      650",
                      "",
                      "Teacher forcing:  hard reset z[:N] := x at t=0; every n_interleave",
                      "steps inject z[:N] := (1-alpha) z[:N] + alpha x.",
                      "Loss:  MSE on the first N=3 hidden coords over T=128 steps.",
                  ])

        # Page 3 - Protocol
        text_page(pdf, "Experimental protocol",
                  [
                      "Data        : Lorenz63 train/test, first 500 steps dropped.",
                      "              Fixed 3x3 random orthogonal Q_dataset (72 deg,",
                      "              det=+1, seed 42) rotates the observations in R^3.",
                      "Seed        : 42 (fixed across vanilla/ortho and across P, so",
                      "              A, W, h, B, Q_dataset are all reproducible).",
                      "Shared B    : one frozen B of shape (3, M) per M, identical",
                      "              across vanilla and ortho runs at that M.",
                      "Optimiser   : Adam, lr 1e-3 -> 1e-5 exponentially decayed.",
                      "Epochs      : 1500 per run; 20 batches/epoch; batch 64.",
                      "Seq length  : 128; teacher-forcing period n_interleave = 16.",
                      "Alpha       : two values sweeped - 1.0 (hard reset) and 0.5 (blend).",
                      "Sweeps      : M in {4, 5, 10} x P in {2, 3, 4}  for both alpha.",
                      "              Plus an earlier M=20 shared-B run at alpha=1.0.",
                      "Free-run    : after training, 5000-step rollout from",
                      "              X_test_trans[0] with no teacher forcing.",
                      "Total compute: ~6 hours across 54 training runs on 4 CPU cores.",
                  ])

        # Page 4 - main results table
        tbl = []
        tbl.append(f"{'alpha':>6} {'M':>3} {'P':>3}  {'vanilla':>10}  {'ortho':>10}  "
                   f"{'rel gap':>8}  {'van. params':>11}  {'ortho params':>12}")
        tbl.append("-" * 80)
        for alpha in ["1.0", "0.5"]:
            for M in sorted(data[alpha].keys()):
                s = data[alpha][M]
                for P in [2, 3, 4]:
                    if str(P) not in s["per_P"]:
                        continue
                    pp = s["per_P"][str(P)]
                    fo = pp["final_loss_orig"]; fr = pp["final_loss_ortho"]
                    gap = 100 * (fr - fo) / fo
                    vp = M * M + 2 * M
                    op = M * M + 3 * M + M * (M - 1) // 2
                    tbl.append(f"{alpha:>6} {M:>3} {P:>3}  {fo:>10.5f}  {fr:>10.5f}  "
                               f"{gap:>+7.1f}%  {vp:>11d}  {op:>12d}")
                tbl.append("")
        text_page(pdf, "Main results (final training MSE)", tbl, fontsize=9)

        # Page 5 - Gap comparison
        image_page(pdf, "alpha_compare_gap.png", "Ortho-vs-vanilla MSE gap across M and alpha")

        # Page 6 - losses side by side
        image_page(pdf, "alpha_compare_losses.png", "Absolute final losses (bars)")

        # Page 7 - parameter efficiency
        image_page(pdf, "paramtradeoff_pareto.png", "Parameter efficiency (MSE vs trainable params)")

        # Page 8 - matched-pair trajectories
        image_page(pdf, "paramtradeoff_trajectories_2d.png",
                   "Matched pairs: ortho at small M vs vanilla at larger M")

        # Page 9 - trajectories alpha=1.0
        image_page(pdf, "Msweep_trajectories_2d.png",
                   "Free-run trajectories (2-D projections), alpha=1.0")
        image_page(pdf, "Msweep_trajectories_3d.png",
                   "Free-run trajectories (3-D), alpha=1.0")

        # Page 11 - trajectories alpha=0.5
        image_page(pdf, "Msweep_a0p5_trajectories_2d.png",
                   "Free-run trajectories (2-D projections), alpha=0.5")
        image_page(pdf, "Msweep_a0p5_trajectories_3d.png",
                   "Free-run trajectories (3-D), alpha=0.5")

        # Page 13 - Q analysis
        image_page(pdf, "Msweep_q_analysis.png", "Learned Q diagnostics across M, alpha=1.0")
        image_page(pdf, "Msweep_a0p5_q_analysis.png", "Learned Q diagnostics across M, alpha=0.5")

        # Page 15 - insights
        text_page(pdf, "Performance insights",
                  [
                      "### Why does ortho help more when M shrinks?",
                      "",
                      "With phi_P fully linear, the ortho step reduces to",
                      "   z_{t+1} = A . z_t + z_t (W Q^T)^T + (const)",
                      "so Q is absorbed into W - pure gauge.  The non-gauge part of Q",
                      "is the choice of WHICH P-dim subspace of z is fed through ReLU.",
                      "With M >> N the 'right' subspace is easy to find among many, and",
                      "any useful rotation W Q^T can be matched by rotating W alone.",
                      "As M -> N every hidden dim matters, the gauge freedom shrinks,",
                      "and Q's extra expressivity becomes decisive.",
                      "",
                      "### Why is alpha=0.5 harder?",
                      "",
                      "With alpha=1.0 the first 3 coords are reset every 16 steps, so",
                      "the model only needs to predict well within a 16-step window.",
                      "alpha=0.5 blends observation with the model's own state - the",
                      "model must produce outputs that stay self-consistent over",
                      "longer effective horizons.  The ortho gap grows and becomes",
                      "uniformly negative under this stricter test.",
                      "",
                      "### Does the learned Q rotate away the dataset rotation?",
                      "",
                      "No.  Q lives in R^M while Q_dataset acts in R^3.  The 3x3",
                      "observation-subspace block Q[:N, :N] is far from Q_dataset,",
                      "from Q_dataset^T, and from I; its Frobenius norm is ~0.5-0.8",
                      "(vs sqrt(3)=1.73 for a pure block-diagonal Q).  Most of Q's",
                      "mass sits in the off-diagonal 3x(M-3) block - Q mixes the",
                      "(teacher-forced) observation coordinates into the latent",
                      "coordinates rather than trying to 'undo' the data rotation.",
                      "",
                      "### Parameter efficiency",
                      "",
                      "At alpha=1.0 P=3, ortho M=5 (50 trainable params) matches",
                      "vanilla M=20 (440 params) within 3 % MSE - 9x fewer params.",
                      "At alpha=1.0 P=4, ortho M=10 matches vanilla M=20 at 0.40x",
                      "params.  On a Pareto sense ortho lies strictly below vanilla",
                      "in (MSE vs params) space at every parameter band we tested.",
                      "",
                      "### Practical take-away",
                      "",
                      "If you care about getting the right attractor shape with few",
                      "hidden units, add an orthogonal reparameterisation of the ReLU",
                      "input.  It will not hurt at high M and dramatically helps at",
                      "low M - exactly the regime where AL-RNN parsimony matters.",
                  ])

    print(f"wrote {PDF}")


if __name__ == "__main__":
    main()
