# -*- coding: utf-8 -*-
"""Does ortho at small M match vanilla at big M?

  * Table: loss vs learnable-parameter count across (family, M, P, alpha).
  * Pareto plot: MSE vs params for both families.
  * Side-by-side 2-D trajectory comparison of matched (ortho_small, vanilla_big) pairs.

Effective trainable params (shared frozen B, frozen only in M-sweeps):
  A (M diag) + W (M^2) + h (M)                           # vanilla: M^2 + 2M
  plus Q (M(M-1)/2 effective) + b_Q (M)                  # ortho:  M^2 + 3M + M(M-1)/2
Input embedding B is frozen -> contributes 0 trainable params here.
"""
import os, json, math
import numpy as np
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_p_sweep")

# (alpha_label, path_suffix, M_list)
SWEEPS = {
    "alpha=1.0": ("", [4, 5, 10]),
    "alpha=0.5": ("_a0p5", [4, 5, 10]),
}
# M=20 run was tagged "sharedB" and ran for P=2..6; keep only P=2..4 for alignment
EXTRA = {"alpha=1.0": ("sharedB", 20, "summary_sharedB.json")}

P_LIST = [2, 3, 4]


def params_vanilla(M):
    return M * M + 2 * M


def params_ortho(M):
    return M * M + 3 * M + M * (M - 1) // 2


def gather():
    rows = []
    for alpha_label, (suffix, Ms) in SWEEPS.items():
        for M in Ms:
            path = os.path.join(OUT, f"summary_M{M}{suffix}.json")
            if not os.path.exists(path):
                continue
            with open(path) as f:
                s = json.load(f)
            for P in P_LIST:
                if str(P) not in s["per_P"]:
                    continue
                pp = s["per_P"][str(P)]
                rows.append({"alpha": alpha_label, "suffix": suffix, "M": M, "P": P,
                             "family": "vanilla", "mse": pp["final_loss_orig"],
                             "params": params_vanilla(M),
                             "arrays_path": os.path.join(OUT, f"P{P}_M{M}{suffix}_arrays.npz")})
                rows.append({"alpha": alpha_label, "suffix": suffix, "M": M, "P": P,
                             "family": "ortho", "mse": pp["final_loss_ortho"],
                             "params": params_ortho(M),
                             "arrays_path": os.path.join(OUT, f"P{P}_M{M}{suffix}_arrays.npz")})

    # extra M=20 shared-B (alpha=1.0 only)
    for alpha_label, (tag, M, fname) in EXTRA.items():
        path = os.path.join(OUT, fname)
        if not os.path.exists(path):
            continue
        with open(path) as f:
            s = json.load(f)
        for P in P_LIST:
            if str(P) not in s["per_P"]:
                continue
            pp = s["per_P"][str(P)]
            rows.append({"alpha": alpha_label, "suffix": "", "M": M, "P": P,
                         "family": "vanilla", "mse": pp["final_loss_orig"],
                         "params": params_vanilla(M),
                         "arrays_path": None})  # M=20 arrays may have been overwritten
            rows.append({"alpha": alpha_label, "suffix": "", "M": M, "P": P,
                         "family": "ortho", "mse": pp["final_loss_ortho"],
                         "params": params_ortho(M),
                         "arrays_path": None})
    return rows


def find_equivalent_pairs(rows, tol_rel=0.15):
    """For each ortho row at some M, find a vanilla row at a STRICTLY larger M whose
    MSE is within tol_rel of this ortho's MSE.  Keep the one with fewest params."""
    matches = []
    for r in rows:
        if r["family"] != "ortho":
            continue
        cands = [v for v in rows
                 if v["family"] == "vanilla" and v["alpha"] == r["alpha"]
                 and v["M"] > r["M"]
                 and abs(v["mse"] - r["mse"]) / r["mse"] <= tol_rel]
        if not cands:
            continue
        best = min(cands, key=lambda v: v["params"])
        matches.append({"ortho": r, "vanilla": best,
                        "param_ratio": r["params"] / best["params"],
                        "mse_ratio": r["mse"] / best["mse"]})
    return matches


def render_table(rows, matches):
    lines = []
    add = lines.append
    add("=" * 110)
    add("Final MSE and trainable-parameter count across sweeps")
    add("(B is frozen in all M-sweeps: A + W + h  for vanilla; + Q (orthog) + b_Q for ortho)")
    add("=" * 110)
    add(f"{'alpha':>9}  {'family':>8}  {'M':>4}  {'P':>3}  {'mse':>10}  {'params':>8}")
    add("-" * 110)
    for r in sorted(rows, key=lambda r: (r["alpha"], r["family"], r["M"], r["P"])):
        add(f"{r['alpha']:>9}  {r['family']:>8}  {r['M']:>4}  {r['P']:>3}  "
            f"{r['mse']:>10.5f}  {r['params']:>8d}")
    add("")
    add("=" * 110)
    add(f"Matched pairs: ortho (small M) ~ vanilla (bigger M) within 15 % MSE")
    add("=" * 110)
    add(f"{'alpha':>9}  {'P':>3}  "
        f"{'ortho M':>7} {'ortho mse':>10} {'ortho params':>12}   "
        f"{'vanilla M':>9} {'vanilla mse':>12} {'vanilla params':>14}   "
        f"{'param ratio':>12}  {'mse ratio':>10}")
    add("-" * 110)
    for m in sorted(matches, key=lambda x: (x["ortho"]["alpha"], x["ortho"]["P"], x["ortho"]["M"])):
        add(f"{m['ortho']['alpha']:>9}  {m['ortho']['P']:>3}  "
            f"{m['ortho']['M']:>7} {m['ortho']['mse']:>10.5f} {m['ortho']['params']:>12d}   "
            f"{m['vanilla']['M']:>9} {m['vanilla']['mse']:>12.5f} {m['vanilla']['params']:>14d}   "
            f"{m['param_ratio']:>11.2f}x  {m['mse_ratio']:>9.2f}x")
    report = "\n".join(lines) + "\n"
    print(report)
    with open(os.path.join(OUT, "paramtradeoff_report.txt"), "w") as f:
        f.write(report)


def render_pareto(rows):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)
    for ax, alpha in zip(axes, ["alpha=1.0", "alpha=0.5"]):
        sub = [r for r in rows if r["alpha"] == alpha]
        # colors: family; marker: P
        p_marker = {2: "o", 3: "s", 4: "^"}
        fam_color = {"vanilla": "#c0392b", "ortho": "#2c5fb0"}
        for r in sub:
            ax.scatter(r["params"], r["mse"],
                       marker=p_marker[r["P"]], color=fam_color[r["family"]],
                       s=110, alpha=0.85, edgecolors="black", linewidth=0.5)
            ax.annotate(f"M={r['M']}", (r["params"], r["mse"]),
                        xytext=(4, 4), textcoords="offset points", fontsize=7)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("trainable parameters")
        ax.set_title(alpha)
        ax.grid(True, which="both", alpha=0.3)
    axes[0].set_ylabel("final MSE (log)")
    from matplotlib.lines import Line2D
    legend = [Line2D([], [], marker="o", linestyle="", color="#c0392b", label="vanilla"),
              Line2D([], [], marker="o", linestyle="", color="#2c5fb0", label="ortho"),
              Line2D([], [], marker="o", linestyle="", color="grey", label="P=2"),
              Line2D([], [], marker="s", linestyle="", color="grey", label="P=3"),
              Line2D([], [], marker="^", linestyle="", color="grey", label="P=4")]
    axes[1].legend(handles=legend, fontsize=9, loc="upper right")
    fig.suptitle("MSE vs trainable parameter count — can ortho@small M match vanilla@big M?",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(OUT, "paramtradeoff_pareto.png"), dpi=140)
    plt.close(fig)


def render_trajectory_matches(matches, num_steps=5000):
    """Pick matches for which we still have the trajectory arrays and render
    truth / vanilla@bigger_M / ortho@smaller_M 2-D projections."""
    labels = ["x", "y", "z"]; pairs = [(0, 1), (0, 2), (1, 2)]
    usable = [m for m in matches
              if m["ortho"]["arrays_path"] and m["vanilla"]["arrays_path"]
              and os.path.exists(m["ortho"]["arrays_path"])
              and os.path.exists(m["vanilla"]["arrays_path"])]
    if not usable:
        return
    fig, axes = plt.subplots(len(usable), 9, figsize=(22, 2.6 * len(usable)))
    if len(usable) == 1:
        axes = np.array([axes])
    for r, m in enumerate(sorted(usable, key=lambda x: (x["ortho"]["alpha"], x["ortho"]["P"]))):
        d_o = np.load(m["ortho"]["arrays_path"])
        d_v = np.load(m["vanilla"]["arrays_path"])
        truth = d_o["X_test_trans"][:num_steps]
        orbit_o = d_o["orbit_ortho"][:num_steps]
        orbit_v = d_v["orbit_orig"][:num_steps]
        triples = [(truth, "black", f"truth  {m['ortho']['alpha']} P={m['ortho']['P']}"),
                   (orbit_v, "#c0392b",
                    f"vanilla M={m['vanilla']['M']}  mse={m['vanilla']['mse']:.4f}  p={m['vanilla']['params']}"),
                   (orbit_o, "#2c5fb0",
                    f"ortho M={m['ortho']['M']}  mse={m['ortho']['mse']:.4f}  p={m['ortho']['params']}")]
        col = 0
        for orb, color, name in triples:
            for i, j in pairs:
                ax = axes[r, col]
                ax.plot(orb[:, i], orb[:, j], color=color, lw=0.5, alpha=0.9)
                ax.set_title(f"{name}  {labels[i]}-{labels[j]}", fontsize=7)
                ax.set_aspect("equal", adjustable="datalim"); ax.tick_params(labelsize=6)
                col += 1
    fig.suptitle("Matched pairs: ortho at smaller M vs vanilla at larger M (same MSE within 15 %)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(os.path.join(OUT, "paramtradeoff_trajectories_2d.png"), dpi=130)
    plt.close(fig)


def main():
    rows = gather()
    matches = find_equivalent_pairs(rows)
    render_table(rows, matches)
    render_pareto(rows)
    render_trajectory_matches(matches)
    print(f"wrote paramtradeoff_report.txt, paramtradeoff_pareto.png, "
          f"paramtradeoff_trajectories_2d.png under {OUT}")


if __name__ == "__main__":
    main()
