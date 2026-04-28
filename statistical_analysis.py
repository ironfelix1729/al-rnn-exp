# -*- coding: utf-8 -*-
"""Statistical analysis of multi-seed M-finescan results.

Focus on the two Hellinger-flavoured metrics the user prioritises:
  * PSE       - mean Hellinger distance between smoothed power spectra
  * D_H_state - Hellinger distance on a 30^3 binned 3-D state-space density
plus secondary: Dstsp (KL) and rmse_train_final.

For each (M, P) cell with paired seeds we compute:
  * mean and SD per family (already in metrics_seedrun.json)
  * mean paired difference o-v with paired SD
  * Cohen's d_z paired effect size
  * Wilcoxon signed-rank statistic (degenerate at n=3 but reported)
  * t 95% CI on paired mean (df=n-1)
  * win rate over seeds (#seeds with ortho < vanilla)

For pooled analysis (across all 20 cells):
  * mean of cell-level mean gaps, with bootstrap 95% CI
  * sign test of cell-level mean gaps  (H0: ortho not better)
  * Wilcoxon signed-rank across cell-level mean gaps
  * stratified analysis by M-band (10-11 vs 14-15) and by P-band
"""
import os, json, math
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_seedrun")
M_LIST = [10, 11, 14, 15]
P_LIST = [2, 3, 4, 5, 6]
HELLINGER_KEYS = [
    ("PSE",       "PSE  (Hellinger of power spectrum)"),
    ("D_H_state", "D_H state-space  (Hellinger)"),
]
ALL_KEYS = HELLINGER_KEYS + [
    ("Dstsp",            "Dstsp (KL state-space)"),
    ("rmse_train_final", "training RMSE"),
]


def cohens_dz(diff):
    diff = np.asarray(diff, float)
    if len(diff) < 2 or np.std(diff, ddof=1) == 0:
        return float("nan")
    return float(np.mean(diff) / np.std(diff, ddof=1))


def t_ci(diff, alpha=0.05):
    diff = np.asarray(diff, float)
    n = len(diff)
    if n < 2:
        return (float("nan"), float("nan"))
    mean = np.mean(diff)
    se = np.std(diff, ddof=1) / np.sqrt(n)
    tcrit = stats.t.ppf(1 - alpha / 2, df=n - 1)
    return float(mean - tcrit * se), float(mean + tcrit * se)


def bootstrap_ci(values, n_boot=10000, alpha=0.05, seed=0):
    rng = np.random.default_rng(seed)
    values = np.asarray(values, float)
    if len(values) == 0:
        return (float("nan"), float("nan"))
    idx = rng.integers(0, len(values), size=(n_boot, len(values)))
    means = values[idx].mean(axis=1)
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def safe_wilcoxon(diff):
    """Wilcoxon signed-rank with n_seeds=3 is essentially a sign test."""
    diff = np.asarray(diff, float)
    diff = diff[diff != 0]
    if len(diff) < 1:
        return float("nan"), float("nan")
    try:
        with np.errstate(all="ignore"):
            res = stats.wilcoxon(diff, alternative="less")  # H1: ortho - vanilla < 0
        return float(res.statistic), float(res.pvalue)
    except Exception:
        return float("nan"), float("nan")


def per_cell_stats(metrics, key):
    rows = []
    for M in M_LIST:
        for P in P_LIST:
            cell = metrics["per_M"].get(str(M), {}).get("per_P", {}).get(str(P), {})
            v = cell.get("summary", {}).get("vanilla", {}).get(key, {}).get("values", [])
            o = cell.get("summary", {}).get("ortho", {}).get(key, {}).get("values", [])
            if len(v) != len(o) or len(v) == 0:
                continue
            v = np.asarray(v, float); o = np.asarray(o, float)
            diff = o - v
            stat, p = safe_wilcoxon(diff)
            ci_lo, ci_hi = t_ci(diff)
            rows.append({
                "M": M, "P": P, "n": int(len(v)),
                "vanilla_mean": float(v.mean()), "vanilla_std": float(v.std(ddof=1) if len(v) > 1 else 0.0),
                "ortho_mean":   float(o.mean()), "ortho_std":   float(o.std(ddof=1) if len(o) > 1 else 0.0),
                "diff_mean":    float(diff.mean()),
                "diff_std":     float(diff.std(ddof=1) if len(diff) > 1 else 0.0),
                "rel_gap_pct":  float(100 * diff.mean() / v.mean()) if v.mean() != 0 else 0.0,
                "cohens_dz":    cohens_dz(diff),
                "t_ci_lo":      ci_lo, "t_ci_hi": ci_hi,
                "wilcoxon_stat": stat, "wilcoxon_p_one_sided": p,
                "ortho_win_rate": float(np.mean(diff < 0)),
            })
    return rows


def pooled(rows, label):
    cell_gaps = np.array([r["diff_mean"] for r in rows], float)
    cell_rel = np.array([r["rel_gap_pct"] for r in rows], float)
    if len(cell_gaps) == 0:
        return {}
    boot_lo, boot_hi = bootstrap_ci(cell_rel, n_boot=20000, seed=42)
    n_neg = int(np.sum(cell_gaps < 0))
    n_total = int(len(cell_gaps))
    sign_p = float(stats.binomtest(n_neg, n_total, 0.5, alternative="greater").pvalue)
    try:
        wstat, wp = stats.wilcoxon(cell_gaps, alternative="less")
        wstat, wp = float(wstat), float(wp)
    except Exception:
        wstat, wp = float("nan"), float("nan")
    return {
        "label": label,
        "n_cells": n_total,
        "cell_mean_gap_abs": float(cell_gaps.mean()),
        "cell_mean_rel_gap_pct": float(cell_rel.mean()),
        "cell_rel_gap_bootstrap_95CI_pct": [boot_lo, boot_hi],
        "n_cells_ortho_wins_on_mean": n_neg,
        "sign_test_p_one_sided": sign_p,
        "wilcoxon_cell_means_p_one_sided": wp,
        "wilcoxon_cell_means_stat": wstat,
    }


def stratify(rows, key, group):
    if group == "low_M":
        sub = [r for r in rows if r["M"] in (10, 11)]
    elif group == "high_M":
        sub = [r for r in rows if r["M"] in (14, 15)]
    elif group.startswith("P="):
        P = int(group[2:])
        sub = [r for r in rows if r["P"] == P]
    else:
        sub = rows
    return pooled(sub, f"{key}/{group}")


def main():
    metrics = json.load(open(os.path.join(OUT, "metrics_seedrun.json")))
    out = {"per_cell": {}, "pooled": {}, "stratified": {}}
    text = []
    add = text.append
    add("=" * 92)
    add("Multi-seed statistical analysis (n=3 seeds per cell, 4 M x 5 P = 20 cells)")
    add("Hellinger-flavoured metrics are the priority (PSE, D_H_state).")
    add("=" * 92)
    for key, label in ALL_KEYS:
        rows = per_cell_stats(metrics, key)
        out["per_cell"][key] = rows
        out["pooled"][key] = pooled(rows, key)
        out["stratified"][key] = {
            "low_M_only_M10_M11":   stratify(rows, key, "low_M"),
            "high_M_only_M14_M15":  stratify(rows, key, "high_M"),
            **{f"P_eq_{P}": stratify(rows, key, f"P={P}") for P in P_LIST},
        }
        add("")
        add(f"### {label}  (lower is better; rel_gap_pct < 0 => ortho wins)")
        add(f"  {'M':>3} {'P':>3}  {'vanilla':>14}  {'ortho':>14}  {'rel_gap%':>9}  "
            f"{'d_z':>6}  {'95% t CI on gap':>18}  {'wilc p':>7}  {'win/3':>5}")
        for r in rows:
            add(f"  {r['M']:>3} {r['P']:>3}  "
                f"{r['vanilla_mean']:>10.5f}±{r['vanilla_std']:>4.3f}  "
                f"{r['ortho_mean']:>10.5f}±{r['ortho_std']:>4.3f}  "
                f"{r['rel_gap_pct']:>+8.1f}%  "
                f"{r['cohens_dz']:>+6.2f}  "
                f"[{r['t_ci_lo']:>+8.4g},{r['t_ci_hi']:>+8.4g}]  "
                f"{r['wilcoxon_p_one_sided']:>6.3f}  "
                f"{int(r['ortho_win_rate']*r['n']):>1}/{r['n']:>1}")
        add("")
        p = out["pooled"][key]
        add(f"  pooled across {p['n_cells']} cells:")
        add(f"    mean rel gap        = {p['cell_mean_rel_gap_pct']:+.2f}%   "
            f"95% bootstrap CI = [{p['cell_rel_gap_bootstrap_95CI_pct'][0]:+.2f}%, "
            f"{p['cell_rel_gap_bootstrap_95CI_pct'][1]:+.2f}%]")
        add(f"    cells where ortho wins on mean: {p['n_cells_ortho_wins_on_mean']}/{p['n_cells']}   "
            f"sign test p (H1: ortho>vanilla cells > 50%) = {p['sign_test_p_one_sided']:.3g}")
        add(f"    Wilcoxon signed rank on cell-level gaps p (one-sided 'ortho<van') = "
            f"{p['wilcoxon_cell_means_p_one_sided']:.3g}")
        add("")
        add(f"  stratified means:")
        for grp_key, grp_val in out["stratified"][key].items():
            add(f"    {grp_key:<22s} mean rel gap = {grp_val['cell_mean_rel_gap_pct']:+7.2f}%   "
                f"ortho wins {grp_val['n_cells_ortho_wins_on_mean']}/{grp_val['n_cells']} cells")
    txt = "\n".join(text)
    with open(os.path.join(OUT, "statistical_analysis.txt"), "w") as f:
        f.write(txt + "\n")
    with open(os.path.join(OUT, "statistical_analysis.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(txt[:6000])

    # render a couple of figures
    # 1. scatter gap vs Cohen's d for the two Hellinger metrics
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4))
    for ax, (key, label) in zip(axes, HELLINGER_KEYS):
        rows = out["per_cell"][key]
        gaps = [r["rel_gap_pct"] for r in rows]
        dz = [r["cohens_dz"] for r in rows]
        labels = [f"M={r['M']} P={r['P']}" for r in rows]
        ax.axhline(0, color="k", lw=0.5, ls=":")
        ax.axvline(0, color="k", lw=0.5, ls=":")
        ax.scatter(gaps, dz, s=50, c=["#2c5fb0" if g < 0 else "#c0392b" for g in gaps],
                   alpha=0.85, edgecolors="black")
        for x, y, lab in zip(gaps, dz, labels):
            ax.annotate(lab, (x, y), xytext=(3, 3), textcoords="offset points",
                        fontsize=7)
        ax.set_xlabel("relative gap (ortho - vanilla) / vanilla  (%)")
        ax.set_ylabel(r"Cohen's $d_z$ (paired effect size)")
        ax.set_title(label)
        ax.grid(True, alpha=0.3)
    fig.suptitle("Per-cell effect sizes on Hellinger metrics (blue = ortho wins on mean)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(OUT, "stat_effectsize_hellinger.png"), dpi=150)
    plt.close(fig)

    # 2. forest plot per cell on PSE: mean gap with 95% t-CI
    for key, label in HELLINGER_KEYS:
        rows = out["per_cell"][key]
        labels = [f"M={r['M']} P={r['P']}" for r in rows]
        means = np.array([r["rel_gap_pct"] for r in rows])
        # convert t CI on absolute gap to relative by dividing by vanilla mean
        rel_lo = []; rel_hi = []
        for r in rows:
            v = max(r["vanilla_mean"], 1e-12)
            rel_lo.append(100 * r["t_ci_lo"] / v)
            rel_hi.append(100 * r["t_ci_hi"] / v)
        rel_lo = np.array(rel_lo); rel_hi = np.array(rel_hi)
        order = np.argsort(means)
        labels = [labels[i] for i in order]; means = means[order]
        lo = rel_lo[order]; hi = rel_hi[order]
        fig, ax = plt.subplots(1, 1, figsize=(9, 7))
        y = np.arange(len(labels))
        for i, (m, l, h) in enumerate(zip(means, lo, hi)):
            color = "#2c5fb0" if m < 0 else "#c0392b"
            ax.plot([l, h], [i, i], color=color, lw=2)
            ax.plot(m, i, "o", color=color, markersize=7)
        ax.axvline(0, color="k", lw=0.5, ls=":")
        ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel("relative gap (%)   95% t-CI")
        ax.set_title(f"Forest plot of per-cell ortho-vs-vanilla gap on {label}")
        ax.grid(True, alpha=0.3, axis="x")
        fig.tight_layout()
        slug = key
        fig.savefig(os.path.join(OUT, f"stat_forest_{slug}.png"), dpi=150)
        plt.close(fig)

    print("\nWrote statistical_analysis.{txt,json}, stat_effectsize_hellinger.png, "
          "stat_forest_PSE.png, stat_forest_D_H_state.png")


if __name__ == "__main__":
    main()
