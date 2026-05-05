# -*- coding: utf-8 -*-
"""All-in-one analysis for results_seedrun_v2/.

Aggregates per-cell metrics, runs statistics on Hellinger metrics,
generates heatmaps + forest plots + box plots + matched-pairs table,
renders high-res trajectories, and stitches a single comprehensive PDF.

The seedrun_v2 sweep has variable P per M:
    M=3 -> P in {2,3}
    M=4 -> P in {2,3,4}
    M=5 -> P in {2,3,4,5}
    M=8 -> P in {2,3,4,5,6}
    M=9 -> P in {2,3,4,5,6}

Compared to seedrun: smaller M values, ortho without ReLU bias, fixed 20-d
master B sliced for each M.
"""
import os, json, math
import numpy as np
import torch as tc
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy import stats as sci

from metrics import (
    state_space_divergence_binning,
    state_space_hellinger_binning,
    power_spectrum_error,
)

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_seedrun_v2")
M_LIST = [3, 4, 5, 8, 9]
SEEDS = [0, 1, 2]
P_FOR_M = {3: [2, 3], 4: [2, 3, 4], 5: [2, 3, 4, 5],
           8: [2, 3, 4, 5, 6], 9: [2, 3, 4, 5, 6]}
ALL_P = sorted({P for ps in P_FOR_M.values() for P in ps})
HELLINGER_KEYS = [
    ("PSE",       "PSE  (Hellinger of power spectrum)"),
    ("D_H_state", "D_H state-space (Hellinger)"),
]
ALL_KEYS = HELLINGER_KEYS + [
    ("Dstsp",            "Dstsp (KL state-space)"),
    ("rmse_train_final", "training RMSE"),
]


# --- aggregation -------------------------------------------------------------
def per_cell(orbit, truth):
    n = min(len(orbit), len(truth), 5000)
    orb3 = np.asarray(orbit)[:n, :3]
    tru3 = np.asarray(truth)[:n, :3]
    dstsp = state_space_divergence_binning(
        tc.tensor(orb3, dtype=tc.float32),
        tc.tensor(tru3, dtype=tc.float32), n_bins=30)
    d_h = state_space_hellinger_binning(
        tc.tensor(orb3, dtype=tc.float32),
        tc.tensor(tru3, dtype=tc.float32), n_bins=30)
    pse = power_spectrum_error(tru3, orb3, smoothing=20)
    h = min(100, n)
    rmse_h = float(np.sqrt(np.mean((orb3[:h] - tru3[:h]) ** 2)))
    return {"Dstsp": float(dstsp), "PSE": float(pse), "D_H_state": float(d_h),
            "rmse_h100": rmse_h}


def aggregate():
    out = {"per_M": {}}
    for M in M_LIST:
        out["per_M"][str(M)] = {"per_P": {}}
        for P in P_FOR_M[M]:
            cell = {"vanilla": {}, "ortho": {}}
            for seed in SEEDS:
                tag = f"M{M}_s{seed}"
                arr_path = os.path.join(OUT, f"P{P}_{tag}_arrays.npz")
                summary_path = os.path.join(OUT, f"summary_{tag}.json")
                if not (os.path.exists(arr_path) and os.path.exists(summary_path)):
                    continue
                d = np.load(arr_path)
                truth = d["X_test_trans"]
                with open(summary_path) as f:
                    s = json.load(f)
                if str(P) not in s["per_P"]:
                    continue
                pp = s["per_P"][str(P)]
                m_v = per_cell(d["orbit_orig"], truth)
                m_o = per_cell(d["orbit_ortho"], truth)
                m_v["mse_train_final"] = pp["final_loss_orig"]
                m_v["rmse_train_final"] = math.sqrt(pp["final_loss_orig"])
                m_o["mse_train_final"] = pp["final_loss_ortho"]
                m_o["rmse_train_final"] = math.sqrt(pp["final_loss_ortho"])
                cell["vanilla"][str(seed)] = m_v
                cell["ortho"][str(seed)] = m_o
            cs = {"vanilla": {}, "ortho": {}}
            for fam in ("vanilla", "ortho"):
                runs = list(cell[fam].values())
                if not runs:
                    continue
                for key in ("rmse_train_final", "rmse_h100", "Dstsp", "PSE", "D_H_state"):
                    vals = np.array([r.get(key, np.nan) for r in runs])
                    cs[fam][key] = {
                        "mean": float(np.nanmean(vals)),
                        "std":  float(np.nanstd(vals, ddof=1) if len(vals) > 1 else 0.0),
                        "n":    int(np.sum(~np.isnan(vals))),
                        "values": vals.tolist(),
                    }
            out["per_M"][str(M)]["per_P"][str(P)] = {"per_seed": cell, "summary": cs}
    with open(os.path.join(OUT, "metrics_seedrun_v2.json"), "w") as f:
        json.dump(out, f, indent=2)
    return out


# --- statistics --------------------------------------------------------------
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
    tcrit = sci.t.ppf(1 - alpha / 2, df=n - 1)
    return float(mean - tcrit * se), float(mean + tcrit * se)


def bootstrap_ci(values, n_boot=20000, alpha=0.05, seed=0):
    rng = np.random.default_rng(seed)
    values = np.asarray(values, float)
    if len(values) == 0:
        return (float("nan"), float("nan"))
    idx = rng.integers(0, len(values), size=(n_boot, len(values)))
    means = values[idx].mean(axis=1)
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def safe_wilcoxon(diff):
    diff = np.asarray(diff, float)
    diff = diff[diff != 0]
    if len(diff) < 1:
        return float("nan"), float("nan")
    try:
        with np.errstate(all="ignore"):
            res = sci.wilcoxon(diff, alternative="less")
        return float(res.statistic), float(res.pvalue)
    except Exception:
        return float("nan"), float("nan")


def per_cell_stats(metrics, key):
    rows = []
    for M in M_LIST:
        for P in P_FOR_M[M]:
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
                "ortho_mean": float(o.mean()),   "ortho_std": float(o.std(ddof=1) if len(o) > 1 else 0.0),
                "diff_mean":   float(diff.mean()),
                "diff_std":    float(diff.std(ddof=1) if len(diff) > 1 else 0.0),
                "rel_gap_pct": float(100 * diff.mean() / v.mean()) if v.mean() != 0 else 0.0,
                "cohens_dz":   cohens_dz(diff),
                "t_ci_lo":     ci_lo, "t_ci_hi": ci_hi,
                "wilcoxon_stat": stat, "wilcoxon_p_one_sided": p,
                "ortho_win_rate": float(np.mean(diff < 0)),
            })
    return rows


def pooled(rows):
    if not rows:
        return {}
    cell_gaps = np.array([r["diff_mean"] for r in rows], float)
    cell_rel  = np.array([r["rel_gap_pct"] for r in rows], float)
    boot_lo, boot_hi = bootstrap_ci(cell_rel)
    n_neg = int(np.sum(cell_gaps < 0))
    n_total = int(len(cell_gaps))
    sign_p = float(sci.binomtest(n_neg, n_total, 0.5, alternative="greater").pvalue)
    try:
        wstat, wp = sci.wilcoxon(cell_gaps, alternative="less")
        wstat, wp = float(wstat), float(wp)
    except Exception:
        wstat, wp = float("nan"), float("nan")
    return {
        "n_cells": n_total,
        "cell_mean_rel_gap_pct": float(cell_rel.mean()),
        "cell_rel_gap_bootstrap_95CI_pct": [boot_lo, boot_hi],
        "n_cells_ortho_wins_on_mean": n_neg,
        "sign_test_p_one_sided": sign_p,
        "wilcoxon_cell_means_p_one_sided": wp,
        "wilcoxon_cell_means_stat": wstat,
    }


def stratify(rows, by):
    if by == "low_M":
        sub = [r for r in rows if r["M"] in (3, 4, 5)]
    elif by == "high_M":
        sub = [r for r in rows if r["M"] in (8, 9)]
    elif by.startswith("M="):
        sub = [r for r in rows if r["M"] == int(by[2:])]
    elif by.startswith("P="):
        sub = [r for r in rows if r["P"] == int(by[2:])]
    else:
        sub = rows
    return pooled(sub)


def stat_analysis(metrics):
    out = {"per_cell": {}, "pooled": {}, "stratified": {}}
    for key, _ in ALL_KEYS:
        rows = per_cell_stats(metrics, key)
        out["per_cell"][key] = rows
        out["pooled"][key] = pooled(rows)
        out["stratified"][key] = {
            "low_M_345":  stratify(rows, "low_M"),
            "high_M_89":  stratify(rows, "high_M"),
            **{f"M{M}": stratify(rows, f"M={M}") for M in M_LIST},
            **{f"P{P}": stratify(rows, f"P={P}") for P in ALL_P},
        }
    with open(os.path.join(OUT, "statistical_analysis_v2.json"), "w") as f:
        json.dump(out, f, indent=2)
    return out


# --- figures -----------------------------------------------------------------
def grid_mean(metrics, family, key):
    G = np.full((len(M_LIST), len(ALL_P)), np.nan)
    for i, M in enumerate(M_LIST):
        for j, P in enumerate(ALL_P):
            if P not in P_FOR_M[M]:
                continue
            cell = metrics["per_M"].get(str(M), {}).get("per_P", {}).get(str(P), {})
            s = cell.get("summary", {}).get(family, {}).get(key)
            if s is not None:
                G[i, j] = s["mean"]
    return G


def heatmap_pair(metrics, key, label, savepath):
    Gv = grid_mean(metrics, "vanilla", key)
    Go = grid_mean(metrics, "ortho", key)
    vmin = float(np.nanmin([Gv, Go])); vmax = float(np.nanmax([Gv, Go]))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for ax, G, name in zip(axes, [Gv, Go], ["Vanilla mean", "Ortho mean"]):
        im = ax.imshow(G, cmap="viridis", vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_xticks(range(len(ALL_P))); ax.set_xticklabels([f"P={P}" for P in ALL_P])
        ax.set_yticks(range(len(M_LIST))); ax.set_yticklabels([f"M={M}" for M in M_LIST])
        ax.set_title(f"{name}: {label}", fontsize=11)
        for i in range(len(M_LIST)):
            for j in range(len(ALL_P)):
                if not np.isnan(G[i, j]):
                    ax.text(j, i, f"{G[i, j]:.3g}", ha="center", va="center",
                            color="white" if G[i, j] < (vmin + vmax) / 2 else "black",
                            fontsize=8)
        plt.colorbar(im, ax=ax, fraction=0.045)
    fig.suptitle(f"{label} (mean over 3 seeds, lower better)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(savepath, dpi=140); plt.close(fig)


def gap_heatmap(metrics, key, label, savepath):
    Gv = grid_mean(metrics, "vanilla", key); Go = grid_mean(metrics, "ortho", key)
    diff = (Go - Gv) / Gv * 100
    vmax = float(np.nanmax(np.abs(diff)))
    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    im = ax.imshow(diff, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(ALL_P))); ax.set_xticklabels([f"P={P}" for P in ALL_P])
    ax.set_yticks(range(len(M_LIST))); ax.set_yticklabels([f"M={M}" for M in M_LIST])
    for i in range(len(M_LIST)):
        for j in range(len(ALL_P)):
            if not np.isnan(diff[i, j]):
                ax.text(j, i, f"{diff[i, j]:+.0f}%", ha="center", va="center",
                        fontsize=9, color="black")
    ax.set_title(f"Gap (ortho-vanilla)/vanilla on mean {label}\nblue=ortho wins, red=vanilla wins")
    plt.colorbar(im, ax=ax, fraction=0.045)
    fig.tight_layout()
    fig.savefig(savepath, dpi=140); plt.close(fig)


def forest_plot(rows, label, savepath):
    labels = [f"M={r['M']} P={r['P']}" for r in rows]
    means = np.array([r["rel_gap_pct"] for r in rows])
    rel_lo = []; rel_hi = []
    for r in rows:
        v = max(r["vanilla_mean"], 1e-12)
        rel_lo.append(100 * r["t_ci_lo"] / v); rel_hi.append(100 * r["t_ci_hi"] / v)
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
    fig.savefig(savepath, dpi=150); plt.close(fig)


# --- trajectory pages --------------------------------------------------------
def median_seed(metrics, M, P):
    pse = metrics["per_M"][str(M)]["per_P"][str(P)]["summary"]["ortho"]["PSE"]["values"]
    order = sorted(range(len(pse)), key=lambda i: pse[i])
    return SEEDS[order[len(order) // 2]]


def stats_for(metrics, M, P, key, fam):
    s = metrics["per_M"][str(M)]["per_P"][str(P)]["summary"][fam][key]
    return s["mean"], s["std"]


def draw_3d(ax, orb, color, title, lw=0.8):
    orb = np.asarray(orb)
    ax.plot(orb[:, 0], orb[:, 1], orb[:, 2], color=color, lw=lw, alpha=0.95)
    ax.set_title(title, fontsize=10)
    ax.view_init(elev=25, azim=-60)
    rng = float(max(np.ptp(orb[:, 0]), np.ptp(orb[:, 1]), np.ptp(orb[:, 2])))
    cx, cy, cz = orb[:, 0].mean(), orb[:, 1].mean(), orb[:, 2].mean()
    ax.set_xlim(cx - rng/2 - 0.3, cx + rng/2 + 0.3)
    ax.set_ylim(cy - rng/2 - 0.3, cy + rng/2 + 0.3)
    ax.set_zlim(cz - rng/2 - 0.3, cz + rng/2 + 0.3)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])


def draw_2d(ax, orb, color, title, lw=0.5):
    orb = np.asarray(orb)
    ax.plot(orb[:, 0], orb[:, 2], color=color, lw=lw, alpha=0.95)
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("x", fontsize=8); ax.set_ylabel("z", fontsize=8)
    ax.set_aspect("equal", adjustable="datalim"); ax.tick_params(labelsize=7)


def trajectory_page(pdf, metrics, M, P, num_steps=5000):
    seed = median_seed(metrics, M, P)
    arr_path = os.path.join(OUT, f"P{P}_M{M}_s{seed}_arrays.npz")
    if not os.path.exists(arr_path):
        return
    d = np.load(arr_path)
    truth = d["X_test_trans"][:num_steps]
    orb_v = d["orbit_orig"][:num_steps]
    orb_o = d["orbit_ortho"][:num_steps]
    fig = plt.figure(figsize=(11, 8.5))
    pse_v_m, pse_v_s = stats_for(metrics, M, P, "PSE", "vanilla")
    pse_o_m, pse_o_s = stats_for(metrics, M, P, "PSE", "ortho")
    dh_v_m, dh_v_s = stats_for(metrics, M, P, "D_H_state", "vanilla")
    dh_o_m, dh_o_s = stats_for(metrics, M, P, "D_H_state", "ortho")
    rel_pse = (pse_o_m - pse_v_m) / pse_v_m * 100 if pse_v_m else 0.0
    rel_dh = (dh_o_m - dh_v_m) / dh_v_m * 100 if dh_v_m else 0.0
    fig.suptitle(
        f"M={M}  P={P}  (median-PSE seed={seed})  "
        f"PSE: van {pse_v_m:.3f}±{pse_v_s:.3f} ortho {pse_o_m:.3f}±{pse_o_s:.3f} ({rel_pse:+.1f}%)  "
        f"D_H: van {dh_v_m:.3f}±{dh_v_s:.3f} ortho {dh_o_m:.3f}±{dh_o_s:.3f} ({rel_dh:+.1f}%)",
        fontsize=10, fontweight="bold")
    ax1 = fig.add_subplot(2, 3, 1, projection="3d"); draw_3d(ax1, truth, "black", "Truth")
    ax2 = fig.add_subplot(2, 3, 2, projection="3d"); draw_3d(ax2, orb_v, "#c0392b", "Vanilla")
    ax3 = fig.add_subplot(2, 3, 3, projection="3d"); draw_3d(ax3, orb_o, "#2c5fb0", "Orthogonal")
    ax4 = fig.add_subplot(2, 3, 4); draw_2d(ax4, truth, "black", "Truth x-z")
    ax5 = fig.add_subplot(2, 3, 5); draw_2d(ax5, orb_v, "#c0392b", "Vanilla x-z")
    ax6 = fig.add_subplot(2, 3, 6); draw_2d(ax6, orb_o, "#2c5fb0", "Orthogonal x-z")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    pdf.savefig(fig, dpi=200); plt.close(fig)


# --- main / PDF --------------------------------------------------------------
def text_page(pdf, title, lines, fontsize=9):
    fig = plt.figure(figsize=(8.5, 11))
    ax = fig.add_axes([0.06, 0.04, 0.88, 0.92]); ax.axis("off")
    if title:
        ax.text(0.0, 0.98, title, fontsize=14, fontweight="bold",
                transform=ax.transAxes, va="top")
        y = 0.93
    else:
        y = 0.98
    for ln in lines:
        if ln.startswith("### "):
            ax.text(0.0, y, ln[4:], fontsize=fontsize + 1, fontweight="bold",
                    transform=ax.transAxes, va="top")
            y -= fontsize / 720.0 * 2.0
        else:
            ax.text(0.0, y, ln, fontsize=fontsize, family="monospace",
                    transform=ax.transAxes, va="top")
            y -= fontsize / 720.0 * 1.55
    pdf.savefig(fig); plt.close(fig)


def matched_pairs_lines(metrics):
    lines = ["Central question (v2): does ortho at smaller M match vanilla",
             "at bigger M?  Means over 3 seeds; ratio = ortho_small / vanilla_big.",
             "  ratio <= 0.95 -> ortho strictly better",
             "  0.95 < r <=1.05 -> match",
             "  r > 1.05 -> vanilla wins",
             ""]
    for slug, label in HELLINGER_KEYS:
        lines.append("")
        lines.append(f"### {label}")
        lines.append(f"  {'P':>2}  {'M_small':>7}  {'best M_big':>10}  "
                     f"{'ortho mean':>11}  {'vanilla mean':>12}  {'ratio':>6}  {'verdict':>9}")
        cnt = {"ortho >>": 0, "match": 0, "no improvement": 0}
        for M_s in M_LIST[:-1]:  # consider all but the largest
            for P in P_FOR_M[M_s]:
                cell_o = metrics["per_M"][str(M_s)]["per_P"][str(P)]["summary"]
                if "ortho" not in cell_o or slug not in cell_o["ortho"]:
                    continue
                o_s = cell_o["ortho"][slug]["mean"]
                best_M_b = None; best_r = None; best_v = None
                for M_b in M_LIST:
                    if M_b <= M_s: continue
                    if P not in P_FOR_M[M_b]: continue
                    cv = metrics["per_M"][str(M_b)]["per_P"][str(P)]["summary"]
                    if "vanilla" not in cv or slug not in cv["vanilla"]:
                        continue
                    v_b = cv["vanilla"][slug]["mean"]
                    r = o_s / v_b if v_b > 0 else float("inf")
                    if r <= 1.05:
                        if best_M_b is None or M_b > best_M_b:
                            best_M_b, best_r, best_v = M_b, r, v_b
                if best_M_b is None:
                    lines.append(f"  {P:>2}  M={M_s:<5}  {'(none)':>10}  "
                                 f"{o_s:>11.4f}  {'-':>12}  {'-':>6}  {'no impr':>9}")
                    cnt["no improvement"] += 1
                else:
                    verdict = "ortho >>" if best_r <= 0.95 else "match"
                    cnt[verdict] += 1
                    lines.append(f"  {P:>2}  M={M_s:<5}  M={best_M_b:<8}  "
                                 f"{o_s:>11.4f}  {best_v:>12.4f}  {best_r:>6.3f}  {verdict:>9}")
        total = sum(cnt.values())
        if total:
            lines.append("")
            lines.append(f"  free-M-reduction count: "
                         f"{cnt['ortho >>'] + cnt['match']}/{total} "
                         f"({100*(cnt['ortho >>']+cnt['match'])/total:.0f}%)")
    return lines


def main():
    print("Aggregating ...")
    metrics = aggregate()
    print("Running statistical analysis ...")
    stats = stat_analysis(metrics)

    paths = []
    for slug, label in ALL_KEYS:
        p = os.path.join(OUT, f"v2_heatmap_{slug}.png"); paths.append(p)
        heatmap_pair(metrics, slug, label, p)
        p = os.path.join(OUT, f"v2_gap_{slug}.png"); paths.append(p)
        gap_heatmap(metrics, slug, label, p)
    for slug, label in HELLINGER_KEYS:
        p = os.path.join(OUT, f"v2_forest_{slug}.png"); paths.append(p)
        forest_plot(stats["per_cell"][slug], label, p)

    pdf_path = os.path.join(OUT, "seedrun_v2_report.pdf")
    with PdfPages(pdf_path) as pdf:
        # cover
        text_page(pdf, "Seedrun v2: small-M sweep with bias-free ortho and B_full slicing",
                  [
                      "Sweep:  M in {3, 4, 5, 8, 9} x P <= M  x  3 seeds  x  {vanilla, ortho}",
                      "Total:  114 runs (M=3:2, M=4:3, M=5:4, M=8:5, M=9:5 P-pairs).",
                      "Epochs: 1000  alpha: 1.0",
                      "B:      one fixed master B in R^{3 x 20} (seed_B = 42); each",
                      "        run uses B[:, :M] as its frozen B.",
                      "Ortho:  no bias inside ReLU (pure rotation Q z).",
                      "",
                      "### Headline (D_H_state, state-space Hellinger)",
                      "  pooled mean rel gap     : "
                      f"{stats['pooled']['D_H_state']['cell_mean_rel_gap_pct']:+.2f}%",
                      "  bootstrap 95% CI        : "
                      f"[{stats['pooled']['D_H_state']['cell_rel_gap_bootstrap_95CI_pct'][0]:+.2f}%, "
                      f"{stats['pooled']['D_H_state']['cell_rel_gap_bootstrap_95CI_pct'][1]:+.2f}%]",
                      "  ortho wins on mean      : "
                      f"{stats['pooled']['D_H_state']['n_cells_ortho_wins_on_mean']}/"
                      f"{stats['pooled']['D_H_state']['n_cells']}",
                      "  sign-test p (one-sided) : "
                      f"{stats['pooled']['D_H_state']['sign_test_p_one_sided']:.4f}",
                      "  Wilcoxon p (one-sided)  : "
                      f"{stats['pooled']['D_H_state']['wilcoxon_cell_means_p_one_sided']:.4f}",
                      "",
                      "### Headline (PSE, power-spectrum Hellinger)",
                      "  pooled mean rel gap     : "
                      f"{stats['pooled']['PSE']['cell_mean_rel_gap_pct']:+.2f}%",
                      "  bootstrap 95% CI        : "
                      f"[{stats['pooled']['PSE']['cell_rel_gap_bootstrap_95CI_pct'][0]:+.2f}%, "
                      f"{stats['pooled']['PSE']['cell_rel_gap_bootstrap_95CI_pct'][1]:+.2f}%]",
                      "  ortho wins on mean      : "
                      f"{stats['pooled']['PSE']['n_cells_ortho_wins_on_mean']}/"
                      f"{stats['pooled']['PSE']['n_cells']}",
                      "  sign-test p             : "
                      f"{stats['pooled']['PSE']['sign_test_p_one_sided']:.4f}",
                  ])

        # per-cell stats tables
        for slug, label in HELLINGER_KEYS:
            rows = stats["per_cell"][slug]
            lines = [f"# Per-cell paired statistics on {label}", ""]
            lines.append(f"  {'M':>2} {'P':>2}  {'vanilla':>14}  {'ortho':>14}  "
                         f"{'rel_gap%':>9}  {'d_z':>5}  {'95% t-CI on diff':>22}  "
                         f"{'wilc p':>6}  {'wins':>5}")
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
            lines.append("")
            p = stats["pooled"][slug]
            lines.append("## Pooled across cells")
            lines.append(f"   mean rel gap         : {p['cell_mean_rel_gap_pct']:+.2f}%")
            lines.append(f"   bootstrap 95% CI     : "
                         f"[{p['cell_rel_gap_bootstrap_95CI_pct'][0]:+.2f}%, "
                         f"{p['cell_rel_gap_bootstrap_95CI_pct'][1]:+.2f}%]")
            lines.append(f"   wins                 : {p['n_cells_ortho_wins_on_mean']}/{p['n_cells']}")
            lines.append(f"   sign-test p          : {p['sign_test_p_one_sided']:.4f}")
            lines.append(f"   Wilcoxon p           : {p['wilcoxon_cell_means_p_one_sided']:.4f}")
            lines.append("")
            lines.append("## Stratified by M")
            for M in M_LIST:
                k = f"M{M}"
                v = stats["stratified"][slug].get(k, {})
                if v:
                    lines.append(f"   M={M:<2}  mean rel gap = "
                                 f"{v.get('cell_mean_rel_gap_pct', 0):+7.2f}%   "
                                 f"({v.get('n_cells_ortho_wins_on_mean', 0)}/"
                                 f"{v.get('n_cells', 0)})")
            lines.append("")
            lines.append("## Stratified by P")
            for P in ALL_P:
                k = f"P{P}"
                v = stats["stratified"][slug].get(k, {})
                if v:
                    lines.append(f"   P={P:<2}  mean rel gap = "
                                 f"{v.get('cell_mean_rel_gap_pct', 0):+7.2f}%   "
                                 f"({v.get('n_cells_ortho_wins_on_mean', 0)}/"
                                 f"{v.get('n_cells', 0)})")
            text_page(pdf, f"Statistics: {label}", lines, fontsize=8)

        # matched pairs
        text_page(pdf, "Can ortho at smaller M match vanilla at bigger M? (v2)",
                  matched_pairs_lines(metrics), fontsize=8)

        # all heatmap+gap pages (D_H_state + PSE first)
        for slug, label in HELLINGER_KEYS:
            for png in [f"v2_heatmap_{slug}.png", f"v2_gap_{slug}.png"]:
                p = os.path.join(OUT, png)
                if os.path.exists(p):
                    img = plt.imread(p)
                    fig = plt.figure(figsize=(8.5, 11))
                    ax = fig.add_axes([0.02, 0.02, 0.96, 0.96]); ax.imshow(img); ax.axis("off")
                    pdf.savefig(fig); plt.close(fig)
        # forest plots
        for slug, _ in HELLINGER_KEYS:
            p = os.path.join(OUT, f"v2_forest_{slug}.png")
            img = plt.imread(p)
            fig = plt.figure(figsize=(8.5, 11))
            ax = fig.add_axes([0.02, 0.02, 0.96, 0.96]); ax.imshow(img); ax.axis("off")
            pdf.savefig(fig); plt.close(fig)

        # highlight trajectories: top 6 ortho wins on D_H_state, plus median + worst
        rows = stats["per_cell"]["D_H_state"]
        ranked = sorted(rows, key=lambda r: r["rel_gap_pct"])
        text_page(pdf, "Highlight trajectories",
                  [
                      "Each of the next pages shows one (M, P) cell at the median-PSE",
                      "seed: 3-D views (top row) and x-z 2-D projections (bottom row)",
                      "for ground truth / vanilla / orthogonal.",
                      "",
                      "Order: best ortho wins first on D_H_state (state-space Hellinger).",
                      "",
                  ] + [
                      f"  M={r['M']:<3} P={r['P']:<3}  D_H gap = {r['rel_gap_pct']:+6.1f}%   "
                      f"PSE gap = {next(x['rel_gap_pct'] for x in stats['per_cell']['PSE'] if x['M']==r['M'] and x['P']==r['P']):+6.1f}%"
                      for r in ranked[:6]
                  ] + [
                      "",
                      "Failure cell (worst ortho regression):",
                      f"  M={ranked[-1]['M']:<3} P={ranked[-1]['P']:<3}  "
                      f"D_H gap = {ranked[-1]['rel_gap_pct']:+6.1f}%",
                  ])
        for r in ranked[:6]:
            trajectory_page(pdf, metrics, r["M"], r["P"])
        trajectory_page(pdf, metrics, ranked[-1]["M"], ranked[-1]["P"])

    print(f"wrote {pdf_path}")
    print(f"  size: {os.path.getsize(pdf_path)/1024/1024:.1f} MB")


if __name__ == "__main__":
    main()
