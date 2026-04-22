# -*- coding: utf-8 -*-
"""Analyse the input-embedding matrix B (3x20) of the trained models.

B is used only to initialise the hidden state:  z_0 = x_0 @ B
Then the first N=3 coordinates of z_0 are overwritten by x_0 (both at t=0
and at every teacher-forcing step).  So B affects only the INITIAL values
of the 17 "hidden" coordinates z_0[N:].  The columns B[:, :N] are therefore
irrelevant to the dynamics (those coords get clobbered); only B[:, N:] (a
3x17 block) matters.
"""
import os, json
import numpy as np
import torch
import matplotlib.pyplot as plt

from experiment_p_sweep import AL_RNN_Original, AL_RNN_Orthogonal

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_p_sweep")
M, N = 20, 3


def load_B(model_class, P):
    model = model_class(M=M, P=P, N=N)
    name = "orig" if model_class is AL_RNN_Original else "ortho"
    sd = torch.load(os.path.join(OUT, f"model_{name}_P{P}.pt"),
                    map_location="cpu", weights_only=True)
    model.load_state_dict(sd, strict=False)
    return model.B.detach().cpu().numpy()  # shape (N, M) = (3, 20)


def row_angles_deg(M_):
    M_ = M_ / np.linalg.norm(M_, axis=1, keepdims=True)
    cos = M_ @ M_.T
    cos = np.clip(cos, -1.0, 1.0)
    angles = np.degrees(np.arccos(cos))
    # off-diagonal pairs
    offd = [angles[i, j] for i in range(M_.shape[0]) for j in range(i + 1, M_.shape[0])]
    return np.array(offd)


def analyze_B(B):
    # B is (N=3, M=20): maps x in R^3 -> z_0 = x @ B in R^20
    row_norms = np.linalg.norm(B, axis=1)  # size of each row = how strongly each input dim is written
    frob = np.linalg.norm(B)
    svs = np.linalg.svd(B, compute_uv=False)  # 3 singular values
    cond = svs.max() / max(svs.min(), 1e-12)
    # how orthogonal are the 3 rows of B?
    row_angles = row_angles_deg(B)
    # what fraction of each column norm lies in the 'hidden' columns M[:,N:]?
    obs_block_norm = np.linalg.norm(B[:, :N])      # 3x3 block
    hidden_block_norm = np.linalg.norm(B[:, N:])   # 3x17 block
    # effective rank
    sv_fracs = svs / svs.sum()
    eff_rank = float(np.exp(-np.sum(sv_fracs * np.log(sv_fracs + 1e-12))))
    return {
        "frob": float(frob),
        "row_norms": row_norms.tolist(),
        "singular_values": svs.tolist(),
        "cond_number": float(cond),
        "row_pair_angles_deg": row_angles.tolist(),
        "mean_row_angle_deg": float(np.mean(row_angles)),
        "obs_block_frob": float(obs_block_norm),
        "hidden_block_frob": float(hidden_block_norm),
        "obs_fraction": float(obs_block_norm**2 / (obs_block_norm**2 + hidden_block_norm**2)),
        "effective_rank": eff_rank,
    }


def main():
    print(f"Initialisation scale of B: uniform[-1/sqrt(N), 1/sqrt(N)] = "
          f"+/- {1/np.sqrt(N):.3f}, expected ||B||_F ~ {np.sqrt(N*M/3) / np.sqrt(3):.2f}")
    print("")
    hdr = (f"  {'P':>3} {'model':<8}  {'||B||_F':>8}  {'cond#':>7}  {'row‖':>18}  "
           f"{'row∠ (deg)':>17}  {'obs_frac':>8}  {'eff_rank':>8}  {'sv':>22}")
    print(hdr)
    rows = []
    for P in [2, 3, 4, 5, 6]:
        for name, cls in [("vanilla", AL_RNN_Original), ("ortho", AL_RNN_Orthogonal)]:
            B = load_B(cls, P)
            s = analyze_B(B)
            rn = ",".join(f"{x:.2f}" for x in s["row_norms"])
            ra = ",".join(f"{x:.1f}" for x in s["row_pair_angles_deg"])
            sv = ",".join(f"{x:.2f}" for x in s["singular_values"])
            print(f"  {P:>3} {name:<8}  {s['frob']:>8.3f}  {s['cond_number']:>7.2f}  "
                  f"{rn:>18}  {ra:>17}  {s['obs_fraction']:>8.3f}  "
                  f"{s['effective_rank']:>8.2f}  {sv:>22}")
            rows.append({"P": P, "model": name, **s})

    with open(os.path.join(OUT, "B_analysis.json"), "w") as f:
        json.dump(rows, f, indent=2)

    # render heatmaps
    fig, axes = plt.subplots(5, 2, figsize=(14, 14))
    for r, P in enumerate([2, 3, 4, 5, 6]):
        for c, (name, cls) in enumerate([("vanilla", AL_RNN_Original), ("ortho", AL_RNN_Orthogonal)]):
            B = load_B(cls, P)
            vmax = float(np.max(np.abs(B)))
            im = axes[r, c].imshow(B, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
            axes[r, c].set_title(f"B  P={P}  {name}   "
                                 f"||B||={np.linalg.norm(B):.2f}  cond={np.linalg.svd(B,compute_uv=False)[0]/np.linalg.svd(B,compute_uv=False)[-1]:.1f}")
            axes[r, c].axvline(N - 0.5, color="black", lw=0.8, ls="--")
            axes[r, c].set_xlabel("hidden index j  (| = obs/hidden split)")
            axes[r, c].set_ylabel("input dim i")
            plt.colorbar(im, ax=axes[r, c], fraction=0.03)
    fig.suptitle("Input embedding B (3 x 20) across P and models", fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "B_heatmaps.png"), dpi=130)
    plt.close(fig)

    # row-norm trend plot
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    Ps = [2, 3, 4, 5, 6]
    for name, cls, ls, marker in [("vanilla", AL_RNN_Original, "--", "o"),
                                  ("ortho",   AL_RNN_Orthogonal, "-",  "s")]:
        frob_list = []
        cond_list = []
        for P in Ps:
            B = load_B(cls, P)
            frob_list.append(np.linalg.norm(B))
            svs = np.linalg.svd(B, compute_uv=False)
            cond_list.append(svs.max() / max(svs.min(), 1e-12))
        ax.plot(Ps, frob_list, ls + marker, label=f"||B||_F  {name}")
        ax.plot(Ps, cond_list, ls + "^", label=f"cond(B)  {name}", alpha=0.7)
    ax.set_xlabel("P"); ax.set_ylabel("value")
    ax.set_title("Frobenius norm and condition number of B")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "B_trends.png"), dpi=130)
    plt.close(fig)
    print(f"\nSaved B_heatmaps.png, B_trends.png, B_analysis.json under {OUT}")


if __name__ == "__main__":
    main()
