# -*- coding: utf-8 -*-
"""Symbolic-code demonstration for vanilla and orthogonal AL-RNN.

The AL-RNN paper (Brenner, Hemmer, Monfared, Durstewitz, NeurIPS 2024) argues
that with P ReLU units a trained AL-RNN partitions the latent space into 2^P
linear regions, each labelled by the sign pattern of the P ReLU inputs - a
symbolic code that preserves the topological structure of the dynamics.

For vanilla AL-RNN those hyperplanes are axis-aligned:
    sigma_t = ( 1{z_{t,M-P} > 0}, ..., 1{z_{t,M-1} > 0} ) in {0,1}^P

For the ortho AL-RNN the P hyperplanes are the last P rows of Q plus b_Q:
    sigma_t = ( 1{(Q z_t + b_Q)_{M-P} > 0}, ..., 1{(Q z_t + b_Q)_{M-1} > 0} )

So the same 2^P symbolic alphabet exists in the ortho model; only the basis
in which the P ReLU hyperplanes are expressed is rotated.  This script
* reads off the symbolic sequence from a saved free-run trajectory;
* renders the 3-D attractor coloured by symbol;
* reports symbol-entropy and transition statistics.
"""
import os, json
import numpy as np
import torch
import matplotlib.pyplot as plt
from collections import Counter

from experiment_p_sweep import AL_RNN_Original, AL_RNN_Orthogonal

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_p_sweep")


def load_model(cls, M, P, tag):
    m = cls(M=M, P=P, N=3)
    sd = torch.load(os.path.join(OUT, f"model_{'orig' if cls is AL_RNN_Original else 'ortho'}_P{P}_{tag}.pt"),
                    map_location="cpu", weights_only=True)
    m.load_state_dict(sd, strict=False)
    m.eval()
    return m


def symbols_vanilla(z, P):
    bits = (z[:, -P:] > 0).astype(np.int64)
    sym = np.zeros(z.shape[0], dtype=np.int64)
    for k in range(P):
        sym |= bits[:, k] << k
    return sym  # values in [0, 2**P)


def symbols_ortho(z, model):
    Q = model.transform.weight.detach().cpu().numpy()
    b = model.transform.bias.detach().cpu().numpy()
    y = z @ Q.T + b
    return symbols_vanilla(y, model.P)


def symbol_entropy(sym, P):
    cnt = Counter(sym.tolist())
    probs = np.array([cnt.get(i, 0) for i in range(2 ** P)], dtype=np.float64)
    probs = probs / probs.sum()
    return float(-np.sum(probs[probs > 0] * np.log2(probs[probs > 0])))


def transition_matrix(sym, P):
    K = 2 ** P
    T = np.zeros((K, K), dtype=np.int64)
    for a, b in zip(sym[:-1], sym[1:]):
        T[a, b] += 1
    return T


def plot_trajectory_coloured(ax, z, sym, title, num_steps=5000):
    z = z[:num_steps]; sym = sym[:num_steps]
    K = int(sym.max()) + 1
    cmap = plt.cm.tab10 if K <= 10 else plt.cm.tab20
    for s in range(K):
        mask = sym == s
        if mask.sum() < 5:
            continue
        ax.scatter(z[mask, 0], z[mask, 1], z[mask, 2],
                   s=1.2, color=cmap(s % cmap.N), alpha=0.6, label=f"σ={s:0{max(1, int(np.ceil(np.log2(max(K,2)))))}b}")
    ax.set_title(title, fontsize=10)
    ax.view_init(elev=25, azim=-60)
    rng = float(max(np.ptp(z[:, 0]), np.ptp(z[:, 1]), np.ptp(z[:, 2])))
    cx, cy, cz = z[:, 0].mean(), z[:, 1].mean(), z[:, 2].mean()
    ax.set_xlim(cx - rng / 2 - .3, cx + rng / 2 + .3)
    ax.set_ylim(cy - rng / 2 - .3, cy + rng / 2 + .3)
    ax.set_zlim(cz - rng / 2 - .3, cz + rng / 2 + .3)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])


def main():
    # pick a representative matched pair from α=1.0: M=5 P=3 (50 ortho params, matches vanilla M=20)
    M, P = 5, 3
    d = np.load(os.path.join(OUT, f"P{P}_M{M}_arrays.npz"))
    z_v = d["orbit_orig"]; z_o = d["orbit_ortho"]
    m_o = load_model(AL_RNN_Orthogonal, M, P, "M5")

    sym_v = symbols_vanilla(z_v, P)
    sym_o = symbols_ortho(z_o, m_o)

    H_v = symbol_entropy(sym_v, P)
    H_o = symbol_entropy(sym_o, P)
    T_v = transition_matrix(sym_v, P)
    T_o = transition_matrix(sym_o, P)

    lines = []
    add = lines.append
    add("Symbolic-code diagnostics (M=5 P=3, free-run 5000 steps)")
    add("=" * 64)
    add(f"Symbol alphabet size:   2^P = {2**P}")
    add(f"Shannon entropy (bits): vanilla = {H_v:.3f},  ortho = {H_o:.3f}   "
        f"(log2(8) = 3.0)")
    add("")
    visited_v = int((np.array([np.sum(sym_v == s) for s in range(2**P)]) > 0).sum())
    visited_o = int((np.array([np.sum(sym_o == s) for s in range(2**P)]) > 0).sum())
    add(f"Distinct symbols visited:  vanilla = {visited_v} / {2**P}   ortho = {visited_o} / {2**P}")
    add("")
    add("Most common symbol transitions (vanilla):")
    pairs_v = sorted([((a, b), int(T_v[a, b])) for a in range(2**P) for b in range(2**P) if T_v[a, b] > 0],
                      key=lambda t: -t[1])[:6]
    for (a, b), c in pairs_v:
        add(f"   {a:0{P}b}  ->  {b:0{P}b}   count={c}")
    add("")
    add("Most common symbol transitions (ortho):")
    pairs_o = sorted([((a, b), int(T_o[a, b])) for a in range(2**P) for b in range(2**P) if T_o[a, b] > 0],
                      key=lambda t: -t[1])[:6]
    for (a, b), c in pairs_o:
        add(f"   {a:0{P}b}  ->  {b:0{P}b}   count={c}")
    print("\n".join(lines))
    with open(os.path.join(OUT, "symbolic_code_diagnostics.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")

    # 2-panel figure: vanilla symbols vs ortho symbols on the 3-D attractor
    fig = plt.figure(figsize=(13, 5.2))
    ax1 = fig.add_subplot(121, projection="3d")
    plot_trajectory_coloured(ax1, z_v, sym_v,
                             f"Vanilla AL-RNN  M={M} P={P} — symbols from sign(z[-P:])")
    ax1.legend(loc="upper left", fontsize=7, markerscale=5, bbox_to_anchor=(1.02, 1.0))
    ax2 = fig.add_subplot(122, projection="3d")
    plot_trajectory_coloured(ax2, z_o, sym_o,
                             f"Orthogonal AL-RNN  M={M} P={P} — symbols from sign((Qz+b)[-P:])")
    ax2.legend(loc="upper left", fontsize=7, markerscale=5, bbox_to_anchor=(1.02, 1.0))
    fig.suptitle("Symbolic code survives the orthogonal reparameterisation",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(os.path.join(OUT, "symbolic_code_coloured_attractor.png"), dpi=140)
    plt.close(fig)

    print(f"wrote symbolic_code_coloured_attractor.png and symbolic_code_diagnostics.txt")


if __name__ == "__main__":
    main()
