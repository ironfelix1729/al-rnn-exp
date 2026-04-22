# -*- coding: utf-8 -*-
"""Compare the learned orthogonal Q (M=20) to the dataset rotation Q_dataset (N=3).

The two matrices live in different spaces.  The interpretable link is:
  * observations x occupy the first N=3 coordinates of z (teacher forcing +
    readout z_hat = z[:N]).
  * so the 3x3 top-left block  Q[:N, :N]  captures how the learned
    transform acts on the observation subspace.
  * equivalently, via the readout projector P_N = [I_N, 0]:
        Q_obs = P_N Q P_N^T  in R^{NxN}.
  * for the vanilla model there is no Q; the analogue is the identity
    (the model sees raw state directly).

We also compute the polar decomposition of Q_obs to extract its rotational
core, and its alignment with Q_dataset and Q_dataset^T.
"""
import os, json
import numpy as np
import torch
import matplotlib.pyplot as plt

from experiment_p_sweep import AL_RNN_Orthogonal

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_p_sweep")
M, N = 20, 3


def polar_rotational_part(A):
    """Return R in the polar decomposition A = R S, so R is orthogonal."""
    U, _, Vt = np.linalg.svd(A)
    R = U @ Vt
    # scale flag
    return R


def principal_angles(Q1, Q2):
    """Principal angles between the column spaces of two NxN matrices."""
    # use SVD of Q1.T Q2 for the full picture
    U, s, Vt = np.linalg.svd(Q1.T @ Q2)
    s = np.clip(s, -1.0, 1.0)
    return np.degrees(np.arccos(s))


def rotation_angle(R):
    """For a proper rotation R in R^3, angle = arccos((tr R - 1)/2)."""
    c = (np.trace(R) - 1.0) / 2.0
    c = float(np.clip(c, -1.0, 1.0))
    return float(np.degrees(np.arccos(c)))


def load_Q(P, N_obs=3):
    model = AL_RNN_Orthogonal(M=M, P=P, N=N_obs)
    sd = torch.load(os.path.join(OUT, f"model_ortho_P{P}.pt"), map_location="cpu", weights_only=True)
    model.load_state_dict(sd, strict=False)
    model.eval()
    # force parametrization to realise the weight
    Q = model.transform.weight.detach().cpu().numpy()
    b = model.transform.bias.detach().cpu().numpy()
    return Q, b


def analyse(P, Q_dataset):
    Q, b = load_Q(P)
    Q_obs = Q[:N, :N]            # top-left NxN block
    # polar-rotational part of the top-left block (orthogonal part)
    R_obs = polar_rotational_part(Q_obs)
    # compare to Q_dataset and its transpose
    dist_to_Qd = float(np.linalg.norm(R_obs - Q_dataset))
    dist_to_QdT = float(np.linalg.norm(R_obs - Q_dataset.T))
    dist_to_I = float(np.linalg.norm(R_obs - np.eye(N)))

    # principal angles of the full Q action restricted to the observation
    # subspace (span of e1..eN).  Row-wise: how much of row i of Q lies in
    # the first N coords?
    obs_leakage = np.linalg.norm(Q[N:, :N])  # mass from obs subspace -> hidden
    obs_inleak = np.linalg.norm(Q[:N, N:])   # mass from hidden -> obs
    obs_block_frob = float(np.linalg.norm(Q_obs))
    obs_block_sv = np.linalg.svd(Q_obs, compute_uv=False)

    # rotation angle of the polar-rotational part
    rot_angle = rotation_angle(R_obs)
    rot_angle_Qd = rotation_angle(Q_dataset)

    # relative rotation between R_obs and Q_dataset
    rel = R_obs @ Q_dataset.T
    # force sign: if det<0, it's a reflection; report angle on R_obs Qd^T anyway
    rel_angle = rotation_angle(polar_rotational_part(rel))

    return {
        "P": P,
        "||Q_obs - Q_dataset||": dist_to_Qd,
        "||Q_obs - Q_dataset^T||": dist_to_QdT,
        "||Q_obs - I||": dist_to_I,
        "||Q_obs||_F (<= sqrt(3))": obs_block_frob,
        "obs_block_singular_values": obs_block_sv.tolist(),
        "||Q[:3,3:]||_F mass_out": float(obs_inleak),
        "||Q[3:,:3]||_F mass_in": float(obs_leakage),
        "R_obs_rotation_angle_deg": rot_angle,
        "Q_dataset_rotation_angle_deg": rot_angle_Qd,
        "relative_rotation_R_obs_Qd^T_deg": rel_angle,
        "det_Q_obs": float(np.linalg.det(Q_obs)),
        "det_R_obs": float(np.linalg.det(R_obs)),
    }


def main():
    Q_dataset = np.load(os.path.join(OUT, "Q_dataset_main.npy")).astype(np.float64)
    print(f"Q_dataset is 3x3 orthogonal, det={np.linalg.det(Q_dataset):+.3f}, "
          f"rotation angle={rotation_angle(Q_dataset):.2f} deg")

    rows = []
    for P in [2, 3, 4, 5, 6]:
        rows.append(analyse(P, Q_dataset))

    # text table
    lines = []
    add = lines.append
    add("Learned Q (20x20) vs dataset rotation Q_dataset (3x3)")
    add("=" * 80)
    add(f"Q_dataset rotation angle: {rotation_angle(Q_dataset):.2f} deg; "
        f"det={np.linalg.det(Q_dataset):+.3f}")
    add("")
    add("Quantities below refer to Q_obs = top-left 3x3 block of the learned Q,")
    add("and R_obs = its polar-rotational part (closest orthogonal to Q_obs).")
    add("If Q were block-diagonal in R^3 x R^17, ||Q_obs||_F would equal sqrt(3)=1.73.")
    add("")
    hdr = f"{'P':>3} {'||Qobs-Qd||':>12} {'||Qobs-Qd^T||':>14} {'||Qobs-I||':>11} " \
          f"{'||Qobs||_F':>10} {'SVs (sorted)':>28} {'||leak in||':>10} {'||leak out||':>11} " \
          f"{'rot(Robs)':>9} {'rel(Robs,Qd)':>12} {'det(Qobs)':>10}"
    add(hdr)
    for r in rows:
        svs = ",".join(f"{s:.2f}" for s in r["obs_block_singular_values"])
        add(f"{r['P']:>3} {r['||Q_obs - Q_dataset||']:>12.3f} "
            f"{r['||Q_obs - Q_dataset^T||']:>14.3f} "
            f"{r['||Q_obs - I||']:>11.3f} "
            f"{r['||Q_obs||_F (<= sqrt(3))']:>10.3f} "
            f"{svs:>28} "
            f"{r['||Q[3:,:3]||_F mass_in']:>10.3f} "
            f"{r['||Q[:3,3:]||_F mass_out']:>11.3f} "
            f"{r['R_obs_rotation_angle_deg']:>9.1f} "
            f"{r['relative_rotation_R_obs_Qd^T_deg']:>12.1f} "
            f"{r['det_Q_obs']:>10.3f}")

    txt = "\n".join(lines)
    print(txt)
    with open(os.path.join(OUT, "q_vs_qdataset.txt"), "w") as f:
        f.write(txt + "\n")
    with open(os.path.join(OUT, "q_vs_qdataset.json"), "w") as f:
        json.dump(rows, f, indent=2)

    # visualise the 3x3 observation blocks vs Q_dataset
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    vmax = 1.1
    im = axes[0, 0].imshow(Q_dataset, vmin=-vmax, vmax=vmax, cmap="RdBu_r")
    axes[0, 0].set_title(f"Q_dataset (3x3, rot {rotation_angle(Q_dataset):.0f}°)")
    axes[0, 0].set_xticks([]); axes[0, 0].set_yticks([])
    plt.colorbar(im, ax=axes[0, 0], fraction=0.046)
    # (0,1) blank
    axes[0, 1].axis("off")
    axes[0, 2].axis("off")
    for ax, P in zip(axes[1], [2, 4, 6]):
        Q, _ = load_Q(P)
        im = ax.imshow(Q[:N, :N], vmin=-vmax, vmax=vmax, cmap="RdBu_r")
        ax.set_title(f"Q_obs  (P={P})")
        ax.set_xticks([]); ax.set_yticks([])
        plt.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle("Dataset rotation vs top-left 3x3 block of learned Q")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "q_vs_qdataset.png"), dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    main()
