# -*- coding: utf-8 -*-
"""P-sweep experiment: vanilla vs orthogonal AL-RNN on rotated Lorenz63.

Trains both models for P in {2,3,4,5,6} at 1500 epochs each, generates
free-run trajectories, and analyses the learned orthogonal matrix Q.

Speed-ups vs al_rnn_22_04.py:
  - max intra-op threads
  - model.forward uses torch.cat instead of clone+assign (one fewer copy)
  - W.T cached per sequence
  - teacher_force_ reduced to a single fused op
  - larger batch (64) to amortise Python/op dispatch overhead
"""
import os, math, time, json, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from dataset import TimeSeriesDataset

torch.set_num_threads(max(1, os.cpu_count() or 1))
torch.set_num_interop_threads(max(1, os.cpu_count() or 1))

DEVICE = torch.device("cpu")
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_p_sweep")
os.makedirs(OUT_DIR, exist_ok=True)


# --- Utilities ---------------------------------------------------------------
def random_orthogonal_matrix(d, seed=None):
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((d, d))
    Q, R = np.linalg.qr(A)
    signs = np.sign(np.diag(R))
    signs[signs == 0] = 1.0
    Q = Q * signs
    return Q.astype(np.float32)


def apply_rotation(X, Q):
    return (X @ Q.T).astype(np.float32)


# --- Models ------------------------------------------------------------------
class AL_RNN_Original(nn.Module):
    def __init__(self, M, P, N, B_shared=None, freeze_B=False):
        super().__init__()
        self.M, self.P, self.N = M, P, N
        self.A = nn.Parameter(self._init_A())
        self.W = nn.Parameter(torch.randn(M, M) * 0.01)
        self.h = nn.Parameter(torch.zeros(M))
        if B_shared is None:
            self.B = nn.Parameter(self._init_uniform((N, M)))
        else:
            assert B_shared.shape == (N, M), f"expected B_shared shape {(N, M)}, got {tuple(B_shared.shape)}"
            self.B = nn.Parameter(B_shared.detach().clone())
            if freeze_B:
                self.B.requires_grad_(False)
        self._W_T = None  # per-sequence cache

    def _init_A(self):
        R = np.random.randn(self.M, self.M).astype(np.float32)
        K = (R.T @ R) / self.M + np.eye(self.M, dtype=np.float32)
        return torch.from_numpy(np.diag(K / np.max(np.abs(np.linalg.eigvals(K)))).astype(np.float32))

    def _init_uniform(self, shape):
        r = 1.0 / math.sqrt(shape[0])
        return nn.init.uniform_(torch.empty(*shape), -r, r)

    def prepare_sequence(self):
        self._W_T = self.W.t()

    def forward(self, z):
        P = self.P
        if P > 0:
            z_nl = torch.cat([z[:, :-P], F.relu(z[:, -P:])], dim=1)
        else:
            z_nl = z
        W_T = self._W_T if self._W_T is not None else self.W.t()
        return self.A * z + z_nl @ W_T + self.h


class AL_RNN_Orthogonal(nn.Module):
    def __init__(self, M, P, N, B_shared=None, freeze_B=False):
        super().__init__()
        self.M, self.P, self.N = M, P, N
        self.A = nn.Parameter(self._init_A())
        self.W = nn.Parameter(torch.randn(M, M) * 0.01)
        self.h = nn.Parameter(torch.zeros(M))
        if B_shared is None:
            self.B = nn.Parameter(self._init_uniform((N, M)))
        else:
            assert B_shared.shape == (N, M), f"expected B_shared shape {(N, M)}, got {tuple(B_shared.shape)}"
            self.B = nn.Parameter(B_shared.detach().clone())
            if freeze_B:
                self.B.requires_grad_(False)
        self.transform = nn.Linear(M, M)
        torch.nn.utils.parametrizations.orthogonal(self.transform)
        self._W_T = None
        self._Q_T = None
        self._qb = None

    def _init_A(self):
        R = np.random.randn(self.M, self.M).astype(np.float32)
        K = (R.T @ R) / self.M + np.eye(self.M, dtype=np.float32)
        return torch.from_numpy(np.diag(K / np.max(np.abs(np.linalg.eigvals(K)))).astype(np.float32))

    def _init_uniform(self, shape):
        r = 1.0 / math.sqrt(shape[0])
        return nn.init.uniform_(torch.empty(*shape), -r, r)

    def prepare_sequence(self):
        # materialize orthogonal Q ONCE per sequence; autograd still flows
        # through the parametrization because we keep the tensor, not a copy
        self._W_T = self.W.t()
        Q = self.transform.weight
        self._Q_T = Q.t()
        self._qb = self.transform.bias

    def forward(self, z):
        if self._Q_T is None:
            z_trans = self.transform(z)
            W_T = self.W.t()
        else:
            z_trans = z @ self._Q_T + self._qb
            W_T = self._W_T
        P = self.P
        if P > 0:
            z_relu = torch.cat([z_trans[:, :-P], F.relu(z_trans[:, -P:])], dim=1)
        else:
            z_relu = z_trans
        return self.A * z + z_relu @ W_T + self.h


# --- Training ----------------------------------------------------------------
def teacher_force_(z, x, alpha, n_obs):
    z[:, :n_obs].mul_(1.0 - alpha).add_(alpha * x)
    return z


def predict_sequence_using_gtf(model, x, alpha, n_interleave):
    b, T, dx = x.shape
    device = x.device
    Z = torch.empty((b, T, model.M), device=device)
    z = x[:, 0, :] @ model.B
    teacher_force_(z, x[:, 0, :], 1.0, model.N)
    model.prepare_sequence()
    for t in range(T):
        if t > 0 and (t % n_interleave == 0):
            teacher_force_(z, x[:, t, :], alpha, model.N)
        z = model(z)
        Z[:, t, :] = z
    return Z


def train_fast(model, dataset, optimizer, scheduler, loss_fn, num_epochs,
               alpha, n_interleave, batches_per_epoch=20, device="cpu",
               log_every=100, tag=""):
    model.to(device)
    model.train()
    losses = []
    t0 = time.time()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for _ in range(batches_per_epoch):
            optimizer.zero_grad(set_to_none=True)
            x, y, s = dataset.sample_batch()
            x, y = x.to(device), y.to(device)
            z_hat = predict_sequence_using_gtf(model, x, alpha, n_interleave)
            loss = loss_fn(z_hat[:, :, :model.N], y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        scheduler.step()
        epoch_loss /= batches_per_epoch
        losses.append(epoch_loss)
        if epoch % log_every == 0 or epoch == num_epochs - 1:
            dt = time.time() - t0
            print(f"[{tag}] epoch {epoch:4d}/{num_epochs}  loss={epoch_loss:.6f}  elapsed={dt:.1f}s", flush=True)
    return losses


def train_model_on_data(model_class, data, M, P, num_epochs, batch_size=64,
                        device="cpu", tag="", B_shared=None, freeze_B=False):
    dataset = TimeSeriesDataset(data, sequence_length=128, batch_size=batch_size)
    model = model_class(M=M, P=P, N=data.shape[-1],
                        B_shared=B_shared, freeze_B=freeze_B).to(device)
    # only optimise params that require grad (B may be frozen)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable, lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer, gamma=np.exp(np.log(1e-5 / 1e-3) / num_epochs)
    )
    losses = train_fast(model, dataset, optimizer, scheduler, nn.MSELoss(),
                        num_epochs, 1.0, 16, batches_per_epoch=20,
                        device=device, tag=tag)
    return model, losses


# --- Free-run generation -----------------------------------------------------
@torch.inference_mode()
def predict_free_sequence(model, x0, T):
    b, N = x0.shape
    device = x0.device
    Z = torch.empty((b, T, model.M), device=device)
    z = x0 @ model.B
    z[:, :N] = x0
    model.prepare_sequence()
    for t in range(T):
        z = model(z)
        Z[:, t, :] = z
    return Z


@torch.inference_mode()
def generate_orbit(model, x0, T_gen, T_r=0, device="cpu"):
    x0_t = torch.tensor(x0[None, :], dtype=torch.float32, device=device)
    orbit = predict_free_sequence(model, x0_t, T_gen + T_r).cpu().numpy()[0]
    if T_r > 0:
        orbit = orbit[T_r:]
    return orbit


# --- Q analysis --------------------------------------------------------------
def analyze_Q(model_ortho):
    """Return diagnostics for the learned orthogonal matrix of the model.

    Does NOT include the raw matrix.
    """
    Q = model_ortho.transform.weight.detach().cpu().numpy()
    M = Q.shape[0]
    I = np.eye(M)

    # orthogonality residual
    ortho_err = np.linalg.norm(Q.T @ Q - I, ord="fro")
    det = float(np.linalg.det(Q))

    # eigenvalues live on unit circle; rotation angles = arg(eig)
    eigs = np.linalg.eigvals(Q)
    moduli = np.abs(eigs)
    angles_deg = np.sort(np.degrees(np.angle(eigs)))

    # distance from identity / reflections / permutations
    dist_I = float(np.linalg.norm(Q - I, ord="fro"))
    # closest signed permutation (pure combinatorial rotation)
    # heuristic: max |Q_ij| per row, then how close abs(Q) is to that one-hot
    absQ = np.abs(Q)
    row_max = absQ.max(axis=1)
    perm_residual = float(np.linalg.norm(absQ - np.diag(row_max) @ (absQ == row_max[:, None]).astype(np.float32), ord="fro"))

    # how "diffuse" is Q: a permutation would have 1 nonzero per row, a
    # dense rotation mixes many coords -> effective participation ratio
    row_l2 = np.sqrt((Q ** 2).sum(axis=1))  # all ~1 for orthogonal
    row_l1 = np.abs(Q).sum(axis=1)
    row_participation = (row_l1 / row_l2) ** 2  # 1 for 1-hot, M for uniform
    mean_participation = float(row_participation.mean())

    # complex eigenvalue pairs (nontrivial rotation planes)
    complex_pairs = int(np.sum(np.abs(eigs.imag) > 1e-3) // 2)
    real_pm1 = int(np.sum(np.abs(eigs.imag) <= 1e-3))

    # singular-value spread (should all be 1 exactly for orthogonal)
    svs = np.linalg.svd(Q, compute_uv=False)
    sv_spread = float(svs.max() - svs.min())

    return {
        "M": M,
        "orthogonality_error_frob": float(ortho_err),
        "determinant": det,
        "distance_from_identity_frob": dist_I,
        "permutation_residual_frob": perm_residual,
        "mean_row_participation": mean_participation,   # 1=permutation, M=uniform mix
        "num_complex_eig_pairs": complex_pairs,
        "num_real_pm1_eigs": real_pm1,
        "rotation_angles_deg_sorted": angles_deg.tolist(),
        "singular_value_spread": sv_spread,
    }


def weight_summary(model, label):
    stats = {"label": label}
    for name, p in model.named_parameters():
        w = p.detach().cpu().numpy()
        stats[name] = {
            "shape": list(w.shape),
            "mean": float(w.mean()),
            "std": float(w.std()),
            "absmax": float(np.abs(w).max()),
            "frob": float(np.linalg.norm(w)),
        }
    return stats


# --- Main --------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=1500)
    parser.add_argument("--P_list", type=int, nargs="+", default=[2, 3, 4, 5, 6])
    parser.add_argument("--M", type=int, default=20)
    parser.add_argument("--T_gen", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tag", type=str, default="main")
    parser.add_argument("--share_B", action="store_true",
                        help="use one B matrix in all 10 runs")
    parser.add_argument("--freeze_B", action="store_true",
                        help="if --share_B, also freeze B (requires_grad=False)")
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    X_train = np.load("lorenz63_train.npy").astype(np.float32)[500:]
    X_test = np.load("lorenz63_test.npy").astype(np.float32)[500:]
    N = X_train.shape[1]

    Q_dataset = random_orthogonal_matrix(N, seed=args.seed)
    X_train_trans = apply_rotation(X_train, Q_dataset)
    X_test_trans = apply_rotation(X_test, Q_dataset)

    # save the dataset rotation for the record
    np.save(os.path.join(OUT_DIR, f"Q_dataset_{args.tag}.npy"), Q_dataset)

    # optionally build a single B used by every run in the sweep
    shared_B = None
    if args.share_B:
        g = torch.Generator().manual_seed(args.seed + 12345)
        r = 1.0 / math.sqrt(N)
        shared_B = torch.empty(N, args.M).uniform_(-r, r, generator=g)
        np.save(os.path.join(OUT_DIR, f"shared_B_{args.tag}.npy"),
                shared_B.numpy())
        print(f"Shared B: shape={tuple(shared_B.shape)}  frozen={args.freeze_B}  "
              f"||B||_F={float(shared_B.norm()):.3f}", flush=True)

    summary = {
        "epochs": args.epochs,
        "M": args.M,
        "P_list": args.P_list,
        "share_B": bool(args.share_B),
        "freeze_B": bool(args.freeze_B),
        "per_P": {},
    }

    t_start = time.time()
    for P in args.P_list:
        print(f"\n================  P = {P}  ================", flush=True)

        t0 = time.time()
        model_orig, losses_orig = train_model_on_data(
            AL_RNN_Original, X_train_trans, M=args.M, P=P,
            num_epochs=args.epochs, device=DEVICE, tag=f"P={P}/orig",
            B_shared=shared_B, freeze_B=args.freeze_B,
        )
        dt_orig = time.time() - t0

        t0 = time.time()
        model_ortho, losses_ortho = train_model_on_data(
            AL_RNN_Orthogonal, X_train_trans, M=args.M, P=P,
            num_epochs=args.epochs, device=DEVICE, tag=f"P={P}/ortho",
            B_shared=shared_B, freeze_B=args.freeze_B,
        )
        dt_ortho = time.time() - t0

        orbit_orig = generate_orbit(model_orig, X_test_trans[0], T_gen=args.T_gen, device=DEVICE)
        orbit_ortho = generate_orbit(model_ortho, X_test_trans[0], T_gen=args.T_gen, device=DEVICE)

        Q_info = analyze_Q(model_ortho)
        w_orig = weight_summary(model_orig, "vanilla")
        w_ortho = weight_summary(model_ortho, "orthogonal")

        # persist  (tag paths so runs with different M/tag do not clobber each other)
        suffix = f"_{args.tag}" if args.tag and args.tag != "main" else ""
        torch.save(model_orig.state_dict(), os.path.join(OUT_DIR, f"model_orig_P{P}{suffix}.pt"))
        torch.save(model_ortho.state_dict(), os.path.join(OUT_DIR, f"model_ortho_P{P}{suffix}.pt"))
        np.savez(
            os.path.join(OUT_DIR, f"P{P}{suffix}_arrays.npz"),
            losses_orig=np.asarray(losses_orig),
            losses_ortho=np.asarray(losses_ortho),
            orbit_orig=orbit_orig,
            orbit_ortho=orbit_ortho,
            X_test_trans=X_test_trans[: args.T_gen],
        )

        summary["per_P"][str(P)] = {
            "final_loss_orig": float(losses_orig[-1]),
            "final_loss_ortho": float(losses_ortho[-1]),
            "min_loss_orig": float(min(losses_orig)),
            "min_loss_ortho": float(min(losses_ortho)),
            "train_seconds_orig": dt_orig,
            "train_seconds_ortho": dt_ortho,
            "Q_analysis": Q_info,
            "weight_summary_orig": w_orig,
            "weight_summary_ortho": w_ortho,
        }
        with open(os.path.join(OUT_DIR, f"summary_{args.tag}.json"), "w") as f:
            json.dump(summary, f, indent=2)

    total = time.time() - t_start
    print(f"\nAll done in {total/60:.1f} min")
    with open(os.path.join(OUT_DIR, f"summary_{args.tag}.json"), "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
