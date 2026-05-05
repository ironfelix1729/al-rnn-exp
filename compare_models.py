# -*- coding: utf-8 -*-
"""
Experiment: Compare Vanilla vs Orthogonal AL-RNN across hidden sizes M and P.

Hypothesis: Ortho model requires smaller M to achieve comparable performance
with vanilla, across varying P (number of piecewise-linear units).
"""

import numpy as np
import json
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from dataset import TimeSeriesDataset

# ---- Data ----------------------------------------------------------------

X_train = np.load("lorenz63_train.npy").astype(np.float32)[500:]
X_test  = np.load("lorenz63_test.npy").astype(np.float32)[500:]
N = X_train.shape[1]  # observation dim = 3

def random_orthogonal_matrix(d, seed=None):
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((d, d))
    Q, R = np.linalg.qr(A)
    signs = np.sign(np.diag(R)); signs[signs == 0] = 1.0
    return (Q * signs).astype(np.float32)

Q_dataset = random_orthogonal_matrix(N, seed=42)
X_train_rot = X_train @ Q_dataset.T
X_test_rot  = X_test  @ Q_dataset.T

# ---- Models --------------------------------------------------------------

class AL_RNN_Vanilla(nn.Module):
    def __init__(self, M, P, N):
        super().__init__()
        self.M, self.P, self.N = M, P, N
        self.A = nn.Parameter(self._init_A())
        self.W = nn.Parameter(torch.randn(M, M) * 0.01)
        self.h = nn.Parameter(torch.zeros(M))
        self.B = nn.Parameter(self._init_uniform((N, M)))

    def _init_A(self):
        R = np.random.randn(self.M, self.M).astype(np.float32)
        K = (R.T @ R) / self.M + np.eye(self.M, dtype=np.float32)
        return torch.from_numpy(np.diag(K / np.max(np.abs(np.linalg.eigvals(K)))).astype(np.float32))

    def _init_uniform(self, shape):
        r = 1.0 / math.sqrt(shape[0])
        return nn.init.uniform_(torch.empty(*shape), -r, r)

    def forward(self, z):
        z_relu = z.clone()
        z_relu[:, -self.P:] = F.relu(z_relu[:, -self.P:])
        return self.A * z + z_relu @ self.W.t() + self.h


class AL_RNN_Ortho(nn.Module):
    def __init__(self, M, P, N):
        super().__init__()
        self.M, self.P, self.N = M, P, N
        self.A = nn.Parameter(self._init_A())
        self.W = nn.Parameter(torch.randn(M, M) * 0.01)
        self.h = nn.Parameter(torch.zeros(M))
        self.B = nn.Parameter(self._init_uniform((N, M)))
        self.transform = nn.Linear(M, M)
        torch.nn.utils.parametrizations.orthogonal(self.transform)

    def _init_A(self):
        R = np.random.randn(self.M, self.M).astype(np.float32)
        K = (R.T @ R) / self.M + np.eye(self.M, dtype=np.float32)
        return torch.from_numpy(np.diag(K / np.max(np.abs(np.linalg.eigvals(K)))).astype(np.float32))

    def _init_uniform(self, shape):
        r = 1.0 / math.sqrt(shape[0])
        return nn.init.uniform_(torch.empty(*shape), -r, r)

    def forward(self, z):
        z_trans = self.transform(z)
        z_relu = z_trans.clone()
        z_relu[:, -self.P:] = F.relu(z_relu[:, -self.P:])
        return self.A * z + z_relu @ self.W.t() + self.h


# ---- Training helpers ----------------------------------------------------

def teacher_force_(z, x, alpha, n_obs):
    z[:, :n_obs].mul_(1.0 - alpha).add_(alpha * x)
    return z

def predict_gtf(model, x, alpha, n_interleave):
    b, T, _ = x.shape
    Z = torch.empty((b, T, model.M))
    z = x[:, 0, :] @ model.B
    teacher_force_(z, x[:, 0, :], 1.0, model.N)
    for t in range(T):
        if t > 0 and (t % n_interleave == 0):
            teacher_force_(z, x[:, t, :], alpha, model.N)
        z = model(z)
        Z[:, t, :] = z
    return Z

def train(model, data, num_epochs=600, lr=1e-3, seq_len=128, batch_size=32,
          batches_per_epoch=20, alpha=1.0, n_interleave=16):
    dataset = TimeSeriesDataset(data, sequence_length=seq_len, batch_size=batch_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    gamma = np.exp(np.log(1e-5 / lr) / num_epochs)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    loss_fn = nn.MSELoss()
    losses = []
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for _ in range(batches_per_epoch):
            optimizer.zero_grad(set_to_none=True)
            x, y, _ = dataset.sample_batch()
            z_hat = predict_gtf(model, x, alpha, n_interleave)
            loss = loss_fn(z_hat[:, :, :model.N], y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        scheduler.step()
        losses.append(epoch_loss / batches_per_epoch)
    return losses


# ---- Experiment grid -----------------------------------------------------

M_values = [10, 15, 20, 25, 30, 40, 50]
P_values = [2, 5, 10]       # P must be <= M; pairs where P > M are skipped
SEEDS    = [0, 1, 2]
NUM_EPOCHS = 600
TAIL     = 60   # average last TAIL epochs as "final loss"

results = {}   # {model_name: {(M, P): [seed_losses...]}}

total = sum(1 for M in M_values for P in P_values if P <= M) * len(SEEDS) * 2
done = 0

for model_name, ModelClass in [("vanilla", AL_RNN_Vanilla), ("ortho", AL_RNN_Ortho)]:
    results[model_name] = {}
    for M in M_values:
        for P in P_values:
            if P > M:
                continue
            key = f"{M},{P}"
            seed_losses = []
            for seed in SEEDS:
                done += 1
                print(f"[{done}/{total}] {model_name} M={M} P={P} seed={seed}")
                torch.manual_seed(seed)
                np.random.seed(seed)
                model = ModelClass(M=M, P=P, N=N)
                losses = train(model, X_train_rot, num_epochs=NUM_EPOCHS)
                final = float(np.mean(losses[-TAIL:]))
                seed_losses.append(final)
                print(f"  => {final:.6f}")
            results[model_name][key] = seed_losses

with open("results_compare.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nResults saved to results_compare.json")


# ---- Analysis ------------------------------------------------------------

print("\n\n=== RESULTS: final training loss (mean ± std over seeds) ===\n")

for P in P_values:
    print(f"\n--- P = {P} ---")
    print(f"{'M':>4}  {'Vanilla':>20}  {'Ortho':>20}  {'Ortho/Vanilla':>14}")
    print("-" * 64)
    for M in M_values:
        if P > M:
            continue
        key = f"{M},{P}"
        v = np.array(results["vanilla"][key])
        o = np.array(results["ortho"][key])
        ratio = o.mean() / v.mean()
        print(f"{M:>4}  {v.mean():.5f} ± {v.std():.5f}  "
              f"  {o.mean():.5f} ± {o.std():.5f}  "
              f"  {ratio:>10.3f}x")

# Hypothesis test: for each P, find smallest M_ortho that matches vanilla at M_ref
M_ref = 30
print(f"\n\n=== HYPOTHESIS TEST ===")
print(f"Reference: vanilla at M={M_ref}.")
print(f"Question: what is the smallest M where ortho matches that performance?\n")

for P in P_values:
    if P > M_ref:
        continue
    v_ref_key = f"{M_ref},{P}"
    if v_ref_key not in results["vanilla"]:
        continue
    v_ref = np.mean(results["vanilla"][v_ref_key])
    print(f"P={P}: vanilla(M={M_ref}) loss = {v_ref:.6f}")

    matched = False
    for M in M_values:
        if P > M:
            continue
        key = f"{M},{P}"
        o = np.mean(results["ortho"][key])
        if o <= v_ref:
            print(f"  -> ortho matches at M={M} (loss={o:.6f})  [{M}/{M_ref} = {M/M_ref:.2f}x]")
            matched = True
            break
    if not matched:
        # find closest M
        diffs = {}
        for M in M_values:
            if P > M:
                continue
            key = f"{M},{P}"
            diffs[M] = np.mean(results["ortho"][key])
        best_M = min(diffs, key=diffs.get)
        print(f"  -> ortho never quite matches; best at M={best_M} (loss={diffs[best_M]:.6f})")

print("\nDone.")
