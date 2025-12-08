#!/usr/bin/env python3
import json
import math
import sys
import traceback

import numpy as np
from numpy.random import default_rng

# Try importing torch, fail with clear JSON if missing
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
except Exception as e:
    out = {"error": "PyTorch import failed", "exception": str(e)}
    print(json.dumps(out))
    raise

# Robust wrapper to ensure final JSON output even on exceptions
metrics = {}
try:
    # Reproducibility
    SEED = 42
    rng = default_rng(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    device = torch.device('cpu')

    # 1) Define 2D landscapes (mixtures)
    D = 2
    mus = np.array([[0.0, 0.0], [3.0, 2.0], [-2.0, 3.0]])
    w_samp = np.array([0.5, 0.3, 0.2])
    sigma_samp = np.array([0.6, 0.5, 0.8])
    w_bio = np.array([0.3, 0.5, 0.2])
    sigma_bio = np.array([0.8, 0.4, 0.6])

    def log_gaussian_component(x, mu, sigma):
        diff = x - mu
        sq = np.sum(diff * diff, axis=-1)
        C = -0.5 * D * np.log(2 * np.pi * (sigma ** 2))
        return C - 0.5 * sq / (sigma ** 2)

    def mixture_logpdf_and_components(x, mus_, weights, sigmas):
        # x: (N,2)
        N = x.shape[0]
        K = mus_.shape[0]
        comps = np.zeros((N, K))
        for k in range(K):
            comps[:, k] = np.log(weights[k] + 1e-20) + log_gaussian_component(x, mus_[k], sigmas[k])
        a = np.max(comps, axis=1)
        logpdf = a + np.log(np.sum(np.exp(comps - a[:, None]), axis=1))
        return logpdf, comps

    def score_mixture(x, mus_, weights, sigmas):
        # analytic grad log p(x) for isotropic Gaussian mixture
        N = x.shape[0]
        K = mus_.shape[0]
        comps = np.zeros((N, K))
        for k in range(K):
            diff = x - mus_[k]
            comps[:, k] = weights[k] * np.exp(-0.5 * np.sum(diff * diff, axis=1) / (sigmas[k] ** 2))
        denom = np.sum(comps, axis=1, keepdims=True) + 1e-20
        contrib = np.zeros((N, D))
        for k in range(K):
            contrib += ( - (x - mus_[k]) / (sigmas[k] ** 2) ) * comps[:, k:k+1]
        score = contrib / denom
        return score

    def sample_mixture(n, mus_, weights, sigmas_, rng_):
        K = mus_.shape[0]
        comp_choices = rng_.choice(K, size=n, p=weights)
        samples = np.zeros((n, D))
        for k in range(K):
            nk = np.sum(comp_choices == k)
            if nk > 0:
                samples[comp_choices == k] = rng_.normal(loc=mus_[k], scale=sigmas_[k], size=(nk, D))
        return samples

    # Biophysical force = grad log q(x)
    def force_bio(x):
        return score_mixture(x, mus, w_bio, sigma_bio)

    def score_samp(x):
        return score_mixture(x, mus, w_samp, sigma_samp)

    # 2) Generate dataset
    N_data = 1200
    X_data = sample_mixture(N_data, mus, w_samp, sigma_samp, rng)

    # 3) Train compact MLP to regress analytic score
    N_train = 2000
    X_train = np.vstack([
        X_data[rng.choice(N_data, size=N_train//2, replace=True)],
        rng.uniform(low=-6, high=6, size=(N_train//2, D))
    ])
    y_train = score_samp(X_train)

    class ArrayDataset(torch.utils.data.Dataset):
        def __init__(self, X, y):
            self.X = torch.tensor(X, dtype=torch.float32)
            self.y = torch.tensor(y, dtype=torch.float32)
        def __len__(self):
            return self.X.shape[0]
        def __getitem__(self, idx):
            return self.X[idx], self.y[idx]

    train_ds = ArrayDataset(X_train, y_train)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=128, shuffle=True)

    class ScoreNet(nn.Module):
        def __init__(self, dim=2, hidden=64):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(dim, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Linear(hidden, dim)
            )
        def forward(self, x):
            return self.net(x)

    score_model = ScoreNet(dim=D, hidden=64).to(device)
    optimizer = optim.Adam(score_model.parameters(), lr=2e-3)
    loss_fn = nn.MSELoss()

    n_epochs = 90
    for epoch in range(n_epochs):
        epoch_loss = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            pred = score_model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * xb.size(0)
        epoch_loss /= len(train_loader.dataset)
        if (epoch + 1) % 30 == 0:
            print(f"[Score] Epoch {epoch+1}/{n_epochs}, loss={epoch_loss:.6f}")

    # 4) Evaluate on a modest grid
    grid_size = 50
    xs = np.linspace(-6, 6, grid_size)
    ys = np.linspace(-1, 6, grid_size)
    xx, yy = np.meshgrid(xs, ys)
    grid = np.stack([xx.ravel(), yy.ravel()], axis=1)

    with torch.no_grad():
        pred_scores = score_model(torch.tensor(grid, dtype=torch.float32)).numpy()
    true_scores = score_samp(grid)
    bio_forces = force_bio(grid)

    def cosine_sim(a, b):
        dot = np.sum(a * b, axis=1)
        na = np.linalg.norm(a, axis=1)
        nb = np.linalg.norm(b, axis=1)
        denom = np.maximum(na * nb, 1e-12)
        return dot / denom

    sim_pred_vs_bio = cosine_sim(pred_scores, bio_forces)
    sim_true_vs_bio = cosine_sim(true_scores, bio_forces)

    # 5) Train a compact GAN
    class Generator(nn.Module):
        def __init__(self, zdim=2, out=2, hidden=64):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(zdim, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Linear(hidden, out)
            )
        def forward(self, z):
            return self.net(z)

    class Discriminator(nn.Module):
        def __init__(self, dim=2, hidden=64):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(dim, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Linear(hidden, 1),
                nn.Sigmoid()
            )
        def forward(self, x):
            return self.net(x)

    G = Generator().to(device)
    D = Discriminator().to(device)
    opt_G = optim.Adam(G.parameters(), lr=2e-3)
    opt_D = optim.Adam(D.parameters(), lr=2e-3)

    real_ds = torch.tensor(X_data, dtype=torch.float32)
    real_loader = torch.utils.data.DataLoader(real_ds, batch_size=128, shuffle=True)
    bce = nn.BCELoss()

    n_gan_steps = 600
    z_dim = 2
    step = 0
    for epoch in range(20):
        for real_batch in real_loader:
            real_batch = real_batch.to(device)
            bsize = real_batch.size(0)
            # D step
            opt_D.zero_grad()
            z = torch.randn(bsize, z_dim, device=device)
            fake = G(z).detach()
            d_real = D(real_batch)
            d_fake = D(fake)
            loss_D = bce(d_real, torch.ones_like(d_real)) + bce(d_fake, torch.zeros_like(d_fake))
            loss_D.backward()
            opt_D.step()
            # G step
            opt_G.zero_grad()
            z2 = torch.randn(bsize, z_dim, device=device)
            fake2 = G(z2)
            d_fake2 = D(fake2)
            loss_G = bce(d_fake2, torch.ones_like(d_fake2))
            loss_G.backward()
            opt_G.step()
            step += 1
            if step % 150 == 0:
                print(f"[GAN] training step {step}, loss_D={loss_D.item():.4f}, loss_G={loss_G.item():.4f}")
            if step >= n_gan_steps:
                break
        if step >= n_gan_steps:
            break

    # 6) Sampling: Langevin using learned score and GAN samples
    def langevin_sample(score_model, n_samples=1000, n_steps=100, step_size=0.05, rng_local=None):
        if rng_local is None:
            rng_local = rng
        x = rng_local.uniform(low=-6, high=6, size=(n_samples, D))
        x_t = torch.tensor(x, dtype=torch.float32)
        with torch.no_grad():
            for t in range(n_steps):
                s = score_model(x_t).numpy()
                noise = rng_local.normal(scale=math.sqrt(2 * step_size), size=x_t.shape)
                x = x_t.numpy() + step_size * s + noise
                x_t = torch.tensor(x, dtype=torch.float32)
        return x_t.numpy()

    score_samples = langevin_sample(score_model, n_samples=1500, n_steps=100, step_size=0.05)

    with torch.no_grad():
        z = torch.randn(1500, z_dim)
        gan_samples = G(z).numpy()

    # 7) KDE-based score for GAN samples (use subset for speed)
    def kde_score_at_points(points, samples, bandwidth=0.6):
        M = points.shape[0]
        N = samples.shape[0]
        diffs = points[:, None, :] - samples[None, :, :]  # (M,N,2)
        sq = np.sum(diffs * diffs, axis=2)
        K = np.exp(-0.5 * sq / (bandwidth ** 2))
        denom = np.sum(K, axis=1, keepdims=True) + 1e-20
        num = - np.einsum('ij,ijk->ik', K, diffs)
        score = num / ((bandwidth ** 2) * denom)
        return score

    # use at most 1000 GAN samples in KDE to limit cost
    gan_for_kde = gan_samples if gan_samples.shape[0] <= 1000 else gan_samples[:1000]

    kde_bandwidth = 0.6
    kde_scores_gan = kde_score_at_points(grid, gan_for_kde, bandwidth=kde_bandwidth)
    sim_kde_vs_bio = cosine_sim(kde_scores_gan, bio_forces)

    # 8) Energies under biophysical model
    def energy_bio(x):
        logpdf, _ = mixture_logpdf_and_components(x, mus, w_bio, sigma_bio)
        return -logpdf

    E_data = energy_bio(X_data)
    E_score = energy_bio(score_samples)
    E_gan = energy_bio(gan_samples)

    mean_E_data = float(np.mean(E_data))
    mean_E_score = float(np.mean(E_score))
    mean_E_gan = float(np.mean(E_gan))
    se_E_data = float(np.std(E_data) / math.sqrt(len(E_data)))
    se_E_score = float(np.std(E_score) / math.sqrt(len(E_score)))
    se_E_gan = float(np.std(E_gan) / math.sqrt(len(E_gan)))

    # 9) Permutation test (faster)
    def permutation_test(a, b, n_perm=500):
        rng_local = default_rng(SEED + 1)
        diff_obs = np.mean(a) - np.mean(b)
        pooled = np.concatenate([a, b])
        count = 0
        for _ in range(n_perm):
            rng_local.shuffle(pooled)
            a_s = pooled[:len(a)]
            b_s = pooled[len(a):]
            if abs(np.mean(a_s) - np.mean(b_s)) >= abs(diff_obs):
                count += 1
        p = (count + 1) / (n_perm + 1)
        return p

    p_energy = permutation_test(E_score, E_gan, n_perm=500)

    # 10) Aggregate metrics
    metrics.update({
        "alignment_score_model_vs_bio_mean": float(np.mean(sim_pred_vs_bio)),
        "alignment_true_score_vs_bio_mean": float(np.mean(sim_true_vs_bio)),
        "alignment_kde_gan_vs_bio_mean": float(np.mean(sim_kde_vs_bio)),
        "mean_energy_data": mean_E_data,
        "mean_energy_score_samples": mean_E_score,
        "mean_energy_gan_samples": mean_E_gan,
        "se_energy_data": se_E_data,
        "se_energy_score_samples": se_E_score,
        "se_energy_gan_samples": se_E_gan,
        "p_value_energy_score_vs_gan": float(p_energy),
        "n_data": int(N_data),
        "n_score_samples": int(score_samples.shape[0]),
        "n_gan_samples": int(gan_samples.shape[0])
    })

except Exception as e:
    # Capture exception and include in metrics
    metrics.update({"error": str(e), "traceback": traceback.format_exc()})

# Ensure final JSON printed
print(json.dumps(metrics))
