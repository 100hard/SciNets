#!/usr/bin/env python3
import json
import math
import sys
import time
import traceback

import numpy as np
from numpy.random import default_rng

# Try importing torch; if missing, we'll stop with a clear message
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
except Exception as e:
    print(json.dumps({
        "error": "PyTorch is required but could not be imported",
        "exception": str(e),
    }))
    raise

# Seed for reproducibility
SEED = 42
rng = default_rng(SEED)
torch.manual_seed(SEED)
np.random.seed(SEED)

device = torch.device('cpu')

# 1) Define two related landscapes in 2D: sampling distribution p_samp (mixture A) and biophysical energy q (mixture B).
D = 2
components = 3
mus = np.array([[0.0, 0.0], [3.0, 2.0], [-2.0, 3.0]])
# mixture weights for sampling distribution
w_samp = np.array([0.5, 0.3, 0.2])
# variances (isotropic) for sampling distribution
sigma_samp = np.array([0.6, 0.5, 0.8])

# biophysical energy distribution (different weights and widths)
w_bio = np.array([0.3, 0.5, 0.2])
sigma_bio = np.array([0.8, 0.4, 0.6])

# helper functions

def log_gaussian_component(x, mu, sigma):
    # x: (...,2), mu: (2,), sigma: scalar
    diff = x - mu
    sq = np.sum(diff * diff, axis=-1)
    C = -0.5 * D * np.log(2 * np.pi * (sigma ** 2))
    return C - 0.5 * sq / (sigma ** 2)


def mixture_logpdf_and_components(x, mus, weights, sigmas):
    # x: (N,2)
    N = x.shape[0]
    K = mus.shape[0]
    comps = np.zeros((N, K))
    for k in range(K):
        comps[:, k] = np.log(weights[k] + 1e-20) + log_gaussian_component(x, mus[k], sigmas[k])
    # log-sum-exp
    a = np.max(comps, axis=1)
    logpdf = a + np.log(np.sum(np.exp(comps - a[:, None]), axis=1))
    return logpdf, comps


def score_mixture(x, mus, weights, sigmas):
    # analytic grad log p(x) for isotropic Gaussian mixture
    # x: (N,2)
    N = x.shape[0]
    K = mus.shape[0]
    # compute unnormalized component densities
    comps = np.zeros((N, K))
    for k in range(K):
        comps[:, k] = weights[k] * np.exp(-0.5 * np.sum((x - mus[k]) ** 2, axis=1) / (sigmas[k] ** 2))
    denom = np.sum(comps, axis=1, keepdims=True) + 1e-20
    # For each component, contribution: - comps * (x - mu) / sigma^2
    contrib = np.zeros((N, D))
    for k in range(K):
        contrib += ( - (x - mus[k]) / (sigmas[k] ** 2) )[:, :] * comps[:, k:k+1]
    score = contrib / denom
    return score  # shape (N,2)


def sample_mixture(n, mus, weights, sigmas, rng):
    K = mus.shape[0]
    comp_choices = rng.choice(K, size=n, p=weights)
    samples = np.zeros((n, D))
    for k in range(K):
        nk = np.sum(comp_choices == k)
        if nk > 0:
            samples[comp_choices == k] = rng.normal(loc=mus[k], scale=sigmas[k], size=(nk, D))
    return samples

# Biophysical energy E_bio = -log q(x) where q is another mixture (w_bio, sigma_bio)
# Force field (negative gradient of E_bio) is grad log q(x)

def force_bio(x):
    return score_mixture(x, mus, w_bio, sigma_bio)

# Sampling distribution score (ground-truth)
def score_samp(x):
    return score_mixture(x, mus, w_samp, sigma_samp)

# 2) Generate dataset
N_data = 2000
X_data = sample_mixture(N_data, mus, w_samp, sigma_samp, rng)

# 3) Train a small MLP to approximate the score of the sampling distribution
# Prepare dataset for score regression: we can sample points from data and surrounding
N_train = 5000
X_train = np.vstack([
    X_data[rng.choice(N_data, size=N_train//2, replace=True)],
    rng.uniform(low=-6, high=6, size=(N_train//2, D))
])
# compute analytic scores as targets
y_train = score_samp(X_train)

# PyTorch dataset
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

# MLP model for score
class ScoreNet(nn.Module):
    def __init__(self, dim=2, hidden=128):
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

score_model = ScoreNet(dim=D, hidden=128).to(device)
optimizer = optim.Adam(score_model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

# Training loop
n_epochs = 300
try:
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
        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch+1}/{n_epochs}, loss={epoch_loss:.6f}")
except Exception as e:
    print(json.dumps({"error": "Training score model failed", "exception": str(e), "trace": traceback.format_exc()}))
    raise

# Evaluate score model on a grid
grid_size = 80
xs = np.linspace(-6, 6, grid_size)
ys = np.linspace(-1, 6, grid_size)
xx, yy = np.meshgrid(xs, ys)
grid = np.stack([xx.ravel(), yy.ravel()], axis=1)

with torch.no_grad():
    pred_scores = score_model(torch.tensor(grid, dtype=torch.float32)).numpy()

true_scores = score_samp(grid)
bio_forces = force_bio(grid)

# compute cosine similarities (per point)
def cosine_sim(a, b):
    # a,b shape (N,2)
    dot = np.sum(a * b, axis=1)
    na = np.linalg.norm(a, axis=1)
    nb = np.linalg.norm(b, axis=1)
    denom = np.maximum(na * nb, 1e-12)
    return dot / denom

sim_pred_vs_bio = cosine_sim(pred_scores, bio_forces)
sim_true_vs_bio = cosine_sim(true_scores, bio_forces)

# 4) Train a simple GAN as an alternative generative model
# Generator: z (2D) -> x(2D)
class Generator(nn.Module):
    def __init__(self, zdim=2, out=2, hidden=128):
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
    def __init__(self, dim=2, hidden=128):
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
opt_G = optim.Adam(G.parameters(), lr=1e-3)
opt_D = optim.Adam(D.parameters(), lr=1e-3)

# Prepare real data loader
real_ds = torch.tensor(X_data, dtype=torch.float32)
real_loader = torch.utils.data.DataLoader(real_ds, batch_size=128, shuffle=True)

# Binary cross-entropy loss for vanilla GAN
bce = nn.BCELoss()

n_gan_steps = 2000
z_dim = 2
try:
    for step in range(n_gan_steps):
        for real_batch in real_loader:
            real_batch = real_batch.to(device)
            bsize = real_batch.size(0)
            # Train Discriminator
            opt_D.zero_grad()
            z = torch.randn(bsize, z_dim, device=device)
            fake = G(z).detach()
            d_real = D(real_batch)
            d_fake = D(fake)
            loss_D = bce(d_real, torch.ones_like(d_real)) + bce(d_fake, torch.zeros_like(d_fake))
            loss_D.backward()
            opt_D.step()
            # Train Generator
            opt_G.zero_grad()
            z2 = torch.randn(bsize, z_dim, device=device)
            fake2 = G(z2)
            d_fake2 = D(fake2)
            loss_G = bce(d_fake2, torch.ones_like(d_fake2))
            loss_G.backward()
            opt_G.step()
            break
        if (step + 1) % 200 == 0:
            print(f"GAN step {step+1}/{n_gan_steps}, loss_D={loss_D.item():.4f}, loss_G={loss_G.item():.4f}")
except Exception as e:
    print(json.dumps({"error": "GAN training failed", "exception": str(e), "trace": traceback.format_exc()}))
    raise

# 5) Generate samples from score-based sampler (Langevin using learned score) and from GAN

def langevin_sample(score_model, n_samples=1000, n_steps=200, step_size=0.1, noise_scale=0.1):
    x = rng.uniform(low=-6, high=6, size=(n_samples, D))
    x_t = torch.tensor(x, dtype=torch.float32)
    with torch.no_grad():
        for t in range(n_steps):
            s = score_model(x_t).numpy()
            # Euler-Maruyama: x <- x + step_size * score + sqrt(2*step_size) * N(0,I)
            noise = rng.normal(scale=math.sqrt(2 * step_size), size=x_t.shape)
            x = x_t.numpy() + step_size * s + noise
            x_t = torch.tensor(x, dtype=torch.float32)
    return x_t.numpy()

score_samples = langevin_sample(score_model, n_samples=2000, n_steps=150, step_size=0.05)

# GAN samples
with torch.no_grad():
    z = torch.randn(2000, z_dim)
    gan_samples = G(z).numpy()

# 6) Compute KDE-based score from a set of samples (analytical for Gaussian kernel)

def kde_score_at_points(points, samples, bandwidth=0.5):
    # points: (M,2), samples: (N,2)
    M = points.shape[0]
    N = samples.shape[0]
    diffs = points[:, None, :] - samples[None, :, :]  # (M,N,2)
    sq = np.sum(diffs * diffs, axis=2)
    K = np.exp(-0.5 * sq / (bandwidth ** 2))
    weights = K  # shape (M,N)
    denom = np.sum(weights, axis=1, keepdims=True) + 1e-20
    # grad log kde = - sum_i K_i * (x - x_i) / h^2 / sum K_i
    num = - np.einsum('ij,ijk->ik', weights, diffs)
    score = num / ( (bandwidth ** 2) * denom )
    return score

# compute kde score on grid for GAN samples
kde_bandwidth = 0.6
kde_scores_gan = kde_score_at_points(grid, gan_samples, bandwidth=kde_bandwidth)
sim_kde_vs_bio = cosine_sim(kde_scores_gan, bio_forces)

# 7) Energies according to biophysical energy E_bio = -log q(x)
def energy_bio(x):
    logpdf, _ = mixture_logpdf_and_components(x, mus, w_bio, sigma_bio)
    return -logpdf

E_data = energy_bio(X_data)
E_score = energy_bio(score_samples)
E_gan = energy_bio(gan_samples)

# compute mean energies and standard errors
mean_E_data = float(np.mean(E_data))
mean_E_score = float(np.mean(E_score))
mean_E_gan = float(np.mean(E_gan))
se_E_data = float(np.std(E_data) / math.sqrt(len(E_data)))
se_E_score = float(np.std(E_score) / math.sqrt(len(E_score)))
se_E_gan = float(np.std(E_gan) / math.sqrt(len(E_gan)))

# 8) Simple permutation test for difference in means between score samples and GAN samples

def permutation_test(a, b, n_perm=2000):
    rng_local = default_rng(SEED+1)
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

p_energy = permutation_test(E_score, E_gan, n_perm=1000)

# 9) Aggregate metrics
metrics = {
    # alignment metrics (mean cosine similarity over the grid)
    "alignment_score_model_vs_bio_mean": float(np.mean(sim_pred_vs_bio)),
    "alignment_true_score_vs_bio_mean": float(np.mean(sim_true_vs_bio)),
    "alignment_kde_gan_vs_bio_mean": float(np.mean(sim_kde_vs_bio)),
    # energy metrics
    "mean_energy_data": mean_E_data,
    "mean_energy_score_samples": mean_E_score,
    "mean_energy_gan_samples": mean_E_gan,
    "se_energy_data": se_E_data,
    "se_energy_score_samples": se_E_score,
    "se_energy_gan_samples": se_E_gan,
    "p_value_energy_score_vs_gan": float(p_energy),
    # simple diagnostics
    "n_data": int(N_data),
    "n_score_samples": int(score_samples.shape[0]),
    "n_gan_samples": int(gan_samples.shape[0])
}

# Print final JSON to stdout
print(json.dumps(metrics))
