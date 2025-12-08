import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import traceback

# Experiment parameters (kept small to respect runtime limits)
SEEDS = [0, 1, 2]
epochs = 20
batch_size = 64
train_size = 800
test_size = 100
ood_size = 100
img_size = 16
l1_lambda = 1e-3  # L1 applied to final layer weights for the L1 model
grad_zero_thresh = 1e-4
corr_threshold = 0.2

# Synthetic dataset: small square in the center encodes label; background color is binary cue spuriously correlated with label in train
def make_dataset(n, cue_balance=0.9, cue_correlated_with_label=True, seed=0):
    rng = np.random.RandomState(seed)
    X = np.zeros((n, 1, img_size, img_size), dtype=np.float32)
    y = np.zeros(n, dtype=np.int64)
    cue = np.zeros(n, dtype=np.int64)
    for i in range(n):
        # object: small 6x6 square in center: label 0 or 1
        label = rng.randint(0, 2)
        y[i] = label
        # cue assignment: if correlated, cue == label with prob cue_balance else opposite
        if cue_correlated_with_label:
            if rng.rand() < cue_balance:
                c = label
            else:
                c = 1 - label
        else:
            # independent cue (for OOD or id depending on use)
            c = rng.randint(0, 2)
        cue[i] = c
        img = rng.randn(img_size, img_size) * 0.1
        # background color: add a constant offset to entire image based on cue
        if c == 1:
            img += 0.8
        else:
            img += -0.8
        # draw center square with intensity based on label
        cx = img_size // 2
        cy = img_size // 2
        half = 3
        intensity = 1.0 if label == 1 else -1.0
        img[cx - half:cx + half, cy - half:cy + half] += intensity
        X[i, 0] = img
    X = (X - X.mean()) / (X.std() + 1e-9)
    return torch.tensor(X), torch.tensor(y), torch.tensor(cue)

class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 8, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((4,4))
        self.fc1 = nn.Linear(8*4*4, 64)
        self.fc2 = nn.Linear(64, 2)  # late layer
    def forward(self, x):
        x = F.relu(self.conv(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        h = F.relu(self.fc1(x))
        out = self.fc2(h)
        return out, h

# utilities

def eval_model(model, X, y):
    model.eval()
    with torch.no_grad():
        out, _ = model(X)
        preds = out.argmax(dim=1)
        acc = (preds == y).float().mean().item()
    return acc

# compute gradient vector for final layer on a specific mini-batch (without optimizer step)
def compute_final_gradients(model, Xb, yb, device, l1_lambda=0.0):
    model.zero_grad()
    model.train()
    out, _ = model(Xb)
    loss = F.cross_entropy(out, yb)
    if l1_lambda > 0:
        loss = loss + l1_lambda * torch.norm(model.fc2.weight, p=1)
    loss.backward()
    g = model.fc2.weight.grad.detach().cpu().numpy().ravel().copy()
    return g

# main experiment function for one seed and one regime

def run_one(seed, apply_l1=False):
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Build datasets
    X_train, y_train, cue_train = make_dataset(train_size, cue_balance=0.9, cue_correlated_with_label=True, seed=seed)
    X_test_id, y_test_id, cue_test_id = make_dataset(test_size, cue_balance=0.9, cue_correlated_with_label=True, seed=seed+100)
    # OOD: reverse the cue-label correlation (cue correlated with opposite label)
    X_test_ood, y_test_ood, cue_test_ood = make_dataset(ood_size, cue_balance=0.9, cue_correlated_with_label=False, seed=seed+200)
    # For OOD set up, we will explicitly correlate cue with flipped label to create strong spurious mismatch
    # Flip labels for OOD to construct a dataset where cue is anti-correlated with label
    # Generate labels then set cue = 1-label for strong mismatch
    # Simpler: generate dataset with independent cue, then flip cue to be opposite of label with high prob
    # To keep simple, we'll create a new OOD where cue is correlated with opposite label
    X_tmp, y_tmp, _ = make_dataset(ood_size, cue_balance=0.9, cue_correlated_with_label=True, seed=seed+300)
    # flip label to make cue predictive of opposite class by swapping labels
    y_test_ood = 1 - y_tmp
    # Keep X_tmp as images; we need to ensure background actually corresponds, rebuild images with cue opposite label
    # Reconstruct OOD images where cue corresponds to 1 - label
    X_test_ood = torch.zeros_like(X_tmp)
    cue_ood = torch.zeros(ood_size, dtype=torch.int64)
    rng = np.random.RandomState(seed+400)
    for i in range(ood_size):
        label = int(y_test_ood[i].item())
        # set cue to opposite label with prob 0.9
        if rng.rand() < 0.9:
            c = 1 - label
        else:
            c = label
        cue_ood[i] = c
        img = rng.randn(img_size, img_size) * 0.1
        if c == 1:
            img += 0.8
        else:
            img += -0.8
        # add center square based on label
        cx = img_size // 2
        cy = img_size // 2
        half = 3
        intensity = 1.0 if label == 1 else -1.0
        img[cx - half:cx + half, cy - half:cy + half] += intensity
        X_test_ood[i, 0] = img
    X_test_ood = (X_test_ood - X_test_ood.mean()) / (X_test_ood.std() + 1e-9)

    device = torch.device('cpu')

    model = SmallCNN().to(device)
    opt = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

    train_dataset = TensorDataset(X_train, y_train, cue_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # metrics to collect
    grad_sparsities = []  # episodic gradient sparsity per update (late-layer grads)
    # heteroscedastic: per-epoch gradient norms of cue-specific mini-batches
    grad_norms_by_cue = {0: [], 1: []}

    # training loop
    for ep in range(1, epochs + 1):
        model.train()
        for Xb, yb, cb in train_loader:
            Xb = Xb.to(device)
            yb = yb.to(device)
            opt.zero_grad()
            out, _ = model(Xb)
            loss = F.cross_entropy(out, yb)
            if apply_l1:
                loss = loss + l1_lambda * torch.norm(model.fc2.weight, p=1)
            loss.backward()
            # collect gradient sparsity of final layer
            g = model.fc2.weight.grad.detach().cpu().numpy().ravel()
            sparsity = float((np.abs(g) < grad_zero_thresh).sum()) / g.size
            grad_sparsities.append(sparsity)
            opt.step()
        # at epoch end, compute gradients on cue-specific mini-batches from training data (without optimizer step)
        # sample up to batch_size samples of each cue
        for cue_val in [0, 1]:
            idxs = (cue_train.numpy() == cue_val).nonzero()[0]
            if len(idxs) == 0:
                continue
            sel = np.random.choice(idxs, size=min(batch_size, len(idxs)), replace=False)
            Xb = X_train[sel].to(device)
            yb = y_train[sel].to(device)
            gvec = compute_final_gradients(model, Xb, yb, device, l1_lambda=(l1_lambda if apply_l1 else 0.0))
            grad_norm = float(np.linalg.norm(gvec))
            grad_norms_by_cue[cue_val].append(grad_norm)

    # After training, compute neuron activations correlation with background cue on the training set
    model.eval()
    with torch.no_grad():
        _, H = model(X_train)
        H = H.numpy()
        cue_arr = cue_train.numpy().astype(float)
        neuron_corrs = []
        for j in range(H.shape[1]):
            v = H[:, j]
            # Pearson correlation
            if np.std(v) < 1e-9:
                corr = 0.0
            else:
                corr = float(np.corrcoef(v, cue_arr)[0,1])
            neuron_corrs.append(corr)
        neuron_corrs = np.array(neuron_corrs)
        bg_sensitive_count = int((np.abs(neuron_corrs) > corr_threshold).sum())
        max_bg_corr = float(np.max(np.abs(neuron_corrs)))

    # Evaluate ID and OOD accuracies
    acc_id = eval_model(model, X_test_id, y_test_id)
    acc_ood = eval_model(model, X_test_ood, y_test_ood)

    # Compute heteroscedastic measure: difference of gradient variances between cues
    var0 = float(np.var(grad_norms_by_cue[0])) if len(grad_norms_by_cue[0])>0 else 0.0
    var1 = float(np.var(grad_norms_by_cue[1])) if len(grad_norms_by_cue[1])>0 else 0.0
    hetero_measure = abs(var0 - var1)

    return {
        'acc_id': acc_id,
        'acc_ood': acc_ood,
        'mean_grad_sparsity': float(np.mean(grad_sparsities)) if len(grad_sparsities)>0 else 0.0,
        'grad_sparsity_series': grad_sparsities,
        'hetero_measure': hetero_measure,
        'grad_norms_by_cue': grad_norms_by_cue,
        'bg_sensitive_count': bg_sensitive_count,
        'max_bg_corr': max_bg_corr
    }

# Run experiments for baseline and L1 across seeds
try:
    results = {
        'baseline': [],
        'l1': []
    }
    for seed in SEEDS:
        res_base = run_one(seed, apply_l1=False)
        results['baseline'].append(res_base)
        res_l1 = run_one(seed, apply_l1=True)
        results['l1'].append(res_l1)

    # Aggregate metrics across seeds
    def agg(list_of_dicts, key):
        vals = [d[key] for d in list_of_dicts]
        return float(np.mean(vals)), float(np.std(vals))

    metrics = {}
    metrics['seeds'] = SEEDS

    metrics['accuracy_id_baseline_mean'], metrics['accuracy_id_baseline_std'] = agg(results['baseline'], 'acc_id')
    metrics['accuracy_ood_baseline_mean'], metrics['accuracy_ood_baseline_std'] = agg(results['baseline'], 'acc_ood')
    metrics['accuracy_id_l1_mean'], metrics['accuracy_id_l1_std'] = agg(results['l1'], 'acc_id')
    metrics['accuracy_ood_l1_mean'], metrics['accuracy_ood_l1_std'] = agg(results['l1'], 'acc_ood')

    metrics['mean_grad_sparsity_baseline_mean'], metrics['mean_grad_sparsity_baseline_std'] = agg(results['baseline'], 'mean_grad_sparsity')
    metrics['mean_grad_sparsity_l1_mean'], metrics['mean_grad_sparsity_l1_std'] = agg(results['l1'], 'mean_grad_sparsity')

    metrics['hetero_measure_baseline_mean'], metrics['hetero_measure_baseline_std'] = agg(results['baseline'], 'hetero_measure')
    metrics['hetero_measure_l1_mean'], metrics['hetero_measure_l1_std'] = agg(results['l1'], 'hetero_measure')

    metrics['bg_sensitive_count_baseline_mean'], metrics['bg_sensitive_count_baseline_std'] = agg(results['baseline'], 'bg_sensitive_count')
    metrics['bg_sensitive_count_l1_mean'], metrics['bg_sensitive_count_l1_std'] = agg(results['l1'], 'bg_sensitive_count')

    # Prepare a plot comparing gradient sparsity time series (average across seeds) and bar chart of ID/OOD accuracies
    # Prepare average sparsity series by aligning lengths (they should be same length roughly)
    spars_base = np.concatenate([np.array(r['grad_sparsity_series']) for r in results['baseline']])
    spars_l1 = np.concatenate([np.array(r['grad_sparsity_series']) for r in results['l1']])

    plt.figure(figsize=(8,4))
    plt.subplot(1,2,1)
    # plot moving average of sparsity
    def moving_avg(x, k=5):
        if len(x) < k:
            return x
        return np.convolve(x, np.ones(k)/k, mode='valid')
    ma_base = moving_avg(spars_base, k=5)
    ma_l1 = moving_avg(spars_l1, k=5)
    plt.plot(ma_base, label='baseline sparsity (mov avg)')
    plt.plot(ma_l1, label='L1 sparsity (mov avg)')
    plt.xlabel('update (moving avg)')
    plt.ylabel('fraction near-zero grads (late layer)')
    plt.legend()

    plt.subplot(1,2,2)
    labels = ['ID acc', 'OOD acc']
    base_vals = [metrics['accuracy_id_baseline_mean'], metrics['accuracy_ood_baseline_mean']]
    l1_vals = [metrics['accuracy_id_l1_mean'], metrics['accuracy_ood_l1_mean']]
    x = np.arange(len(labels))
    width = 0.35
    plt.bar(x - width/2, base_vals, width, label='baseline')
    plt.bar(x + width/2, l1_vals, width, label='L1')
    plt.ylim(0,1)
    plt.xticks(x, labels)
    plt.legend()
    plt.tight_layout()
    plt.savefig('plot.png')

    # Final JSON metrics to stdout
    print(json.dumps(metrics))

except Exception as e:
    tb = traceback.format_exc()
    print(json.dumps({'error': str(e), 'traceback': tb}))
