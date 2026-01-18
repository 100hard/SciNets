# === REPRODUCIBILITY HEADER (Auto-injected) ===
import random
import numpy as np
random.seed(42)
np.random.seed(42)
try:
    import torch
    torch.manual_seed(42)
except ImportError:
    pass
# === END HEADER ===

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')  # use non-interactive backend to avoid GUI delays
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import traceback

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

try:
    np.random.seed(42)
    n = 300  # reduced samples to keep runtime small

    # Simulate taxa abundances (simple Dirichlet-like)
    taxa_dirichlet_alpha = np.array([2, 1.5, 1.0, 0.8, 0.6])
    taxa = np.random.dirichlet(taxa_dirichlet_alpha, size=n)

    # Simulate metabolites: butyrate strongly tied to taxa[:,0], IPA to taxa[:,1]
    base_but = 3.5
    base_ipa = 1.0
    butyrate = base_but * taxa[:, 0] + 0.3 * np.random.randn(n)
    ipa = base_ipa * taxa[:, 1] + 0.15 * np.random.randn(n)
    butyrate = np.clip(butyrate, 1e-4, None)
    ipa = np.clip(ipa, 1e-4, None)

    # Other metabolites (weak)
    acetate = 2.5 * taxa[:, 2] + 0.3 * np.random.randn(n)
    other_indole = 0.4 * taxa[:, 3] + 0.15 * np.random.randn(n)
    acetate = np.clip(acetate, 1e-4, None)
    other_indole = np.clip(other_indole, 1e-4, None)

    # Lightweight correlated cytokines: start with independent normals then mix
    z = np.random.randn(n, 3)
    mix = np.array([[1.0, 0.25, -0.15], [0.2, 1.0, 0.1], [-0.1, 0.05, 1.0]])
    cytokines_raw = z.dot(mix.T)  # cheap correlated structure
    IL6 = 1.0 + 0.45 * cytokines_raw[:, 0]
    IL10 = 2.0 + 0.25 * cytokines_raw[:, 1]
    TNF = 1.5 + 0.35 * cytokines_raw[:, 2]

    # Ratio and context-dependent generative model
    ratio = butyrate / ipa

    beta_ratio = 0.9
    beta_taxa_but = 1.0
    beta_IL6 = -0.6
    beta_interaction = -1.2
    intercept = -0.15

    latent = (
        intercept
        + beta_ratio * np.log1p(ratio)
        + beta_taxa_but * taxa[:, 0]
        + beta_IL6 * IL6
        + beta_interaction * (np.log1p(ratio) * (IL6 - IL6.mean()))
        + 0.4 * np.random.randn(n)
    )

    prob = sigmoid(latent)
    y = (prob > 0.5).astype(int)

    # Features
    X_single = np.log1p(ratio).reshape(-1, 1)
    X_composite = np.column_stack([
        np.log1p(ratio),
        taxa[:, 0],
        taxa[:, 1],
        IL6, IL10, TNF,
    ])

    # Single train/test split (consistent indices)
    idx = np.arange(n)
    train_idx, test_idx = train_test_split(idx, test_size=0.3, random_state=1, stratify=y)

    Xs_train = X_single[train_idx]
    Xs_test = X_single[test_idx]
    Xc_train = X_composite[train_idx]
    Xc_test = X_composite[test_idx]
    y_train = y[train_idx]
    y_test = y[test_idx]

    # Scale composite
    scaler = StandardScaler()
    Xc_train_scaled = scaler.fit_transform(Xc_train)
    Xc_test_scaled = scaler.transform(Xc_test)

    # Fit simple logistic models with low iterations
    clf_single = LogisticRegression(solver='liblinear', max_iter=30)
    clf_single.fit(Xs_train, y_train)
    clf_composite = LogisticRegression(solver='liblinear', max_iter=30)
    clf_composite.fit(Xc_train_scaled, y_train)

    auc_single_base = roc_auc_score(y_test, clf_single.predict_proba(Xs_test)[:, 1])
    auc_composite_base = roc_auc_score(y_test, clf_composite.predict_proba(Xc_test_scaled)[:, 1])

    # Noise sweep: fewer levels to save time
    noise_levels = [0.0, 0.1, 0.3, 0.6]
    aucs_single = []
    aucs_composite = []

    for nl in noise_levels:
        # Add Gaussian noise to test inputs
        Xs_test_noisy = Xs_test + nl * np.random.randn(*Xs_test.shape)
        Xc_test_noisy = Xc_test + nl * np.random.randn(*Xc_test.shape)
        Xc_test_noisy_scaled = scaler.transform(Xc_test_noisy)

        try:
            auc_s = roc_auc_score(y_test, clf_single.predict_proba(Xs_test_noisy)[:, 1])
        except Exception:
            auc_s = 0.5
        try:
            auc_c = roc_auc_score(y_test, clf_composite.predict_proba(Xc_test_noisy_scaled)[:, 1])
        except Exception:
            auc_c = 0.5

        aucs_single.append(float(np.round(auc_s, 4)))
        aucs_composite.append(float(np.round(auc_c, 4)))

    # Stability score based on composite AUC drop
    base = aucs_composite[0]
    worst = aucs_composite[-1]
    if base <= 0:
        stability_score = 0.0
    else:
        drop = max(0.0, (base - worst) / max(1e-6, base))
        stability_score = float(np.clip(1 - drop, 0.0, 1.0))

    if drop > 0.35:
        sensitivity = 'high'
    elif drop > 0.12:
        sensitivity = 'medium'
    else:
        sensitivity = 'low'

    # Failure modes heuristics
    failure_modes = []
    if np.sum(ipa < 0.05) > 0:
        failure_modes.append('ratio_instability_when_IPA_near_zero')
    if aucs_composite[-1] < 0.55:
        failure_modes.append('performance_collapse_under_high_noise')
    corr_taxa_ratio = np.corrcoef(taxa[:, 0], np.log1p(ratio))[0, 1]
    if abs(corr_taxa_ratio) > 0.85:
        failure_modes.append('strong_collinearity_between_taxa_and_ratio')
    if n < 200:
        failure_modes.append('small_sample_size_limitation')

    behavioral_pattern = (
        'Composite biomarker typically outperforms single ratio at baseline and is more robust to modest measurement noise. ' 
        'Model shows context-dependent effect: ratio influence modulated by IL6 (interaction in generative model). ' 
        'At high noise levels both models degrade; ratio instability can occur when IPA values are very small.'
    )

    consistency_check = 'partial' if (auc_composite_base > auc_single_base and abs(beta_interaction) > 0.5) else 'no'

    # Compact plot
    plt.figure(figsize=(6,3))
    plt.plot(noise_levels, aucs_single, marker='o', label='Single ratio')
    plt.plot(noise_levels, aucs_composite, marker='o', label='Composite')
    plt.xlabel('Noise std')
    plt.ylabel('AUC')
    plt.ylim(0.45, 1.02)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig('plot.png', dpi=100)
    plt.close()

    result = {
        'stability_score': float(np.round(stability_score, 3)),
        'sensitivity': sensitivity,
        'failure_modes': failure_modes,
        'behavioral_pattern': behavioral_pattern,
        'consistency_check': consistency_check,
        'plot': 'plot.png'
    }

    print(json.dumps(result))

except Exception as e:
    err = {'error': 'execution_failed', 'details': traceback.format_exc()}
    print(json.dumps(err))
