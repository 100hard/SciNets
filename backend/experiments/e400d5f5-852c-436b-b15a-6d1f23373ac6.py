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
import matplotlib.pyplot as plt
from scipy.special import expit

np.random.seed(42)

# --- model functions ---

def simulate_cohort(params, N=500, noise_scale=0.1):
    # params: dict of model parameters
    # generate bai abundance
    b = np.random.beta(2,2,size=N)  # abundance in (0,1), unimodal

    # LCA production increases with bai
    LCA_base = params['LCA_base']
    LCA = LCA_base + params['alpha_bai'] * b + np.random.normal(0, noise_scale*LCA_base, size=N)
    LCA = np.maximum(LCA, 0.0)

    # CDCA decreases with LCA (competitive balance)
    CDCA_base = params['CDCA_base']
    CDCA = CDCA_base * np.exp(-params['k_cdca_lca'] * LCA) + np.random.normal(0, noise_scale*CDCA_base, size=N)
    CDCA = np.maximum(CDCA, 1e-6)

    # receptor (FXR/TGR5) activity suppressed by LCA (nonlinear)
    receptor = params['receptor_base'] * np.exp(-params['k_lca_receptor'] * LCA)
    receptor += np.random.normal(0, noise_scale*params['receptor_base'], size=N)
    receptor = np.clip(receptor, 0.0, 1.0)

    # tight junction integrity scales with receptor activity
    TJ = params['TJ_base'] * (receptor ** params['tj_nonlinearity'])
    TJ += np.random.normal(0, noise_scale*params['TJ_base'], size=N)
    TJ = np.clip(TJ, 0.0, 1.0)

    # permeability inverse of TJ
    permeability = params['permeability_base'] + (1 - TJ) * params['permeability_slope']
    permeability = np.clip(permeability, 0.0, 1.0)

    # endotoxin translocation proportional to permeability and gut LPS load, with clearance
    gut_LPS = params['gut_LPS']
    endotoxin = permeability * gut_LPS - params['LPS_clearance']
    endotoxin = np.maximum(endotoxin, 0.0)

    # peripheral inflammation: sigmoid of endotoxin
    inflammation = expit(params['k_endotoxin_inflam'] * (endotoxin - params['inflam_offset']))

    # microglial activation as sigmoid of peripheral inflammation
    microglia = expit(params['k_inflam_micro'] * (inflammation - params['micro_offset']))

    # baseline depressive score and treatment effectiveness reduced by microglial activation
    baseline_dep = params['baseline_depression']
    treat_base = params['treatment_base']
    treat_reduction = treat_base * (1 - params['k_micro_treat_supp'] * microglia)
    treat_reduction = np.clip(treat_reduction, 0.0, treat_base)

    depressive_score = baseline_dep + params['k_micro_dep'] * microglia - treat_reduction
    depressive_score = np.clip(depressive_score, 0.0, None)

    out = {
        'b': b,
        'LCA': LCA,
        'CDCA': CDCA,
        'receptor': receptor,
        'TJ': TJ,
        'permeability': permeability,
        'endotoxin': endotoxin,
        'inflammation': inflammation,
        'microglia': microglia,
        'depressive_score': depressive_score
    }
    return out

# --- nominal parameters ---
nominal_params = {
    'LCA_base': 0.5,
    'alpha_bai': 1.2,         # effect of bai on LCA
    'CDCA_base': 1.0,
    'k_cdca_lca': 0.8,
    'receptor_base': 0.9,
    'k_lca_receptor': 1.0,   # suppression strength of LCA on receptor
    'TJ_base': 0.9,
    'tj_nonlinearity': 1.0,
    'permeability_base': 0.05,
    'permeability_slope': 0.9,
    'gut_LPS': 1.0,
    'LPS_clearance': 0.05,
    'k_endotoxin_inflam': 6.0,
    'inflam_offset': 0.1,
    'k_inflam_micro': 8.0,
    'micro_offset': 0.2,
    'baseline_depression': 0.2,
    'treatment_base': 0.5,
    'k_micro_treat_supp': 0.8, # microglia suppress treatment response
    'k_micro_dep': 1.2
}

# --- stability assessment ---
N = 500
reps = 50
all_dep = np.zeros(reps)
for i in range(reps):
    out = simulate_cohort(nominal_params, N=N, noise_scale=0.12)
    all_dep[i] = out['depressive_score'].mean()

mean_dep = all_dep.mean()
std_dep = all_dep.std()
# normalized coefficient of variation
cv = std_dep / (mean_dep + 1e-6)
stability_score = float(max(0.0, min(1.0, 1.0 - cv)))

# --- sensitivity sweep ---
# pick key parameters to vary: alpha_bai, k_lca_receptor, tj_nonlinearity, k_micro_treat_supp
P = 300
param_samples = []
for _ in range(P):
    s = {
        'LCA_base': 0.4,
        'alpha_bai': np.random.uniform(0.2, 2.0),
        'CDCA_base': 1.0,
        'k_cdca_lca': np.random.uniform(0.2,1.5),
        'receptor_base': 0.9,
        'k_lca_receptor': np.random.uniform(0.2,2.0),
        'TJ_base': 0.9,
        'tj_nonlinearity': np.random.uniform(0.5,2.0),
        'permeability_base': 0.05,
        'permeability_slope': 0.9,
        'gut_LPS': 1.0,
        'LPS_clearance': 0.05,
        'k_endotoxin_inflam': np.random.uniform(3,10),
        'inflam_offset': 0.1,
        'k_inflam_micro': np.random.uniform(4,12),
        'micro_offset': 0.2,
        'baseline_depression': 0.2,
        'treatment_base': 0.5,
        'k_micro_treat_supp': np.random.uniform(0.0,1.0),
        'k_micro_dep': np.random.uniform(0.6,1.6)
    }
    param_samples.append(s)

mean_deps = np.zeros(P)
all_deps_vector = []
for i,ps in enumerate(param_samples):
    out = simulate_cohort(ps, N=300, noise_scale=0.12)
    mean_deps[i] = out['depressive_score'].mean()
    all_deps_vector.append(out['depressive_score'])
all_deps_vector = np.concatenate(all_deps_vector)

var_between = np.var(mean_deps)
var_total = np.var(all_deps_vector)
sensitivity_index = float(var_between / (var_total + 1e-9))
if sensitivity_index > 0.6:
    sensitivity_label = 'high'
elif sensitivity_index > 0.2:
    sensitivity_label = 'medium'
else:
    sensitivity_label = 'low'

# --- failure mode detections ---
failure_modes = []
# 1) negative intermediate detection before clipping (we used clipping, but check if inputs would have been negative in reasonable ranges)
# We'll re-run without clipping on some intermediates to see if negatives occur
out_raw = simulate_cohort(nominal_params, N=500, noise_scale=0.5)
if (out_raw['LCA'] < 0).any() or (out_raw['CDCA'] <= 0).any():
    failure_modes.append('negative_intermediates')

# 2) receptor saturation (most samples near 0 or 1)
r = out_raw['receptor']
if (np.mean(r < 0.05) > 0.5) or (np.mean(r > 0.95) > 0.5):
    failure_modes.append('receptor_saturation')

# 3) microglia saturation
m = out_raw['microglia']
if (np.mean(m < 0.05) > 0.5) or (np.mean(m > 0.95) > 0.5):
    failure_modes.append('microglia_saturation')

# 4) unexpected direction: if bai abundance correlates negatively with depressive score
corr = np.corrcoef(out_raw['b'], out_raw['depressive_score'])[0,1]
if corr < -0.1:
    failure_modes.append('unexpected_direction')

# 5) treatment paradox: check whether higher microglia sometimes increases treatment effectiveness
# here treatment effectiveness is treated as treat_reduction; check correlation between microglia and treatment reduction
# Recompute treatment_reduction from simulate_cohort internals
# We'll approximate as treatment_base*(1 - k_micro_treat_supp * microglia)
tr = nominal_params['treatment_base'] * (1 - nominal_params['k_micro_treat_supp'] * out_raw['microglia'])
if np.mean(tr < 0) > 0.01:
    failure_modes.append('treatment_over_suppression')

if len(failure_modes) == 0:
    failure_modes.append('none_detected')

# --- behavioral pattern summary ---
# compute correlations across nominal run
out_nom = simulate_cohort(nominal_params, N=500, noise_scale=0.12)
corr_b_LCA = np.corrcoef(out_nom['b'], out_nom['LCA'])[0,1]
corr_LCA_receptor = np.corrcoef(out_nom['LCA'], out_nom['receptor'])[0,1]
corr_receptor_TJ = np.corrcoef(out_nom['receptor'], out_nom['TJ'])[0,1]
corr_perme_endotoxin = np.corrcoef(out_nom['permeability'], out_nom['endotoxin'])[0,1]
corr_endotoxin_inflam = np.corrcoef(out_nom['endotoxin'], out_nom['inflammation'])[0,1]
corr_inflam_micro = np.corrcoef(out_nom['inflammation'], out_nom['microglia'])[0,1]
corr_micro_dep = np.corrcoef(out_nom['microglia'], out_nom['depressive_score'])[0,1]

behavioral_pattern = (
    f"Simulated monotonic chain: bai -> LCA (corr={corr_b_LCA:.2f}), LCA -> receptor (corr={corr_LCA_receptor:.2f}), "
    f"receptor -> TJ (corr={corr_receptor_TJ:.2f}), permeability -> endotoxin (corr={corr_perme_endotoxin:.2f}), "
    f"endotoxin -> inflammation (corr={corr_endotoxin_inflam:.2f}), inflammation -> microglia (corr={corr_inflam_micro:.2f}), "
    f"microglia -> depressive_score (corr={corr_micro_dep:.2f})."
)

# --- consistency check ---
# crude rule: if bai->depressive corr positive and chain correlations indicate mediation, label 'partial' or 'yes'
corr_b_dep = np.corrcoef(out_nom['b'], out_nom['depressive_score'])[0,1]
if (corr_b_dep > 0.25) and (corr_LCA_receptor < -0.25) and (corr_micro_dep > 0.25):
    consistency_check = 'yes'
elif (corr_b_dep > 0.05) and (corr_LCA_receptor < -0.1) and (corr_micro_dep > 0.1):
    consistency_check = 'partial'
else:
    consistency_check = 'no'

# --- plot ---
plt.figure(figsize=(6,4))
sc = plt.scatter(out_nom['b'], out_nom['depressive_score'], c=out_nom['receptor'], cmap='viridis', alpha=0.6)
plt.colorbar(sc, label='receptor activity (FXR/TGR5)')
plt.xlabel('bai operon abundance (synthetic)')
plt.ylabel('depressive score (synthetic)')
plt.title('Depressive score vs bai abundance (color = receptor activity)')
plt.tight_layout()
plt.savefig('plot.png', dpi=150)
plt.close()

# --- assemble final metrics JSON ---
metrics = {
    'stability_score': round(float(stability_score), 3),
    'sensitivity': sensitivity_label,
    'failure_modes': failure_modes,
    'behavioral_pattern': behavioral_pattern,
    'consistency_check': consistency_check,
    'plot': 'plot.png'
}

print(json.dumps(metrics))
