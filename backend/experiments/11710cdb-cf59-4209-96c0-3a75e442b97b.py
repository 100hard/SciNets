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
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

np.random.seed(42)

try:
    # PARAMETERS
    n = 500
    eps = 1e-6

    # Latent groups: 1 = indole-dominant, 0 = kynurenine-dominant
    p_indole = 0.5
    group = np.random.binomial(1, p_indole, size=n)

    # Simulate tnaA_abundance (copies per gram) - lognormal
    # Indole group has higher tnaA
    tnaA_mu = np.where(group == 1, 6.5, 5.5)  # log-space mean
    tnaA_sigma = 0.7
    tnaA = np.random.lognormal(mean=tnaA_mu, sigma=tnaA_sigma)

    # Simulate plasma IPA (indole-3-propionic acid) correlated with log10(tnaA)
    # IPA in uM scale
    a_base = 0.8
    ipa_noise_sigma = 0.2
    ipa = a_base * (np.log10(tnaA + eps)) + np.random.normal(0, ipa_noise_sigma, size=n)
    ipa = np.maximum(ipa, eps)  # ensure positive

    # Simulate host peripheral IDO1 expression (relative units) - higher in kynurenine group
    ido1_mu = np.where(group == 1, 1.0, 1.6)
    ido1_sigma = 0.25
    ido1 = np.random.normal(loc=ido1_mu, scale=ido1_sigma)
    ido1 = np.clip(ido1, eps, None)

    # Simulate quinolinic acid (uM) correlated with IDO1
    q_base = 0.9
    quin_noise_sigma = 0.25
    quin = q_base * ido1 + np.random.normal(0, quin_noise_sigma, size=n)
    quin = np.maximum(quin, eps)

    # Compute MNI as specified
    log_tnaA = np.log10(tnaA + eps)
    mni = (log_tnaA * ipa) / (ido1 * quin)

    df = pd.DataFrame({
        'group': group,
        'tnaA': tnaA,
        'log_tnaA': log_tnaA,
        'ipa': ipa,
        'ido1': ido1,
        'quin': quin,
        'mni': mni
    })

    # Baseline separation metric (internal only, not reported outward as validation metric)
    baseline_diff = df.loc[df['group']==1, 'mni'].mean() - df.loc[df['group']==0, 'mni'].mean()

    # STABILITY: Add noise at a range of levels and compute Spearman correlation vs baseline MNI
    noise_levels = [0.01, 0.05, 0.1, 0.2, 0.5]
    spearman_corrs = []
    for nl in noise_levels:
        # multiplicative noise on raw measurements
        tnaA_noisy = df['tnaA'] * np.exp(np.random.normal(0, nl, size=n))
        ipa_noisy = df['ipa'] + np.random.normal(0, nl * df['ipa'].std(), size=n)
        ido1_noisy = df['ido1'] + np.random.normal(0, nl * df['ido1'].std(), size=n)
        quin_noisy = df['quin'] + np.random.normal(0, nl * df['quin'].std(), size=n)

        # sanitize
        ipa_noisy = np.maximum(ipa_noisy, eps)
        ido1_noisy = np.clip(ido1_noisy, eps, None)
        quin_noisy = np.maximum(quin_noisy, eps)

        mni_noisy = (np.log10(tnaA_noisy + eps) * ipa_noisy) / (ido1_noisy * quin_noisy)

        # Spearman correlation
        corr = df['mni'].rank().corr(pd.Series(mni_noisy).rank(), method='pearson')
        if np.isnan(corr):
            corr = 0.0
        spearman_corrs.append(float(corr))

    # Stability score: mean spearman correlation across noise levels, clipped to [0,1]
    stability_score = float(np.clip(np.mean(spearman_corrs), 0.0, 1.0))

    # SENSITIVITY: vary biochemical scaling parameters and measure relative change in between-group MNI difference
    # We'll vary IPA scaling (a_base) and Quin scaling (q_base)
    perturb_factors = [0.7, 0.85, 1.0, 1.15, 1.3]
    rel_changes = []
    for af in perturb_factors:
        for qf in perturb_factors:
            ipa_p = af * (np.log10(df['tnaA'] + eps)) + np.random.normal(0, ipa_noise_sigma, size=n)
            ipa_p = np.maximum(ipa_p, eps)
            quin_p = (q_base * qf) * df['ido1'] + np.random.normal(0, quin_noise_sigma, size=n)
            quin_p = np.maximum(quin_p, eps)
            mni_p = (np.log10(df['tnaA'] + eps) * ipa_p) / (df['ido1'] * quin_p)
            diff_p = mni_p[df['group']==1].mean() - mni_p[df['group']==0].mean()
            if np.abs(baseline_diff) < eps:
                rel = np.nan
            else:
                rel = np.abs(diff_p - baseline_diff) / (np.abs(baseline_diff))
            if not np.isnan(rel):
                rel_changes.append(rel)

    avg_rel_change = float(np.nanmean(rel_changes)) if len(rel_changes) > 0 else 0.0

    # Map avg_rel_change to sensitivity category
    if avg_rel_change > 0.5:
        sensitivity = 'high'
    elif avg_rel_change > 0.2:
        sensitivity = 'medium'
    else:
        sensitivity = 'low'

    # FAILURE MODES: detect conditions observed in synthetic runs
    failure_modes = []
    # 1) zeros or extremely small tnaA causing unstable log
    zero_tnaA_count = int((df['tnaA'] <= 1e-3).sum())
    if zero_tnaA_count > 0:
        failure_modes.append('zeros_or_tiny_tnaA_leading_to_unstable_log')

    # 2) very small denominator (IDO1 * Quin) leading to extreme MNI
    denom = df['ido1'] * df['quin']
    tiny_denom_count = int((denom < 1e-3).sum())
    if tiny_denom_count > 0:
        failure_modes.append('tiny_denominator_causing_extreme_MNI_values')

    # 3) outliers in MNI (extreme percentiles)
    lowp = df['mni'].quantile(0.01)
    highp = df['mni'].quantile(0.99)
    outlier_count = int(((df['mni'] < lowp) | (df['mni'] > highp)).sum())
    if outlier_count > max(5, 0.01 * n):
        failure_modes.append('high_fraction_of_MNI_outliers')

    # 4) overlap under realistic noise: if stability low under modest noise, flag
    if stability_score < 0.6:
        failure_modes.append('substantial_overlap_under_modest_measurement_noise')

    # 5) sensitivity to parameter scaling
    if avg_rel_change > 0.3:
        failure_modes.append('sensitivity_to_biochemical_scaling_parameters')

    if len(failure_modes) == 0:
        failure_modes = ['none_observed_in_synthetic_experiment']

    # BEHAVIORAL PATTERN: descriptive summary
    # We keep this concise and mechanistic
    mean_mni_indole = float(df.loc[df['group']==1, 'mni'].mean())
    mean_mni_kyn = float(df.loc[df['group']==0, 'mni'].mean())
    behavioral_pattern = (
        f"MNI tends to be higher in the simulated indole-dominant group (mean ~ {mean_mni_indole:.3g}) "
        f"than in the kynurenine-dominant group (mean ~ {mean_mni_kyn:.3g}). "
        f"Under increasing measurement noise the rank-ordering of MNI is {('relatively preserved' if stability_score>0.7 else 'partially degraded')}, "
        f"and between-group separation is {('robust' if sensitivity=='low' else 'sensitive to parameter choices')}."
    )

    # CONSISTENCY CHECK: does behavior align with hypothesized mechanism?
    # Because the synthetic data was generated with the mechanism baked in, we evaluate whether MNI reflects the designed directions
    # Use a conservative mapping: if mean_mni_indole > mean_mni_kyn and stability reasonable -> partial or yes
    if (mean_mni_indole > mean_mni_kyn) and (stability_score > 0.75) and (sensitivity == 'low'):
        consistency_check = 'yes'
    elif (mean_mni_indole > mean_mni_kyn):
        consistency_check = 'partial'
    else:
        consistency_check = 'no'

    # PLOT: MNI distributions by group
    plt.figure(figsize=(6,4))
    plt.boxplot([df.loc[df['group']==0, 'mni'], df.loc[df['group']==1, 'mni']], labels=['kynurenine', 'indole'])
    plt.ylabel('MNI')
    plt.title('MNI by simulated subtype')
    plt.tight_layout()
    plt.savefig('plot.png', dpi=150)
    plt.close()

    # Build final metrics dict to print
    metrics = {
        'stability_score': round(stability_score, 3),
        'sensitivity': sensitivity,
        'failure_modes': failure_modes,
        'behavioral_pattern': behavioral_pattern,
        'consistency_check': consistency_check,
        'plot': 'plot.png'
    }

    print(json.dumps(metrics))

except Exception as e:
    # If anything fails, print a JSON with error key as required
    err = {'error': str(e)}
    print(json.dumps(err))
