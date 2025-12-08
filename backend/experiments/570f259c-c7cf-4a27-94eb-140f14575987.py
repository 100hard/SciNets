import numpy as np
import pandas as pd
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from math import sqrt, erf
import sys

# Helper functions
def normal_cdf(x):
    # standard normal CDF using erf
    return 0.5 * (1 + erf(x / np.sqrt(2)))

def two_sided_p_from_z(z):
    return 2 * (1 - normal_cdf(abs(z)))

def ols_numpy(X, y, feature_names=None):
    # X should include intercept column if desired
    n, p = X.shape
    XtX = X.T.dot(X)
    try:
        XtX_inv = np.linalg.inv(XtX)
    except np.linalg.LinAlgError:
        XtX_inv = np.linalg.pinv(XtX)
    beta = XtX_inv.dot(X.T).dot(y)
    y_hat = X.dot(beta)
    resid = y - y_hat
    rss = np.sum(resid ** 2)
    df_resid = max(n - p, 1)
    sigma2 = rss / df_resid
    cov_beta = sigma2 * XtX_inv
    se = np.sqrt(np.diag(cov_beta))
    t_stats = beta / se
    p_values = np.array([two_sided_p_from_z(t) for t in t_stats])
    # R-squared
    tss = np.sum((y - np.mean(y)) ** 2)
    rsq = 1 - rss / tss if tss > 0 else 0.0
    results = {
        'n': int(n),
        'p': int(p),
        'beta': beta,
        'se': se,
        't': t_stats,
        'p_values': p_values,
        'cov_beta': cov_beta,
        'rsq': float(rsq),
        'y_hat': y_hat,
        'resid': resid
    }
    if feature_names is not None:
        results['feature_names'] = feature_names
    return results


def add_intercept(X):
    return np.column_stack([np.ones(X.shape[0]), X])


# Set reproducible seed
RND = 42
np.random.seed(RND)

try:
    # 1) Simulate cohort
    n = 2000

    # Baseline covariates
    age = np.random.normal(60, 10, size=n)  # years
    sex = np.random.binomial(1, 0.5, size=n)  # 0 female, 1 male
    baseline_eGFR = np.clip(np.random.normal(55, 15, size=n), 10, 120)  # ml/min/1.73m2

    # Chronic PCS exposure (simulate skewed distribution typical for uremic solutes)
    pcs = np.random.lognormal(mean=1.5, sigma=0.8, size=n)  # arbitrary units

    # Experimental/Genetic TLR4 status groups: 0=WT,1=fib-KO,2=mac-KO,3=both-KO
    tlr_group = np.random.choice([0,1,2,3], size=n, p=[0.4,0.2,0.2,0.2])
    tlr4_fib = np.where(tlr_group == 1, 0, np.where(tlr_group == 3, 0, 1))  # 1 if functional
    tlr4_mac = np.where(tlr_group == 2, 0, np.where(tlr_group == 3, 0, 1))

    # Baseline cell activation noise
    base_fib_act = np.random.normal(0.5, 0.2, size=n)
    base_mac_act = np.random.normal(0.4, 0.25, size=n)

    # Parameters (ground truth)
    alpha_fib = 0.8   # strength of PCS effect on fibroblast activation when TLR4 present
    alpha_mac = 0.6   # strength on macrophage activation when TLR4 present
    leaky_factor = 0.15  # small TLR4-independent PCS effect on activation

    # Generate mediators: fibroblast and macrophage activation scores
    fibro_activation = (base_fib_act
                        + (alpha_fib * pcs * tlr4_fib)
                        + (leaky_factor * pcs * (1 - tlr4_fib))
                        + 0.01 * (age - 60)
                        + np.random.normal(0, 0.5, size=n))

    mac_activation = (base_mac_act
                      + (alpha_mac * pcs * tlr4_mac)
                      + (leaky_factor * pcs * (1 - tlr4_mac))
                      + 0.02 * (age - 60)
                      + np.random.normal(0, 0.6, size=n))

    # Outcome: fibrosis score (continuous)
    gamma_fib = 1.2
    gamma_mac = 0.9
    direct_pcs_on_fibrosis = 0.05  # small direct PCS effect not via TLR4

    fibrosis_score = (0.5
                      + gamma_fib * fibro_activation
                      + gamma_mac * mac_activation
                      + direct_pcs_on_fibrosis * pcs
                      - 0.005 * (baseline_eGFR - 55)
                      + np.random.normal(0, 1.0, size=n))

    # Binary CKD progression within follow-up (simulated): probability increases with fibrosis
    logistic = lambda x: 1.0 / (1.0 + np.exp(-x))
    prog_logit = -3.0 + 0.9 * fibrosis_score - 0.02 * (baseline_eGFR - 55)
    prog_prob = logistic(prog_logit)
    ckd_progression = np.random.binomial(1, prog_prob, size=n)

    # Assemble DataFrame
    df = pd.DataFrame({
        'pcs': pcs,
        'age': age,
        'sex': sex,
        'baseline_eGFR': baseline_eGFR,
        'tlr_group': tlr_group,
        'tlr4_fib': tlr4_fib,
        'tlr4_mac': tlr4_mac,
        'fibro_activation': fibro_activation,
        'mac_activation': mac_activation,
        'fibrosis_score': fibrosis_score,
        'ckd_progression': ckd_progression
    })

    # 2) Analysis 1: Does PCS associate with fibrosis and is it modified by fibroblast TLR4?
    df['pcs_std'] = (df['pcs'] - df['pcs'].mean()) / df['pcs'].std()
    df['fibro_tlr'] = df['tlr4_fib'].astype(int)
    df['pcs_x_fibroTLR'] = df['pcs_std'] * df['fibro_tlr']

    features_lm1 = ['pcs_std', 'fibro_tlr', 'pcs_x_fibroTLR', 'age', 'sex', 'baseline_eGFR']
    X_lm1 = add_intercept(df[features_lm1].values)
    y_lm1 = df['fibrosis_score'].values
    lm1 = ols_numpy(X_lm1, y_lm1, feature_names=['intercept'] + features_lm1)

    # Map feature names to indices
    feat_names_lm1 = lm1['feature_names']
    name_to_idx_lm1 = {name: i for i, name in enumerate(feat_names_lm1)}

    # 3) Analysis 2: Mediation by fibro_activation
    # Path a: mediator ~ pcs + fibro_tlr + pcs:fibro_tlr + covariates
    features_a = ['pcs_std', 'fibro_tlr', 'pcs_x_fibroTLR', 'age', 'sex', 'baseline_eGFR']
    X_a = add_intercept(df[features_a].values)
    y_a = df['fibro_activation'].values
    model_a = ols_numpy(X_a, y_a, feature_names=['intercept'] + features_a)

    # Path b: fibrosis ~ standardized fibro_activation + pcs + covariates
    scaler = StandardScaler()
    df['fibro_activation_std'] = scaler.fit_transform(df[['fibro_activation']])
    features_b = ['fibro_activation_std', 'pcs_std', 'age', 'sex', 'baseline_eGFR']
    X_b = add_intercept(df[features_b].values)
    y_b = df['fibrosis_score'].values
    model_b = ols_numpy(X_b, y_b, feature_names=['intercept'] + features_b)

    # Indirect effect: because model_a includes interaction, compute indirect effect for fibro_tlr=1 (TLR present)
    # a_wt = coef_pcs_std + coef_pcs_x_fibroTLR
    idx_a_pcs = model_a['feature_names'].index('pcs_std')
    idx_a_inter = model_a['feature_names'].index('pcs_x_fibroTLR')
    a_coef_pcs = float(model_a['beta'][idx_a_pcs])
    a_coef_inter = float(model_a['beta'][idx_a_inter])
    a_wt = a_coef_pcs + a_coef_inter

    # variance of a_wt = var(pcs_coef) + var(interaction_coef) + 2*cov(pcs,interaction)
    cov_a = model_a['cov_beta']
    var_a_wt = float(cov_a[idx_a_pcs, idx_a_pcs] + cov_a[idx_a_inter, idx_a_inter] + 2 * cov_a[idx_a_pcs, idx_a_inter])
    se_a_wt = sqrt(var_a_wt) if var_a_wt > 0 else np.nan

    # b_coef for fibro_activation_std
    idx_b_fibact = model_b['feature_names'].index('fibro_activation_std')
    b_coef = float(model_b['beta'][idx_b_fibact])
    var_b = float(model_b['cov_beta'][idx_b_fibact, idx_b_fibact])
    se_b = sqrt(var_b) if var_b > 0 else np.nan

    indirect_ab = a_wt * b_coef
    # delta method for SE: sqrt( (b^2)*Var(a_wt) + (a_wt^2)*Var(b) )
    se_indirect = sqrt((b_coef ** 2) * var_a_wt + (a_wt ** 2) * var_b) if (var_a_wt >= 0 and var_b >= 0) else np.nan
    z_sobel = indirect_ab / se_indirect if se_indirect and se_indirect > 0 else np.nan
    p_sobel = two_sided_p_from_z(z_sobel) if not np.isnan(z_sobel) else None

    # 4) Analysis 3: Effect on CKD progression (logistic), adjusting for fibrosis
    df['fibrosis_std'] = (df['fibrosis_score'] - df['fibrosis_score'].mean()) / df['fibrosis_score'].std()
    features_log = ['fibrosis_std', 'pcs_std', 'age', 'sex', 'baseline_eGFR']
    X_log = df[features_log].values
    y_log = df['ckd_progression'].values
    # Fit logistic
    logreg = LogisticRegression(solver='lbfgs', max_iter=200, random_state=RND)
    logreg.fit(X_log, y_log)
    pred_prob = logreg.predict_proba(X_log)[:,1]
    try:
        auc = float(roc_auc_score(y_log, pred_prob))
    except Exception:
        auc = float('nan')

    # Estimate p-value for fibrosis coefficient via bootstrap
    B = 300
    coefs_fib = []
    rng = np.random.RandomState(RND)
    for i in range(B):
        idx = rng.randint(0, n, n)
        Xb = X_log[idx]
        yb = y_log[idx]
        try:
            m = LogisticRegression(solver='lbfgs', max_iter=200)
            m.fit(Xb, yb)
            coefs_fib.append(m.coef_[0][0])
        except Exception:
            # If fitting fails on a bootstrap sample (e.g., separability), skip
            continue
    coefs_fib = np.array(coefs_fib)
    if coefs_fib.size > 5:
        se_coefs_fib = np.std(coefs_fib, ddof=1)
        coef_fib_point = float(logreg.coef_[0][0])
        z_fib = coef_fib_point / se_coefs_fib if se_coefs_fib > 0 else np.nan
        pval_fib_log = two_sided_p_from_z(z_fib) if not np.isnan(z_fib) else None
    else:
        se_coefs_fib = None
        coef_fib_point = float(logreg.coef_[0][0])
        pval_fib_log = None

    # 5) Interaction analyses for macrophage TLR4 as well (linear model)
    df['pcs_x_macTLR'] = df['pcs_std'] * df['tlr4_mac']
    features_mac = ['pcs_std', 'tlr4_mac', 'pcs_x_macTLR', 'age', 'sex', 'baseline_eGFR']
    X_mac = add_intercept(df[features_mac].values)
    y_mac = df['fibrosis_score'].values
    lm_mac = ols_numpy(X_mac, y_mac, feature_names=['intercept'] + features_mac)
    # find interaction index
    try:
        idx_mac_inter = lm_mac['feature_names'].index('pcs_x_macTLR')
        inter_coef_mac = float(lm_mac['beta'][idx_mac_inter])
        inter_pval_mac = float(lm_mac['p_values'][idx_mac_inter])
    except Exception:
        inter_coef_mac = None
        inter_pval_mac = None

    # 6) Simple visualization: PCS vs fibrosis colored by TLR group
    plt.figure(figsize=(8,6))
    colors = {0:'C0',1:'C1',2:'C2',3:'C3'}
    for g in sorted(df['tlr_group'].unique()):
        subset = df[df['tlr_group']==g]
        plt.scatter(subset['pcs'], subset['fibrosis_score'], alpha=0.4, s=10, label=f'group_{g}', color=colors[g])
    plt.xlabel('PCS (arb units)')
    plt.ylabel('Fibrosis score (arb units)')
    plt.legend()
    plt.title('Simulated PCS vs Fibrosis by TLR4 group')
    plot_filename = 'pcs_vs_fibrosis.png'
    plt.tight_layout()
    plt.savefig(plot_filename, dpi=150)
    plt.close()

    # 7) Collect metrics to output
    # Extract pcs and interaction results from lm1
    try:
        idx_pcs = lm1['feature_names'].index('pcs_std')
        idx_inter = lm1['feature_names'].index('pcs_x_fibroTLR')
        lm1_coef_pcs = float(lm1['beta'][idx_pcs])
        lm1_pval_pcs = float(lm1['p_values'][idx_pcs])
        lm1_coef_inter = float(lm1['beta'][idx_inter])
        lm1_pval_inter = float(lm1['p_values'][idx_inter])
    except Exception:
        lm1_coef_pcs = None
        lm1_pval_pcs = None
        lm1_coef_inter = None
        lm1_pval_inter = None

    metrics = {
        'n': int(n),
        'lm_fibrosis_coef_pcs_std': lm1_coef_pcs,
        'lm_fibrosis_pval_pcs_std': lm1_pval_pcs,
        'lm_fibrosis_coef_interaction_pcs_x_fibroTLR': lm1_coef_inter,
        'lm_fibrosis_pval_interaction_pcs_x_fibroTLR': lm1_pval_inter,
        'indirect_effect_via_fibro_activation_for_fibroTLR_present': float(indirect_ab),
        'pval_indirect_sobel': float(p_sobel) if p_sobel is not None else None,
        'fibrosis_model_r2': float(lm1['rsq']),
        'logistic_ckd_coef_fibrosis_std': coef_fib_point,
        'logistic_ckd_pval_fibrosis_std_bootstrap': pval_fib_log,
        'logistic_ckd_auc': auc,
        'lm_mac_coef_interaction_pcs_x_macTLR': inter_coef_mac,
        'lm_mac_pval_interaction_pcs_x_macTLR': inter_pval_mac,
        'plot': plot_filename
    }

except Exception as e:
    # In case of error, return an error JSON
    metrics = {'error': str(e)}

# Final output
print(json.dumps(metrics))
