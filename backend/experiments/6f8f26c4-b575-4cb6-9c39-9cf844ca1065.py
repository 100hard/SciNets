#!/usr/bin/env python3
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
import json
import sys

np.random.seed(42)

# Helper: OLS regression (intercept + single or multiple predictors)
def ols_regression(y, X):
    # X should be 2D design matrix without intercept; we'll add intercept inside
    y = np.asarray(y).reshape(-1, 1)
    X = np.asarray(X)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    n = X.shape[0]
    X_design = np.hstack([np.ones((n, 1)), X])
    # coefficients via pseudoinverse for stability
    XtX = X_design.T.dot(X_design)
    try:
        XtX_inv = np.linalg.inv(XtX)
    except np.linalg.LinAlgError:
        XtX_inv = np.linalg.pinv(XtX)
    beta = XtX_inv.dot(X_design.T).dot(y).flatten()
    y_pred = X_design.dot(beta)
    resid = (y.flatten() - y_pred)
    p = X_design.shape[1]
    rss = np.sum(resid ** 2)
    dof = max(n - p, 1)
    s2 = rss / dof
    cov_beta = s2 * XtX_inv
    se = np.sqrt(np.abs(np.diag(cov_beta)))  # protect small negatives from numeric
    t_stats = beta / se
    p_values = 2 * stats.t.sf(np.abs(t_stats), df=dof)
    return {
        'beta': beta,            # array, intercept first
        'se': se,
        't': t_stats,
        'p': p_values,
        'resid': resid,
        'y_pred': y_pred.flatten(),
        's2': s2,
        'cov_beta': cov_beta
    }

# Helper: compute logistic regression with approximate Wald p-values
def logistic_regression_with_pvalues(y, X, C=1e6, max_iter=1000):
    # X should include intercept column if desired; we'll add it
    y = np.asarray(y).ravel()
    X = np.asarray(X)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    n = X.shape[0]
    X_design = np.hstack([np.ones((n, 1)), X])
    # Fit sklearn logistic (with large C to approximate unregularized)
    try:
        model = LogisticRegression(fit_intercept=False, C=C, solver='lbfgs', max_iter=max_iter)
        model.fit(X_design, y)
    except Exception:
        # fallback: add small regularization by using default intercept handling
        model = LogisticRegression(C=C, solver='lbfgs', max_iter=max_iter)
        model.fit(X, y)
        # reconstruct X_design accordingly
        X_design = np.hstack([np.ones((n, 1)), X])
        coef = np.hstack([model.intercept_, model.coef_.ravel()])
        model_coef = coef
    coef = np.hstack([model.coef_.ravel()])
    # sklearn with fit_intercept=False returns coef including intercept at position 0
    if coef.shape[0] == X_design.shape[1]:
        params = coef
    else:
        # if model used fit_intercept=True, assemble params
        params = np.hstack([model.intercept_, model.coef_.ravel()])
    # predicted probabilities
    linpred = X_design.dot(params)
    p = 1.0 / (1.0 + np.exp(-linpred))
    # Fisher information approx: X^T W X
    W = p * (1 - p)
    # avoid zero weights
    W = np.clip(W, 1e-8, None)
    WX = X_design.T * W
    XtWX = WX.dot(X_design)
    try:
        cov = np.linalg.inv(XtWX)
    except np.linalg.LinAlgError:
        cov = np.linalg.pinv(XtWX)
    se = np.sqrt(np.abs(np.diag(cov)))
    z = params / se
    pvals = 2 * (1 - stats.norm.cdf(np.abs(z)))
    return {
        'params': params,
        'se': se,
        'z': z,
        'p': pvals,
        'p_pred': p,
        'cov': cov
    }


try:
    # 1. Simulate cohort (as in the first version)
    n = 1000
    lactate = np.random.lognormal(mean=0.5, sigma=0.6, size=n)

    promoter_sens = 1.2
    enh1_sens = 0.5
    enh2_sens = 0.2

    promoter_lactyl = promoter_sens * lactate + np.random.normal(0, 0.5, size=n)
    enh1_lactyl = enh1_sens * lactate + np.random.normal(0, 0.6, size=n)
    enh2_lactyl = enh2_sens * lactate + np.random.normal(0, 0.7, size=n)

    pd_l1 = 0.5 + 0.9 * promoter_lactyl + 0.2 * enh1_lactyl + 0.1 * enh2_lactyl + np.random.normal(0, 1.0, size=n)

    tcell = 3.0 - 0.6 * pd_l1 + np.random.normal(0, 1.0, size=n)

    logits = -1.0 + 0.8 * tcell - 0.4 * pd_l1
    prob_resp = 1 / (1 + np.exp(-logits))
    response = np.random.binomial(1, prob_resp, size=n)

    df = pd.DataFrame({
        'lactate': lactate,
        'promoter_lactyl': promoter_lactyl,
        'enh1_lactyl': enh1_lactyl,
        'enh2_lactyl': enh2_lactyl,
        'pd_l1': pd_l1,
        'tcell': tcell,
        'response': response,
        'prob_resp': prob_resp
    })

    # 2. Interventions
    # A: reduce lactate by 50%
    lactate_reduced = lactate * 0.5
    promoter_lactyl_red = promoter_sens * lactate_reduced + np.random.normal(0, 0.5, size=n)
    enh1_lactyl_red = enh1_sens * lactate_reduced + np.random.normal(0, 0.6, size=n)
    enh2_lactyl_red = enh2_sens * lactate_reduced + np.random.normal(0, 0.7, size=n)
    pd_l1_red = 0.5 + 0.9 * promoter_lactyl_red + 0.2 * enh1_lactyl_red + 0.1 * enh2_lactyl_red + np.random.normal(0, 1.0, size=n)
    tcell_red = 3.0 - 0.6 * pd_l1_red + np.random.normal(0, 1.0, size=n)
    logits_red = -1.0 + 0.8 * tcell_red - 0.4 * pd_l1_red
    prob_resp_red = 1 / (1 + np.exp(-logits_red))
    response_red = np.random.binomial(1, prob_resp_red, size=n)

    # B: block promoter lactylation
    promoter_lactyl_block = np.random.normal(0, 0.5, size=n)
    enh1_lactyl_block = enh1_sens * lactate + np.random.normal(0, 0.6, size=n)
    enh2_lactyl_block = enh2_sens * lactate + np.random.normal(0, 0.7, size=n)
    pd_l1_block = 0.5 + 0.9 * promoter_lactyl_block + 0.2 * enh1_lactyl_block + 0.1 * enh2_lactyl_block + np.random.normal(0, 1.0, size=n)
    tcell_block = 3.0 - 0.6 * pd_l1_block + np.random.normal(0, 1.0, size=n)
    logits_block = -1.0 + 0.8 * tcell_block - 0.4 * pd_l1_block
    prob_resp_block = 1 / (1 + np.exp(-logits_block))
    response_block = np.random.binomial(1, prob_resp_block, size=n)

    df['promoter_lactyl_red'] = promoter_lactyl_red
    df['pd_l1_red'] = pd_l1_red
    df['tcell_red'] = tcell_red
    df['response_red'] = response_red
    df['prob_resp_red'] = prob_resp_red

    df['promoter_lactyl_block'] = promoter_lactyl_block
    df['pd_l1_block'] = pd_l1_block
    df['tcell_block'] = tcell_block
    df['response_block'] = response_block
    df['prob_resp_block'] = prob_resp_block

    # 3. Analyses
    results = {}

    # Correlation lactate vs promoter lactyl
    r, p_corr = stats.pearsonr(df['lactate'], df['promoter_lactyl'])
    results['lactate_promoter_corr_r'] = float(r)
    results['lactate_promoter_corr_p'] = float(p_corr)

    # OLS regressions
    reg_a = ols_regression(df['promoter_lactyl'], df['lactate'])
    a_coef = float(reg_a['beta'][1])
    a_se = float(reg_a['se'][1])
    results['reg_promoter_on_lactate_coef'] = a_coef
    results['reg_promoter_on_lactate_p'] = float(reg_a['p'][1])

    reg_b = ols_regression(df['pd_l1'], df['promoter_lactyl'])
    b_coef = float(reg_b['beta'][1])
    b_se = float(reg_b['se'][1])
    results['reg_pd_l1_on_promoter_coef'] = b_coef
    results['reg_pd_l1_on_promoter_p'] = float(reg_b['p'][1])

    reg_c = ols_regression(df['pd_l1'], df['lactate'])
    c_coef = float(reg_c['beta'][1])
    results['reg_pd_l1_on_lactate_coef'] = c_coef
    results['reg_pd_l1_on_lactate_p'] = float(reg_c['p'][1])

    # Mediation (Sobel)
    indirect = a_coef * b_coef
    se_indirect = np.sqrt((b_coef ** 2) * (a_se ** 2) + (a_coef ** 2) * (b_se ** 2))
    if se_indirect == 0:
        z_sobel = 0.0
        p_sobel = 1.0
    else:
        z_sobel = indirect / se_indirect
        p_sobel = 2 * (1 - stats.norm.cdf(abs(z_sobel)))
    results['mediation_indirect_effect'] = float(indirect)
    results['mediation_sobel_z'] = float(z_sobel)
    results['mediation_sobel_p'] = float(p_sobel)

    # Intervention mean PD-L1
    results['pd_l1_mean_control'] = float(df['pd_l1'].mean())
    results['pd_l1_mean_reduced_lactate'] = float(df['pd_l1_red'].mean())
    results['pd_l1_mean_blocked_lactylation'] = float(df['pd_l1_block'].mean())

    # Paired t-tests for PD-L1
    t_red, p_red = stats.ttest_rel(df['pd_l1'], df['pd_l1_red'])
    t_block, p_block = stats.ttest_rel(df['pd_l1'], df['pd_l1_block'])
    results['pd_l1_reduction_t_stat_reduced_lactate'] = float(t_red)
    results['pd_l1_reduction_p_reduced_lactate'] = float(p_red)
    results['pd_l1_reduction_t_stat_blocked_lactylation'] = float(t_block)
    results['pd_l1_reduction_p_blocked_lactylation'] = float(p_block)

    # Response rates
    resp_control = df['response'].values
    resp_red = df['response_red'].values
    resp_block = df['response_block'].values

    rate_control = float(resp_control.mean())
    rate_red = float(resp_red.mean())
    rate_block = float(resp_block.mean())
    results['response_rate_control'] = rate_control
    results['response_rate_reduced_lactate'] = rate_red
    results['response_rate_blocked_lactylation'] = rate_block
    results['response_rate_abs_change_reduced_lactate'] = rate_red - rate_control
    results['response_rate_abs_change_blocked_lactylation'] = rate_block - rate_control

    # Permutation sign-flip for paired binary differences
    def paired_sign_flip_pvalue(diff_array, n_permutations=5000, seed=123):
        obs = diff_array.mean()
        rng = np.random.default_rng(seed)
        count = 0
        for _ in range(n_permutations):
            signs = rng.choice([-1, 1], size=diff_array.shape[0])
            perm_mean = (signs * diff_array).mean()
            if abs(perm_mean) >= abs(obs):
                count += 1
        return (count + 1) / (n_permutations + 1)

    diff_red = resp_red - resp_control
    diff_block = resp_block - resp_control
    p_resp_red = paired_sign_flip_pvalue(diff_red)
    p_resp_block = paired_sign_flip_pvalue(diff_block)
    results['response_rate_change_p_reduced_lactate'] = float(p_resp_red)
    results['response_rate_change_p_blocked_lactylation'] = float(p_resp_block)

    # Effect sizes (Cohen's d for paired)
    def cohens_d_paired(a, b):
        diff = np.asarray(a) - np.asarray(b)
        return diff.mean() / diff.std(ddof=1)

    results['cohens_d_pd_l1_reduced_lactate'] = float(cohens_d_paired(df['pd_l1'], df['pd_l1_red']))
    results['cohens_d_pd_l1_blocked_lactylation'] = float(cohens_d_paired(df['pd_l1'], df['pd_l1_block']))

    # Logistic regression: response on pd_l1 and tcell
    df_log = df.copy()
    df_log['pd_l1_z'] = (df_log['pd_l1'] - df_log['pd_l1'].mean()) / df_log['pd_l1'].std()
    df_log['tcell_z'] = (df_log['tcell'] - df_log['tcell'].mean()) / df_log['tcell'].std()

    log_res = logistic_regression_with_pvalues(df_log['response'].values, df_log[['pd_l1_z', 'tcell_z']].values)
    # params: intercept, pd_l1_z, tcell_z
    params = log_res['params']
    pvals = log_res['p']
    # store pd_l1 coefficient (index 1) and tcell (index 2)
    results['logit_coef_pd_l1_z'] = float(params[1])
    results['logit_p_pd_l1_z'] = float(pvals[1])
    results['logit_coef_tcell_z'] = float(params[2])
    results['logit_p_tcell_z'] = float(pvals[2])

    # Package and print JSON
    print(json.dumps(results))

except Exception as e:
    err = {'error': str(e)}
    print(json.dumps(err))
    sys.exit(1)
