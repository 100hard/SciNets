#!/usr/bin/env python3
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
import json
import sys

np.random.seed(42)

try:
    # 1. Simulate cohort
    n = 1000
    # Baseline lactate levels (log-normal to represent skew)
    lactate = np.random.lognormal(mean=0.5, sigma=0.6, size=n)

    # Site-specific sensitivities (some regulatory sites respond more)
    # promoter is strongly lactylated by lactate; enhancers weaker or noisy
    promoter_sens = 1.2
    enh1_sens = 0.5
    enh2_sens = 0.2

    # Baseline (site-specific) lactylation levels: linear response + noise
    promoter_lactyl = promoter_sens * lactate + np.random.normal(0, 0.5, size=n)
    enh1_lactyl = enh1_sens * lactate + np.random.normal(0, 0.6, size=n)
    enh2_lactyl = enh2_sens * lactate + np.random.normal(0, 0.7, size=n)

    # PD-L1 expression (continuous) is driven primarily by promoter lactylation
    # plus smaller contributions from enhancers
    pd_l1 = 0.5 + 0.9 * promoter_lactyl + 0.2 * enh1_lactyl + 0.1 * enh2_lactyl + np.random.normal(0, 1.0, size=n)

    # T-cell infiltration/function: decreased by PD-L1 (immune evasion), plus noise
    tcell = 3.0 - 0.6 * pd_l1 + np.random.normal(0, 1.0, size=n)

    # Response to anti-PD-1: logistic function of T-cell function and PD-L1 (here elevated PD-L1 contributes to immune evasion, lowering response)
    # We assume T-cell positive effect, PD-L1 negative effect on response probability
    logits = -1.0 + 0.8 * tcell - 0.4 * pd_l1
    prob_resp = 1 / (1 + np.exp(-logits))
    response = np.random.binomial(1, prob_resp, size=n)

    # Assemble DataFrame
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

    # 2. Interventions (counterfactuals applied to same individuals)
    # Intervention A: Reduce lactate levels (e.g., metabolic therapy) by 50%
    lactate_reduced = lactate * 0.5
    promoter_lactyl_red = promoter_sens * lactate_reduced + np.random.normal(0, 0.5, size=n)
    enh1_lactyl_red = enh1_sens * lactate_reduced + np.random.normal(0, 0.6, size=n)
    enh2_lactyl_red = enh2_sens * lactate_reduced + np.random.normal(0, 0.7, size=n)
    pd_l1_red = 0.5 + 0.9 * promoter_lactyl_red + 0.2 * enh1_lactyl_red + 0.1 * enh2_lactyl_red + np.random.normal(0, 1.0, size=n)
    tcell_red = 3.0 - 0.6 * pd_l1_red + np.random.normal(0, 1.0, size=n)
    logits_red = -1.0 + 0.8 * tcell_red - 0.4 * pd_l1_red
    prob_resp_red = 1 / (1 + np.exp(-logits_red))
    response_red = np.random.binomial(1, prob_resp_red, size=n)

    # Intervention B: Block lactylation at promoter specifically (e.g., inhibitor of lactylation writers or blocking site)
    # Here we set promoter lactylation to baseline noise (remove lactate->promoter effect)
    promoter_lactyl_block = np.random.normal(0, 0.5, size=n)  # noise-only promoter lactylation
    enh1_lactyl_block = enh1_sens * lactate + np.random.normal(0, 0.6, size=n)  # enhancers unchanged
    enh2_lactyl_block = enh2_sens * lactate + np.random.normal(0, 0.7, size=n)
    pd_l1_block = 0.5 + 0.9 * promoter_lactyl_block + 0.2 * enh1_lactyl_block + 0.1 * enh2_lactyl_block + np.random.normal(0, 1.0, size=n)
    tcell_block = 3.0 - 0.6 * pd_l1_block + np.random.normal(0, 1.0, size=n)
    logits_block = -1.0 + 0.8 * tcell_block - 0.4 * pd_l1_block
    prob_resp_block = 1 / (1 + np.exp(-logits_block))
    response_block = np.random.binomial(1, prob_resp_block, size=n)

    # Add counterfactuals to DataFrame
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

    # 3. Statistical analyses
    results = {}

    # Correlation: lactate vs promoter lactylation
    r, p_corr = stats.pearsonr(df['lactate'], df['promoter_lactyl'])
    results['lactate_promoter_corr_r'] = float(r)
    results['lactate_promoter_corr_p'] = float(p_corr)

    # Regression: promoter_lactyl ~ lactate
    X = sm.add_constant(df['lactate'])
    model_a = sm.OLS(df['promoter_lactyl'], X).fit()
    a_coef = float(model_a.params['lactate'])
    a_se = float(model_a.bse['lactate'])
    results['reg_promoter_on_lactate_coef'] = a_coef
    results['reg_promoter_on_lactate_p'] = float(model_a.pvalues['lactate'])

    # Regression: pd_l1 ~ promoter_lactyl (b path)
    Xb = sm.add_constant(df['promoter_lactyl'])
    model_b = sm.OLS(df['pd_l1'], Xb).fit()
    b_coef = float(model_b.params['promoter_lactyl'])
    b_se = float(model_b.bse['promoter_lactyl'])
    results['reg_pd_l1_on_promoter_coef'] = b_coef
    results['reg_pd_l1_on_promoter_p'] = float(model_b.pvalues['promoter_lactyl'])

    # Total effect: pd_l1 ~ lactate (c path)
    Xc = sm.add_constant(df['lactate'])
    model_c = sm.OLS(df['pd_l1'], Xc).fit()
    c_coef = float(model_c.params['lactate'])
    results['reg_pd_l1_on_lactate_coef'] = c_coef
    results['reg_pd_l1_on_lactate_p'] = float(model_c.pvalues['lactate'])

    # Mediation: indirect = a*b. Sobel test
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

    # 4. Intervention effects: paired comparisons (each individual has control and counterfactual)
    # PD-L1 mean reductions
    pd_l1_mean = df['pd_l1'].mean()
    pd_l1_mean_red = df['pd_l1_red'].mean()
    pd_l1_mean_block = df['pd_l1_block'].mean()
    results['pd_l1_mean_control'] = float(pd_l1_mean)
    results['pd_l1_mean_reduced_lactate'] = float(pd_l1_mean_red)
    results['pd_l1_mean_blocked_lactylation'] = float(pd_l1_mean_block)

    # Paired t-tests
    t_red, p_red = stats.ttest_rel(df['pd_l1'], df['pd_l1_red'])
    t_block, p_block = stats.ttest_rel(df['pd_l1'], df['pd_l1_block'])
    results['pd_l1_reduction_t_stat_reduced_lactate'] = float(t_red)
    results['pd_l1_reduction_p_reduced_lactate'] = float(p_red)
    results['pd_l1_reduction_t_stat_blocked_lactylation'] = float(t_block)
    results['pd_l1_reduction_p_blocked_lactylation'] = float(p_block)

    # Response rate changes and permutation (sign-flip) test for paired binary responses
    resp_control = df['response'].values
    resp_red = df['response_red'].values
    resp_block = df['response_block'].values

    rate_control = resp_control.mean()
    rate_red = resp_red.mean()
    rate_block = resp_block.mean()
    results['response_rate_control'] = float(rate_control)
    results['response_rate_reduced_lactate'] = float(rate_red)
    results['response_rate_blocked_lactylation'] = float(rate_block)
    results['response_rate_abs_change_reduced_lactate'] = float(rate_red - rate_control)
    results['response_rate_abs_change_blocked_lactylation'] = float(rate_block - rate_control)

    # Sign-flip permutation test for mean paired difference
    def paired_sign_flip_pvalue(diff_array, n_permutations=5000):
        obs = diff_array.mean()
        rng = np.random.default_rng(123)
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

    # Effect sizes (Cohen's d) for PD-L1 change (paired)
    def cohens_d_paired(a, b):
        diff = a - b
        return diff.mean() / diff.std(ddof=1)

    d_pd_red = cohens_d_paired(df['pd_l1'], df['pd_l1_red'])
    d_pd_block = cohens_d_paired(df['pd_l1'], df['pd_l1_block'])
    results['cohens_d_pd_l1_reduced_lactate'] = float(d_pd_red)
    results['cohens_d_pd_l1_blocked_lactylation'] = float(d_pd_block)

    # 5. Additional checks: regress response on PD-L1 and T-cell to show directionality
    # Logistic regression (statsmodels discrete)
    df_log = df.copy()
    df_log['response'] = df_log['response'].astype(int)
    # Standardize predictors
    df_log['pd_l1_z'] = (df_log['pd_l1'] - df_log['pd_l1'].mean()) / df_log['pd_l1'].std()
    df_log['tcell_z'] = (df_log['tcell'] - df_log['tcell'].mean()) / df_log['tcell'].std()

    logit_mod = sm.Logit(df_log['response'], sm.add_constant(df_log[['pd_l1_z', 'tcell_z']])).fit(disp=False)
    # coefficients
    results['logit_coef_pd_l1_z'] = float(logit_mod.params['pd_l1_z'])
    results['logit_p_pd_l1_z'] = float(logit_mod.pvalues['pd_l1_z'])
    results['logit_coef_tcell_z'] = float(logit_mod.params['tcell_z'])
    results['logit_p_tcell_z'] = float(logit_mod.pvalues['tcell_z'])

    # Final metrics JSON
    metrics = results

    print(json.dumps(metrics))

except Exception as e:
    # If any error occurs, provide diagnostic and exit with JSON containing error
    err = {'error': str(e)}
    print(json.dumps(err))
    sys.exit(1)
