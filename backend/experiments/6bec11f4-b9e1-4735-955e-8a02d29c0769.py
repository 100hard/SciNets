#!/usr/bin/env python3
import math
import json
import sys
import traceback
import random

try:
    # Use only stdlib to avoid importing heavy site-packages
    rng = random.Random(42)

    # Simulation parameters
    losses_db = [20.0 + i * (60.0 - 20.0) / 16.0 for i in range(17)]  # 20..60 dB, 17 points
    n_losses = len(losses_db)

    # Source/detection parameters
    p_pair = 0.01  # pair generation probability per pulse
    eta_ground = 0.6
    eta_sat_base = 0.5

    # Background parameters (daytime / bright background uplink)
    B_base = 5e-4

    # Polarization-specific noise
    p_dep0 = 0.01
    alpha_dep = 0.015

    # Energy-time thermal open-system dephasing
    gamma0 = 0.005
    beta = 1.5

    # Spectral-temporal engineering parameters
    r_bg = 0.12
    r_sig = 0.92

    eps = 1e-12
    qber_threshold = 0.11

    # Containers
    vis_pol = []
    qber_pol = []
    vis_time = []
    qber_time = []
    vis_time_eng = []
    qber_time_eng = []

    def transmission_from_loss_db(loss_db):
        return 10 ** (-loss_db / 10.0)

    for loss in losses_db:
        T = transmission_from_loss_db(loss)
        eta_sat = eta_sat_base

        S = p_pair * (T * eta_sat) * eta_ground

        # Background increases mildly with loss (heuristic)
        B_sat = B_base * (1.0 + 0.2 * (loss - losses_db[0]) / max(1.0, losses_db[-1] - losses_db[0]))

        # Accidental coincidences: background at satellite * ground singles
        single_ground = p_pair * eta_ground
        N_acc = B_sat * single_ground + B_sat * 1e-6

        # Polarization model
        p_dep = p_dep0 + alpha_dep * (loss / 10.0)
        if p_dep < 0.0:
            p_dep = 0.0
        if p_dep > 0.5:
            p_dep = 0.5
        coherence_pol = 1.0 - p_dep

        V_pol = coherence_pol * S / (S + N_acc + eps)
        errors_from_coherence = (1.0 - coherence_pol) * S * 0.5
        errors_accidental = N_acc * 0.5
        Q_pol = (errors_from_coherence + errors_accidental) / (S + N_acc + eps)

        vis_pol.append(max(0.0, min(1.0, V_pol)))
        qber_pol.append(max(0.0, min(0.5, Q_pol)))

        # Energy-time (time-bin) plain
        n_th = B_sat
        gamma = gamma0 + beta * n_th
        coherence_time = math.exp(-gamma)

        V_time = coherence_time * S / (S + N_acc + eps)
        errors_from_coherence_time = (1.0 - coherence_time) * S * 0.5
        errors_accidental_time = N_acc * 0.5
        Q_time = (errors_from_coherence_time + errors_accidental_time) / (S + N_acc + eps)

        vis_time.append(max(0.0, min(1.0, V_time)))
        qber_time.append(max(0.0, min(0.5, Q_time)))

        # Energy-time with engineering
        B_sat_eng = max(0.0, B_sat * r_bg)
        S_eng = p_pair * (T * eta_sat) * eta_ground * r_sig
        N_acc_eng = B_sat_eng * (p_pair * eta_ground)
        n_th_eng = B_sat_eng
        gamma_eng = gamma0 + beta * n_th_eng
        coherence_time_eng = math.exp(-gamma_eng)

        V_time_eng = coherence_time_eng * S_eng / (S_eng + N_acc_eng + eps)
        errors_from_coherence_time_eng = (1.0 - coherence_time_eng) * S_eng * 0.5
        errors_accidental_time_eng = N_acc_eng * 0.5
        Q_time_eng = (errors_from_coherence_time_eng + errors_accidental_time_eng) / (S_eng + N_acc_eng + eps)

        vis_time_eng.append(max(0.0, min(1.0, V_time_eng)))
        qber_time_eng.append(max(0.0, min(0.5, Q_time_eng)))

    # Compare
    visibility_win = [1 if vis_time_eng[i] > vis_pol[i] + 1e-12 else 0 for i in range(n_losses)]
    qber_win = [1 if qber_time_eng[i] < qber_pol[i] - 1e-12 else 0 for i in range(n_losses)]
    wins_visibility_fraction = sum(visibility_win) / float(n_losses)
    wins_qber_fraction = sum(qber_win) / float(n_losses)

    # Bootstrap (manual)
    def bootstrap_pvalue(diffs, n_boot=2000, rng=rng):
        # one-sided: test mean(diffs) > 0
        n = len(diffs)
        if n == 0:
            return 1.0, 0.0, 0.0
        obs_mean = sum(diffs) / n
        boot_means = []
        for _ in range(n_boot):
            s = 0.0
            for _ in range(n):
                s += rng.choice(diffs)
            boot_means.append(s / n)
        if obs_mean > 0:
            pval = sum(1 for m in boot_means if m <= 0.0) / float(n_boot)
        else:
            pval = sum(1 for m in boot_means if m >= 0.0) / float(n_boot)
        # compute std of diffs
        mean_d = obs_mean
        var = sum((d - mean_d) ** 2 for d in diffs) / (n - 1) if n > 1 else 0.0
        std_d = math.sqrt(var) if var >= 0.0 else 0.0
        return pval, obs_mean, std_d

    diffs_vis = [vis_time_eng[i] - vis_pol[i] for i in range(n_losses)]
    pval_vis, mean_diff_vis, std_diff_vis = bootstrap_pvalue(diffs_vis, n_boot=2000, rng=rng)

    diffs_qber = [qber_pol[i] - qber_time_eng[i] for i in range(n_losses)]
    pval_qber, mean_diff_qber, std_diff_qber = bootstrap_pvalue(diffs_qber, n_boot=2000, rng=rng)

    def max_secure_loss(losses, qber_array, threshold):
        valid_idxs = [i for i, q in enumerate(qber_array) if q < threshold]
        if not valid_idxs:
            return None
        return float(losses[valid_idxs[-1]])

    max_loss_pol = max_secure_loss(losses_db, qber_pol, qber_threshold)
    max_loss_time = max_secure_loss(losses_db, qber_time, qber_threshold)
    max_loss_time_eng = max_secure_loss(losses_db, qber_time_eng, qber_threshold)

    results = {
        "losses_db": [float(x) for x in losses_db],
        "visibility": {
            "polarization": [float(x) for x in vis_pol],
            "timebin_plain": [float(x) for x in vis_time],
            "timebin_engineered": [float(x) for x in vis_time_eng]
        },
        "qber": {
            "polarization": [float(x) for x in qber_pol],
            "timebin_plain": [float(x) for x in qber_time],
            "timebin_engineered": [float(x) for x in qber_time_eng]
        },
        "summary": {
            "visibility_win_fraction_timebin_engineered_vs_pol": wins_visibility_fraction,
            "qber_win_fraction_timebin_engineered_vs_pol": wins_qber_fraction,
            "bootstrap_visibility_pvalue_one_sided_timebin_eng_greater": pval_vis,
            "bootstrap_qber_pvalue_one_sided_timebin_eng_lower": pval_qber,
            "mean_visibility_difference_timebin_eng_minus_pol": mean_diff_vis,
            "std_visibility_difference": std_diff_vis,
            "mean_qber_difference_pol_minus_timebin_eng": mean_diff_qber,
            "std_qber_difference": std_diff_qber
        },
        "max_secure_loss_db": {
            "polarization": max_loss_pol,
            "timebin_plain": max_loss_time,
            "timebin_engineered": max_loss_time_eng
        },
        "parameters": {
            "p_pair": p_pair,
            "eta_ground": eta_ground,
            "eta_sat_base": eta_sat_base,
            "B_base": B_base,
            "p_dep0": p_dep0,
            "alpha_dep": alpha_dep,
            "gamma0": gamma0,
            "beta": beta,
            "r_bg": r_bg,
            "r_sig": r_sig,
            "qber_threshold": qber_threshold
        }
    }

    print(json.dumps(results))

except Exception as e:
    tb = traceback.format_exc()
    err = {
        "error": True,
        "message": str(e),
        "traceback": tb
    }
    print(json.dumps(err))
    sys.exit(1)
