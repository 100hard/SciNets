#!/usr/bin/env python3
import numpy as np
import json
import sys
import traceback

try:
    # Seed for reproducibility
    rng = np.random.default_rng(42)

    # Simulation parameters
    losses_db = np.linspace(20.0, 60.0, 17)  # channel loss in dB (uplink)
    n_losses = len(losses_db)

    # Source/detection parameters
    p_pair = 0.01  # pair generation probability per pulse
    eta_ground = 0.6  # ground detection efficiency (local detector)
    eta_sat_base = 0.5  # satellite detector efficiency (intrinsic)
    pulse_rate = 1e6  # pulses per second (not used directly but for rates intuition)

    # Background parameters (daytime / bright background uplink)
    # background probability per detection window at satellite (without filtering)
    B_base = 5e-4  # baseline background probability per pulse/window (daytime bright)

    # Polarization-specific noise: depolarization increases with channel degradation
    p_dep0 = 0.01  # base depolarization
    alpha_dep = 0.015  # growth per (dB/10) unit (tunable)

    # Time-bin (energy-time) thermal open-system dephasing model
    # gamma = gamma0 + beta * n_th ; we model n_th proportional to B_sat
    gamma0 = 0.005
    beta = 1.5  # scaling from background occupancy to dephasing

    # Spectral-temporal engineering parameters (applied to time-bin protocol)
    # This reduces background by r_bg and reduces signal throughput by r_sig
    r_bg = 0.12  # background reduction factor (e.g., narrowband filtering + gating)
    r_sig = 0.92  # signal throughput factor (some loss due to filtering)

    # Small epsilon to avoid divide-by-zero
    eps = 1e-12

    # QBER threshold for secure key extraction (approx. BB84-like one-way threshold)
    qber_threshold = 0.11

    # Containers for metrics
    vis_pol = np.zeros(n_losses)
    qber_pol = np.zeros(n_losses)
    vis_time = np.zeros(n_losses)
    qber_time = np.zeros(n_losses)
    vis_time_eng = np.zeros(n_losses)
    qber_time_eng = np.zeros(n_losses)

    # Analytical/model functions
    def transmission_from_loss_db(loss_db):
        # Transmission fraction due to channel loss
        return 10 ** (-loss_db / 10.0)

    for i, loss in enumerate(losses_db):
        T = transmission_from_loss_db(loss)
        # Effective satellite optical throughput
        eta_sat = eta_sat_base

        # True coincidence (both photons detected) probability per pulse (ideal pair)
        # signal coincidence rate S = p_pair * (transmission * eta_sat) * (eta_ground)
        S = p_pair * (T * eta_sat) * eta_ground

        # Satellite background probability per pulse (before engineering)
        B_sat = B_base * (1.0 + 0.2 * (loss - losses_db[0]) / max(1.0, losses_db[-1] - losses_db[0]))
        # we allow some increase of background with larger collected aperture/noise conditions

        # Accidental coincidences approximated by background at satellite coinciding with ground singles
        # ground singles roughly p_pair * eta_ground (from generated pair) + negligible other background
        single_ground = p_pair * eta_ground
        N_acc = B_sat * single_ground + B_sat * 1e-6  # include tiny ground background second term

        # --------- Polarization model ---------
        # Depolarization probability increases with loss (e.g., turbulence-induced mixing)
        p_dep = p_dep0 + alpha_dep * (loss / 10.0)
        p_dep = min(0.5, max(0.0, p_dep))
        coherence_pol = 1.0 - p_dep  # coherence-like factor

        # Observed visibility: coherence times fraction true coincidences in total coincidences
        V_pol = coherence_pol * S / (S + N_acc + eps)

        # QBER: errors come from depolarization and accidental coincidences (assumed random -> 50% errors)
        errors_from_coherence = (1.0 - coherence_pol) * S * 0.5 * 2.0  # factorization to reflect mapping to bit errors
        # The simple physical model: fraction (1-coherence) of true coincidences produce random outcomes -> 50% error
        errors_from_coherence = (1.0 - coherence_pol) * S * 0.5
        errors_accidental = N_acc * 0.5
        Q_pol = (errors_from_coherence + errors_accidental) / (S + N_acc + eps)

        vis_pol[i] = float(np.clip(V_pol, 0.0, 1.0))
        qber_pol[i] = float(np.clip(Q_pol, 0.0, 0.5))

        # --------- Energy-time (time-bin) without engineering ---------
        # Thermal occupancy maps to background B_sat; dephasing gamma increases with B_sat
        n_th = B_sat  # proxy thermal occupancy per window
        gamma = gamma0 + beta * n_th
        coherence_time = np.exp(-gamma)

        V_time = coherence_time * S / (S + N_acc + eps)
        errors_from_coherence_time = (1.0 - coherence_time) * S * 0.5
        errors_accidental_time = N_acc * 0.5
        Q_time = (errors_from_coherence_time + errors_accidental_time) / (S + N_acc + eps)

        vis_time[i] = float(np.clip(V_time, 0.0, 1.0))
        qber_time[i] = float(np.clip(Q_time, 0.0, 0.5))

        # --------- Energy-time with spectral-temporal engineering ---------
        # Engineering reduces background and slightly reduces signal throughput
        B_sat_eng = max(0.0, B_sat * r_bg)
        S_eng = p_pair * (T * eta_sat) * eta_ground * r_sig

        # Recompute accidental coincidences
        N_acc_eng = B_sat_eng * (p_pair * eta_ground)

        n_th_eng = B_sat_eng
        gamma_eng = gamma0 + beta * n_th_eng
        coherence_time_eng = np.exp(-gamma_eng)

        V_time_eng = coherence_time_eng * S_eng / (S_eng + N_acc_eng + eps)
        errors_from_coherence_time_eng = (1.0 - coherence_time_eng) * S_eng * 0.5
        errors_accidental_time_eng = N_acc_eng * 0.5
        Q_time_eng = (errors_from_coherence_time_eng + errors_accidental_time_eng) / (S_eng + N_acc_eng + eps)

        vis_time_eng[i] = float(np.clip(V_time_eng, 0.0, 1.0))
        qber_time_eng[i] = float(np.clip(Q_time_eng, 0.0, 0.5))

    # Summary comparisons across losses
    # Determine at each loss whether engineered time-bin yields better visibility and lower QBER
    visibility_win = vis_time_eng > vis_pol + 1e-6
    qber_win = qber_time_eng < qber_pol - 1e-9

    wins_visibility_fraction = float(np.mean(visibility_win))
    wins_qber_fraction = float(np.mean(qber_win))

    # Statistical test (bootstrap on paired differences)
    def bootstrap_pvalue(diffs, n_boot=5000, rng=rng):
        # One-sided test: H0 mean_diff <= 0 ; Ha mean_diff > 0
        obs_mean = np.mean(diffs)
        # bootstrap-resample diffs with replacement to compute distribution of mean
        boot_means = rng.choice(diffs, size=(n_boot, len(diffs)), replace=True).mean(axis=1)
        # p-value = fraction of bootstrap means <= 0 (if obs_mean > 0)
        if obs_mean > 0:
            pval = float(np.mean(boot_means <= 0.0))
        else:
            # if obs not positive, p-value near 1 for one-sided >0 test
            pval = float(np.mean(boot_means >= 0.0))
        return pval, float(obs_mean), float(np.std(diffs, ddof=1))

    # Visibility comparison (time-bin engineered minus polarization)
    diffs_vis = vis_time_eng - vis_pol
    pval_vis, mean_diff_vis, std_diff_vis = bootstrap_pvalue(diffs_vis, n_boot=5000)

    # QBER comparison: define diffs_qber = qber_pol - qber_time_eng (positive means time-bin eng has lower QBER)
    diffs_qber = qber_pol - qber_time_eng
    pval_qber, mean_diff_qber, std_diff_qber = bootstrap_pvalue(diffs_qber, n_boot=5000)

    # Compute maximum secure loss (largest loss where QBER < threshold) for each protocol
    def max_secure_loss(losses, qber_array, threshold):
        # Return max loss in dB where qber < threshold; if none, return None
        valid = np.where(qber_array < threshold)[0]
        if len(valid) == 0:
            return None
        else:
            idx = valid[-1]
            return float(losses[idx])

    max_loss_pol = max_secure_loss(losses_db, qber_pol, qber_threshold)
    max_loss_time = max_secure_loss(losses_db, qber_time, qber_threshold)
    max_loss_time_eng = max_secure_loss(losses_db, qber_time_eng, qber_threshold)

    # Prepare results JSON
    results = {
        "losses_db": list(map(float, losses_db)),
        "visibility": {
            "polarization": list(map(float, vis_pol)),
            "timebin_plain": list(map(float, vis_time)),
            "timebin_engineered": list(map(float, vis_time_eng))
        },
        "qber": {
            "polarization": list(map(float, qber_pol)),
            "timebin_plain": list(map(float, qber_time)),
            "timebin_engineered": list(map(float, qber_time_eng))
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
    # If any error occurs, analyze and provide traceback, then exit with JSON error object
    tb = traceback.format_exc()
    err = {
        "error": True,
        "message": str(e),
        "traceback": tb
    }
    print(json.dumps(err))
    sys.exit(1)
