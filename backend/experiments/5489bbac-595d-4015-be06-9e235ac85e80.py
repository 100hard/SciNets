#!/usr/bin/env python3
import numpy as np
import json
import time

# Monte Carlo simulation to evaluate heuristic Holevo upper bounds for:
# - Continuous thermal-state central-broadcast (CB) with HBT g(2) monitoring
# - Entanglement-based Bell tests (EB) (modified E91)
# - A hybrid protocol that alternates both and uses joint constraints

# The script synthesizes trials across uplink loss 10-30 dB and realistic
# detector imperfections/background, maps observed g2 and S to Holevo
# upper bounds via physically-motivated heuristic formulas, and evaluates
# whether the hybrid gives a strictly tighter operational upper bound
# (and reduces privacy-amplification budget) by >=20% in a measurable
# fraction of cases.

np.random.seed(42)

# Parameters
N_TRIALS = 4000  # number of Monte Carlo trials
LOSS_MIN_DB = 10.0
LOSS_MAX_DB = 30.0
S_MAX = 2.0 * np.sqrt(2)  # Tsirelson bound for Bell-CHSH

# Heuristic mapping hyperparameters (tunable to reflect reasonable physics-like behavior)
# These coefficients convert observed deviations and losses/noise into an estimated Holevo bound (bits per signal)
A_CB = 0.6   # weight of loss contribution for central-broadcast
B_CB = 0.85  # weight of g2 deviation contribution
C_CB = 0.25  # weight of background/noise contribution

A_EB = 0.6   # weight of loss contribution for entanglement-based
B_EB = 0.95  # weight of Bell-S deviation contribution
C_EB = 0.25  # weight of background/noise contribution

GAMMA_JOINT = 0.45  # joint reduction strength when deviations correlate (0..1)

# Detector/background parameter ranges (realistic for ground-to-CubeSat uplink, detection-only CubeSat)
ETA_MIN = 0.01   # detector+optics effective coupling efficiency lower bound
ETA_MAX = 0.18   # upper bound (small CubeSat aperture + coupling)
DARK_RATE_MIN = 10.0   # dark counts per second equivalent (normalized unit)
DARK_RATE_MAX = 200.0
BACKGROUND_MIN = 0.0   # background photons per pulse equivalent (normalized)
BACKGROUND_MAX = 5.0

# For mapping, we normalize loss contribution by max span (LOSS_MAX_DB-LOSS_MIN_DB)
LOSS_SPAN = max(LOSS_MAX_DB - LOSS_MIN_DB, 1.0)

# Simulation functions

def simulate_observed_statistics(loss_db, eta, dark_rate, background):
    """Simulate observed g2 (thermal HBT) and Bell S under given channel params.
    Returns (g2_obs, S_obs, bg_norm, signal_transmission)
    """
    # Transmission probability / signal strength scaling
    # Free-space uplink loss factor (linear scale). 10^(-dB/10).
    tx = 10 ** (-loss_db / 10.0)
    # Effective detected signal probability (proportional to eta * tx)
    detected_signal = eta * tx

    # Background contribution normalized to detected signal
    # background is expressed in same units as counts per pulse; convert to comparable scale
    bg = background + dark_rate * 1e-4  # scale dark rate to per-pulse-equivalent (small)
    # Signal-to-background ratio
    s2b = detected_signal / (detected_signal + bg + 1e-12)

    # For thermal light without contamination g2 = 2. Contamination (background, loss) pulls towards 1.
    # Use a physically-motivated interpolation: g2_obs = 1 + (2-1) * SNR_fraction
    g2_mean = 1.0 + 1.0 * s2b
    # Add measurement noise
    g2_obs = g2_mean + np.random.normal(0, 0.03)
    g2_obs = float(np.clip(g2_obs, 1.0, 2.5))

    # For Bell tests, ideal maximal value is S_MAX. Visibility degrades with loss & background.
    # We'll model visibility as proportional to s2b and an additional exponential degradation with loss.
    visibility = s2b * np.exp(-loss_db / 40.0)  # loss acts to reduce coherence/visibility
    # Allow some Bell-specific system error (misalignment)
    misalignment = np.random.normal(0, 0.03)

    S_mean = S_MAX * max(0.0, visibility - 0.02 + misalignment)
    S_obs = float(np.clip(S_mean + np.random.normal(0, 0.04), 0.0, S_MAX))

    # Normalize background metric for mapping to Holevo (0..1)
    bg_norm = float(np.clip(bg / (bg + detected_signal + 1e-12), 0.0, 1.0))

    return g2_obs, S_obs, bg_norm, detected_signal


def holevo_from_cb(g2_obs, loss_db, bg_norm):
    """Heuristic upper bound mapping for central-broadcast based on g2 deviations and loss/noise."""
    loss_term = A_CB * ((loss_db - LOSS_MIN_DB) / LOSS_SPAN)
    g2_term = B_CB * max(0.0, (2.0 - g2_obs) / 1.0)  # normalized deviation (2->1 maps to 1)
    noise_term = C_CB * bg_norm
    raw = loss_term + g2_term + noise_term
    return float(np.clip(raw, 0.0, 1.0))


def holevo_from_eb(S_obs, loss_db, bg_norm):
    """Heuristic upper bound mapping for entanglement-based protocol based on S deviation and loss/noise."""
    loss_term = A_EB * ((loss_db - LOSS_MIN_DB) / LOSS_SPAN)
    # Normalize S deviation: S_max maps to 0, classical bound 2 maps to max
    s_range_norm = (S_MAX - 2.0)
    s_dev = max(0.0, (S_MAX - S_obs) / (s_range_norm + 1e-12))
    s_term = B_EB * s_dev
    noise_term = C_EB * bg_norm
    raw = loss_term + s_term + noise_term
    return float(np.clip(raw, 0.0, 1.0))


def hybrid_holevo(k_cb, k_eb, g2_obs, S_obs):
    """Combine the two upper bounds and apply a joint reduction when deviations correlate.
    Returns a heuristic reduced Holevo upper bound for the hybrid protocol.
    """
    base = min(k_cb, k_eb)
    # Correlation factor: product of normalized deviations (0..1)
    g2_dev_norm = max(0.0, (2.0 - g2_obs) / 1.0)  # 0..1
    s_dev_norm = max(0.0, (S_MAX - S_obs) / (S_MAX - 2.0 + 1e-12))  # 0..1
    corr = g2_dev_norm * s_dev_norm
    reduction = GAMMA_JOINT * corr
    kh = base * (1.0 - reduction)
    return float(np.clip(kh, 0.0, 1.0))


# Run Monte Carlo
start_time = time.time()
results = []

try:
    for i in range(N_TRIALS):
        loss_db = np.random.uniform(LOSS_MIN_DB, LOSS_MAX_DB)
        eta = np.random.uniform(ETA_MIN, ETA_MAX)
        dark_rate = np.random.uniform(DARK_RATE_MIN, DARK_RATE_MAX)
        background = np.random.uniform(BACKGROUND_MIN, BACKGROUND_MAX)

        g2_obs, S_obs, bg_norm, detected_signal = simulate_observed_statistics(loss_db, eta, dark_rate, background)
        k_cb = holevo_from_cb(g2_obs, loss_db, bg_norm)
        k_eb = holevo_from_eb(S_obs, loss_db, bg_norm)
        k_hybrid = hybrid_holevo(k_cb, k_eb, g2_obs, S_obs)

        base_min = min(k_cb, k_eb)
        if base_min > 1e-12:
            reduction_frac = (base_min - k_hybrid) / base_min
        else:
            reduction_frac = 0.0

        results.append({
            'loss_db': loss_db,
            'eta': eta,
            'dark_rate': dark_rate,
            'background': background,
            'g2_obs': g2_obs,
            'S_obs': S_obs,
            'bg_norm': bg_norm,
            'k_cb': k_cb,
            'k_eb': k_eb,
            'k_hybrid': k_hybrid,
            'base_min': base_min,
            'reduction_frac': reduction_frac
        })

except Exception as e:
    # If an error occurs, retry with a smaller trial count and more robust settings
    err_msg = str(e)
    print(json.dumps({'error': 'simulation_failed', 'message': err_msg}))
    # Retry with fewer trials
    N_TRIALS2 = max(200, N_TRIALS // 10)
    results = []
    for i in range(N_TRIALS2):
        loss_db = np.random.uniform(LOSS_MIN_DB, LOSS_MAX_DB)
        eta = np.random.uniform(ETA_MIN, ETA_MAX)
        dark_rate = np.random.uniform(DARK_RATE_MIN, DARK_RATE_MAX)
        background = np.random.uniform(BACKGROUND_MIN, BACKGROUND_MAX)

        g2_obs, S_obs, bg_norm, detected_signal = simulate_observed_statistics(loss_db, eta, dark_rate, background)
        k_cb = holevo_from_cb(g2_obs, loss_db, bg_norm)
        k_eb = holevo_from_eb(S_obs, loss_db, bg_norm)
        k_hybrid = hybrid_holevo(k_cb, k_eb, g2_obs, S_obs)

        base_min = min(k_cb, k_eb)
        if base_min > 1e-12:
            reduction_frac = (base_min - k_hybrid) / base_min
        else:
            reduction_frac = 0.0

        results.append({
            'loss_db': loss_db,
            'eta': eta,
            'dark_rate': dark_rate,
            'background': background,
            'g2_obs': g2_obs,
            'S_obs': S_obs,
            'bg_norm': bg_norm,
            'k_cb': k_cb,
            'k_eb': k_eb,
            'k_hybrid': k_hybrid,
            'base_min': base_min,
            'reduction_frac': reduction_frac
        })

# Convert to arrays for statistics
import math
k_cb_arr = np.array([r['k_cb'] for r in results])
k_eb_arr = np.array([r['k_eb'] for r in results])
k_hybrid_arr = np.array([r['k_hybrid'] for r in results])
base_min_arr = np.array([r['base_min'] for r in results])
reduction_arr = np.array([r['reduction_frac'] for r in results])

# Metrics
avg_k_cb = float(np.mean(k_cb_arr))
avg_k_eb = float(np.mean(k_eb_arr))
avg_k_hybrid = float(np.mean(k_hybrid_arr))
median_reduction = float(np.median(reduction_arr))
mean_reduction = float(np.mean(reduction_arr))
frac_ge_20 = float(np.mean(reduction_arr >= 0.20))
frac_ge_10 = float(np.mean(reduction_arr >= 0.10))

# Paired t-test (min(single) vs hybrid)
paired_diff = base_min_arr - k_hybrid_arr

# Try using scipy if available, otherwise compute t-test manually
p_value = None
try:
    from scipy import stats
    tstat, p_value = stats.ttest_rel(base_min_arr, k_hybrid_arr)
    p_value = float(p_value)
except Exception:
    # manual paired t-test
    n = len(paired_diff)
    mean_diff = float(np.mean(paired_diff))
    std_diff = float(np.std(paired_diff, ddof=1))
    if std_diff <= 0 or n < 2:
        p_value = 1.0
    else:
        t_stat = mean_diff / (std_diff / math.sqrt(n))
        # two-sided p from t
        # approximate using survival function for large n via normal approx
        from math import erf
        z = abs(t_stat)
        p_value = float(2.0 * (1.0 - 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))))

# Additional diagnostics
losses = np.array([r['loss_db'] for r in results])
low_loss_idx = losses <= (LOSS_MIN_DB + 0.5 * LOSS_SPAN)
high_loss_idx = losses > (LOSS_MIN_DB + 0.5 * LOSS_SPAN)

metrics = {
    'n_trials': len(results),
    'avg_k_cb': avg_k_cb,
    'avg_k_eb': avg_k_eb,
    'avg_k_hybrid': avg_k_hybrid,
    'mean_reduction_fraction': mean_reduction,
    'median_reduction_fraction': median_reduction,
    'fraction_trials_reduction_ge_20pct': frac_ge_20,
    'fraction_trials_reduction_ge_10pct': frac_ge_10,
    'p_value_paired_ttest_base_vs_hybrid': p_value,
    'loss_range_db': [LOSS_MIN_DB, LOSS_MAX_DB],
    'eta_range': [ETA_MIN, ETA_MAX],
    'gamma_joint': GAMMA_JOINT,
    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime()),
    'runtime_seconds': time.time() - start_time
}

# Print final JSON to stdout as required
print(json.dumps(metrics))
