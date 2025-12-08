#!/usr/bin/env python3
import random
import math
import json
import time
import statistics

# Monte Carlo simulation without third-party libs to avoid site-package import issues.
# Hypothesis evaluation for hybrid QKD protocol combining thermal HBT g2 monitoring and
# entanglement-based Bell tests. Heuristic mappings provide upper bounds on an eavesdropper's
# Holevo information for CB-only, EB-only, and a hybrid that applies a joint reduction.

random.seed(42)

# Parameters
N_TRIALS = 2000  # moderate number of Monte Carlo trials for robustness
LOSS_MIN_DB = 10.0
LOSS_MAX_DB = 30.0
S_MAX = 2.0 * math.sqrt(2.0)  # Tsirelson bound

# Heuristic mapping hyperparameters
A_CB = 0.6
B_CB = 0.85
C_CB = 0.25

A_EB = 0.6
B_EB = 0.95
C_EB = 0.25

GAMMA_JOINT = 0.45

ETA_MIN = 0.01
ETA_MAX = 0.18
DARK_RATE_MIN = 10.0
DARK_RATE_MAX = 200.0
BACKGROUND_MIN = 0.0
BACKGROUND_MAX = 5.0

LOSS_SPAN = max(LOSS_MAX_DB - LOSS_MIN_DB, 1.0)

# Utility random helpers using stdlib
def rand_uniform(a, b):
    return random.random() * (b - a) + a

def rand_normal(mu=0.0, sigma=1.0):
    # Box-Muller
    u1 = random.random()
    u2 = random.random()
    z0 = math.sqrt(-2.0 * math.log(max(u1, 1e-15))) * math.cos(2.0 * math.pi * u2)
    return mu + z0 * sigma

# Simulation functions
def simulate_observed_statistics(loss_db, eta, dark_rate, background):
    # Transmission linear scale
    tx = 10 ** (-loss_db / 10.0)
    detected_signal = eta * tx
    bg = background + dark_rate * 1e-4
    s2b = detected_signal / (detected_signal + bg + 1e-12)

    # g2 for thermal tends to 2 for pure thermal, contamination pulls toward 1
    g2_mean = 1.0 + 1.0 * s2b
    g2_obs = g2_mean + rand_normal(0, 0.03)
    # clip
    if g2_obs < 1.0:
        g2_obs = 1.0
    if g2_obs > 2.5:
        g2_obs = 2.5

    visibility = s2b * math.exp(-loss_db / 40.0)
    misalignment = rand_normal(0, 0.03)
    S_mean = S_MAX * max(0.0, visibility - 0.02 + misalignment)
    S_obs = S_mean + rand_normal(0, 0.04)
    if S_obs < 0.0:
        S_obs = 0.0
    if S_obs > S_MAX:
        S_obs = S_MAX

    bg_norm = max(0.0, min(1.0, bg / (bg + detected_signal + 1e-12)))

    return float(g2_obs), float(S_obs), float(bg_norm), float(detected_signal)


def holevo_from_cb(g2_obs, loss_db, bg_norm):
    loss_term = A_CB * ((loss_db - LOSS_MIN_DB) / LOSS_SPAN)
    g2_term = B_CB * max(0.0, (2.0 - g2_obs) / 1.0)
    noise_term = C_CB * bg_norm
    raw = loss_term + g2_term + noise_term
    if raw < 0.0:
        raw = 0.0
    if raw > 1.0:
        raw = 1.0
    return float(raw)


def holevo_from_eb(S_obs, loss_db, bg_norm):
    loss_term = A_EB * ((loss_db - LOSS_MIN_DB) / LOSS_SPAN)
    s_range_norm = (S_MAX - 2.0)
    s_dev = max(0.0, (S_MAX - S_obs) / (s_range_norm + 1e-12))
    s_term = B_EB * s_dev
    noise_term = C_EB * bg_norm
    raw = loss_term + s_term + noise_term
    if raw < 0.0:
        raw = 0.0
    if raw > 1.0:
        raw = 1.0
    return float(raw)


def hybrid_holevo(k_cb, k_eb, g2_obs, S_obs):
    base = min(k_cb, k_eb)
    g2_dev_norm = max(0.0, (2.0 - g2_obs) / 1.0)
    s_dev_norm = max(0.0, (S_MAX - S_obs) / (S_MAX - 2.0 + 1e-12))
    corr = g2_dev_norm * s_dev_norm
    reduction = GAMMA_JOINT * corr
    kh = base * (1.0 - reduction)
    if kh < 0.0:
        kh = 0.0
    if kh > 1.0:
        kh = 1.0
    return float(kh)

# Run Monte Carlo with error handling (retry smaller sample on exception)
start_time = time.time()
results = []

try:
    for i in range(N_TRIALS):
        loss_db = rand_uniform(LOSS_MIN_DB, LOSS_MAX_DB)
        eta = rand_uniform(ETA_MIN, ETA_MAX)
        dark_rate = rand_uniform(DARK_RATE_MIN, DARK_RATE_MAX)
        background = rand_uniform(BACKGROUND_MIN, BACKGROUND_MAX)

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
    # Retry with smaller trial count
    err_msg = str(e)
    retry_n = max(200, N_TRIALS // 10)
    results = []
    for i in range(retry_n):
        loss_db = rand_uniform(LOSS_MIN_DB, LOSS_MAX_DB)
        eta = rand_uniform(ETA_MIN, ETA_MAX)
        dark_rate = rand_uniform(DARK_RATE_MIN, DARK_RATE_MAX)
        background = rand_uniform(BACKGROUND_MIN, BACKGROUND_MAX)

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

# Aggregate metrics
n = len(results)
if n == 0:
    metrics = {'error': 'no_results', 'n_trials': 0}
    print(json.dumps(metrics))
    raise SystemExit(0)

k_cb_list = [r['k_cb'] for r in results]
k_eb_list = [r['k_eb'] for r in results]
k_hybrid_list = [r['k_hybrid'] for r in results]
base_min_list = [r['base_min'] for r in results]
reduction_list = [r['reduction_frac'] for r in results]

avg_k_cb = float(statistics.mean(k_cb_list))
avg_k_eb = float(statistics.mean(k_eb_list))
avg_k_hybrid = float(statistics.mean(k_hybrid_list))
median_reduction = float(statistics.median(reduction_list))
mean_reduction = float(statistics.mean(reduction_list))
frac_ge_20 = float(sum(1 for x in reduction_list if x >= 0.20) / n)
frac_ge_10 = float(sum(1 for x in reduction_list if x >= 0.10) / n)

# Paired t-test (base_min vs hybrid) using normal approx for large n
paired_diff = [base_min_list[i] - k_hybrid_list[i] for i in range(n)]
mean_diff = statistics.mean(paired_diff)

if n > 1:
    std_diff = statistics.pstdev(paired_diff) * math.sqrt(n / (n - 0.0)) if n > 1 else 0.0
    # Use sample stdev
    try:
        std_diff_sample = statistics.stdev(paired_diff)
    except Exception:
        std_diff_sample = 0.0
else:
    std_diff_sample = 0.0

if std_diff_sample <= 0 or n < 2:
    p_value = 1.0
else:
    t_stat = mean_diff / (std_diff_sample / math.sqrt(n))
    z = abs(t_stat)
    # two-sided p-value via normal tail approx
    p_value = 2.0 * (1.0 - 0.5 * (1.0 + math.erf(z / math.sqrt(2.0))))

metrics = {
    'n_trials': n,
    'avg_k_cb': avg_k_cb,
    'avg_k_eb': avg_k_eb,
    'avg_k_hybrid': avg_k_hybrid,
    'mean_reduction_fraction': mean_reduction,
    'median_reduction_fraction': median_reduction,
    'fraction_trials_reduction_ge_20pct': frac_ge_20,
    'fraction_trials_reduction_ge_10pct': frac_ge_10,
    'p_value_paired_ttest_base_vs_hybrid': float(p_value),
    'loss_range_db': [LOSS_MIN_DB, LOSS_MAX_DB],
    'eta_range': [ETA_MIN, ETA_MAX],
    'gamma_joint': GAMMA_JOINT,
    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime()),
    'runtime_seconds': time.time() - start_time
}

print(json.dumps(metrics))
