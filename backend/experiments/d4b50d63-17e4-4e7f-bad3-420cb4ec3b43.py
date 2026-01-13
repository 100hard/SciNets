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
import time
import traceback

try:
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
except Exception as e:
    print(json.dumps({"error": f"Missing plotting or numeric library: {e}"}))
    raise SystemExit(0)

# Wrap main experiment in a try/except to ensure any runtime error is reported as JSON
try:
    # === Simulation settings (kept small to respect runtime/data limits) ===
    RNG = np.random.RandomState(42)
    num_runs = 200           # number of simulated experiments (kept moderate)
    time_steps = 720         # minutes, e.g. 12 hours (<=1000)
    baseline = {
        'alpha': 0.010,   # adenosine accumulation per minute while awake
        'beta': 0.030,    # adenosine clearance per minute while asleep
        'theta': 1.0,     # threshold for sleep onset
        'theta_wake': 0.5,# threshold for wake (hysteresis)
        'min_sleep': 30,  # min sleep duration (minutes)
    }

    # Noise levels for stability check
    noise_baseline = 0.01
    noise_high = 0.05

    # Parameter sweep ranges for sensitivity analysis
    alpha_range = [0.006, 0.014]
    beta_range = [0.015, 0.045]
    theta_range = [0.7, 1.3]

    start_time = time.time()

    def simulate_run(params, noise_level, rng, time_steps=time_steps):
        alpha = params['alpha']
        beta = params['beta']
        theta = params['theta']
        theta_wake = params['theta_wake']
        min_sleep = params['min_sleep']

        A = np.zeros(time_steps)  # adenosine
        S = np.zeros(time_steps, dtype=int)  # sleep state (0 wake, 1 sleep)

        a = 0.0
        sleep_timer = 0
        for t in range(time_steps):
            prev_sleep = (S[t-1] == 1) if t > 0 else False
            # Noise injected into accumulation/clearance
            eps = rng.normal(scale=noise_level)
            if prev_sleep:
                # asleep: clear adenosine
                delta = -beta + eps
            else:
                # awake: accumulate adenosine
                delta = alpha + eps

            a = max(0.0, a + delta)
            A[t] = a

            # State transitions
            if t == 0:
                S[t] = 0
            else:
                if S[t-1] == 0:
                    # currently awake: can transit to sleep if A >= theta
                    if a >= theta:
                        S[t] = 1
                        sleep_timer = 1
                    else:
                        S[t] = 0
                else:
                    # currently asleep: stay asleep until min_sleep satisfied and A <= theta_wake
                    if sleep_timer < min_sleep:
                        S[t] = 1
                        sleep_timer += 1
                    else:
                        if a <= theta_wake:
                            S[t] = 0
                        else:
                            S[t] = 1
                            sleep_timer += 1
        # detect sleep episodes
        transitions = np.diff(np.concatenate(([0], S)))
        sleep_onsets = np.where(transitions == 1)[0]
        total_sleep = S.sum()
        n_episodes = len(sleep_onsets)
        return {
            'A': A,
            'S': S,
            'sleep_onsets': sleep_onsets,
            'total_sleep': int(total_sleep),
            'n_episodes': int(n_episodes)
        }

    # Helper to run ensembles with parameter perturbations
    results = []
    params_list = []
    for i in range(num_runs):
        # sample parameters within ranges
        p = {
            'alpha': RNG.uniform(*alpha_range),
            'beta': RNG.uniform(*beta_range),
            'theta': RNG.uniform(*theta_range),
            'theta_wake': baseline['theta_wake'],
            'min_sleep': baseline['min_sleep']
        }
        params_list.append(p)

    # Run baseline noise simulations
    ensemble_baseline = []
    for p in params_list:
        r = simulate_run(p, noise_baseline, RNG)
        ensemble_baseline.append({'params': p, 'out': r})

    # Run higher noise simulations for stability check
    ensemble_highnoise = []
    for p in params_list:
        r = simulate_run(p, noise_high, RNG)
        ensemble_highnoise.append({'params': p, 'out': r})

    # Compute metrics across ensembles
    total_sleeps_baseline = np.array([e['out']['total_sleep'] for e in ensemble_baseline])
    total_sleeps_highnoise = np.array([e['out']['total_sleep'] for e in ensemble_highnoise])

    # Stability: how stable is total sleep under noise perturbation?
    # Use fraction of runs where relative change < 20%
    rel_change = np.abs(total_sleeps_highnoise - total_sleeps_baseline) / (np.maximum(total_sleeps_baseline, 1))
    stable_fraction = float(np.mean(rel_change < 0.20))
    stability_score = float(np.clip(stable_fraction, 0.0, 1.0))

    # Sensitivity: how much do outputs change across parameter sweep?
    # Use coefficient of variation (CV) of total sleep across parameter variations
    mean_sleep = total_sleeps_baseline.mean()
    std_sleep = total_sleeps_baseline.std()
    cv = std_sleep / (mean_sleep + 1e-9)
    if cv > 0.5:
        sensitivity = 'high'
    elif cv > 0.20:
        sensitivity = 'medium'
    else:
        sensitivity = 'low'

    # Failure modes detection
    failure_modes = set()
    no_sleep_mask = total_sleeps_baseline <= (0.05 * time_steps)  # <5% of time as sleep
    continuous_sleep_mask = total_sleeps_baseline >= (0.95 * time_steps)  # >95% time asleep
    many_episodes_mask = np.array([e['out']['n_episodes'] for e in ensemble_baseline]) > (time_steps / 10)

    if no_sleep_mask.any():
        failure_modes.add('no_sleep_in_some_params')
    if continuous_sleep_mask.any():
        failure_modes.add('continuous_sleep_in_some_params')
    if many_episodes_mask.any():
        failure_modes.add('fragmented_sleep_many_episodes')

    # Detect noise-dominated behavior: if high noise removes correlation between A and sleep
    count_supporting_onsets = 0
    for e in ensemble_baseline:
        out = e['out']
        A = out['A']
        onsets = out['sleep_onsets']
        if len(onsets) == 0:
            continue
        # sample A at onsets and random non-onset (awake) times; compare means
        a_on = A[onsets]
        awake_times = np.where(out['S'] == 0)[0]
        if len(awake_times) == 0:
            continue
        sample_idx = RNG.choice(awake_times, size=min(len(a_on), len(awake_times)), replace=False)
        a_awake = A[sample_idx]
        if a_on.mean() > a_awake.mean():
            count_supporting_onsets += 1

    if count_supporting_onsets < (0.5 * num_runs):
        failure_modes.add('weak_adenosine_onset_signal')

    failure_modes = list(failure_modes)

    # Behavioral pattern summary (qualitative)
    median_idx = int(np.argsort(total_sleeps_baseline)[len(total_sleeps_baseline)//2])
    rep = ensemble_baseline[median_idx]
    A_rep = rep['out']['A']
    S_rep = rep['out']['S']

    avg_accumulation = np.nan
    avg_clearance = np.nan
    if np.any(S_rep[:-1] == 0):
        avg_accumulation = np.mean(np.diff(A_rep)[S_rep[:-1] == 0])
    if np.any(S_rep[:-1] == 1):
        avg_clearance = np.mean(np.diff(A_rep)[S_rep[:-1] == 1])
    n_episodes = rep['out']['n_episodes']
    total_sleep = rep['out']['total_sleep']

    behavioral_pattern = (
        f"Typical run: adenosine accumulates during wake and is reduced during sleep. "
        f"Representative run had {n_episodes} sleep episodes totaling {total_sleep} min over {time_steps} min. "
        f"Observed mean accumulation (wake) ~{avg_accumulation:.4f}/min and mean clearance (sleep) ~{avg_clearance:.4f}/min."
    )

    # Consistency check: compute whether higher adenosine precedes sleep onsets across runs
    onset_supports = []
    for e in ensemble_baseline:
        p = e['params']
        out = e['out']
        onsets = out['sleep_onsets']
        if len(onsets) == 0:
            continue
        A = out['A']
        prop = np.mean(A[onsets] >= p['theta'])
        onset_supports.append(prop)

    if len(onset_supports) == 0:
        consistency_fraction = 0.0
    else:
        consistency_fraction = float(np.mean(onset_supports))

    if consistency_fraction > 0.75:
        consistency_check = 'yes'
    elif consistency_fraction > 0.40:
        consistency_check = 'partial'
    else:
        consistency_check = 'no'

    # Save a representative plot
    try:
        plt.figure(figsize=(8, 4))
        t = np.arange(time_steps)
        plt.plot(t, A_rep, label='Adenosine (A)')
        plt.fill_between(t, 0, S_rep * A_rep.max(), color='gray', alpha=0.25, label='Sleep')
        plt.title('Representative run: Adenosine and Sleep State')
        plt.xlabel('Time (min)')
        plt.ylabel('Adenosine / Sleep')
        plt.legend(loc='upper left')
        plt.tight_layout()
        plt.savefig('plot.png', dpi=150)
        plt.close()
        plot_file = 'plot.png'
    except Exception as e:
        plot_file = None

    # Compose final behavioral metrics JSON
    metrics = {
        'stability_score': round(float(stability_score), 3),
        'sensitivity': sensitivity,
        'failure_modes': failure_modes,
        'behavioral_pattern': behavioral_pattern,
        'consistency_check': consistency_check
    }
    if plot_file is not None:
        metrics['plot'] = plot_file

    # Include some auxiliary diagnostics for traceability (not validation metrics)
    metrics['_diagnostics'] = {
        'num_runs': num_runs,
        'time_steps': time_steps,
        'mean_total_sleep': float(mean_sleep),
        'std_total_sleep': float(std_sleep),
        'cv_total_sleep': float(cv),
        'consistency_fraction_onset_meets_theta': float(consistency_fraction)
    }

    # Print final JSON to stdout
    print(json.dumps(metrics))

except Exception as e:
    tb = traceback.format_exc()
    print(json.dumps({"error": str(e), "traceback": tb}))
