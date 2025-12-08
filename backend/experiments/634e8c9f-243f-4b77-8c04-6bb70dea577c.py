import json
import time
import numpy as np
import matplotlib.pyplot as plt

# Simulation parameters
SEED = 42
np.random.seed(SEED)

n_per_group = 200                # number of replicates per membrane type (total 400)
T = 30                           # time steps per replicate
initial_bacteria = 100           # bacteria initially adhered per replicate

# Membrane-specific parameters (simplified / phenomenological)
# Laminated membrane: higher event frequency and amplitude, higher intrinsic rejection
lam_params = {
    'base_rejection': 0.95,     # baseline salt rejection (fraction)
    'event_lambda': 0.8,        # expected number of transient events per time step
    'amp_mean': 2.0,            # mean amplitude of transient events
    'amp_sd': 0.8,
}
# Control membrane: fewer/weaker transients, lower base rejection
ctrl_params = {
    'base_rejection': 0.60,
    'event_lambda': 0.2,
    'amp_mean': 0.6,
    'amp_sd': 0.3,
}

# Detachment model parameters
alpha = 0.8   # conversion factor amplitude -> detachment propensity
# Fouling effect on rejection: rejection decreases linearly with fraction of bacteria remaining
fouling_sensitivity = 0.6  # fraction of possible rejection lost when bacteria fraction goes to 1

# Simulation function for one group

def simulate_group(n_samples, params):
    final_rejections = np.zeros(n_samples)
    final_bacteria_frac = np.zeros(n_samples)

    for i in range(n_samples):
        bacteria = initial_bacteria
        rejection = params['base_rejection']

        # Time evolution
        for t in range(T):
            # number of transient events this timestep
            n_events = np.random.poisson(params['event_lambda'])
            if n_events > 0:
                # draw amplitudes
                amps = np.random.normal(loc=params['amp_mean'], scale=params['amp_sd'], size=n_events)
                amps = np.clip(amps, 0.0, None)
                total_amp = amps.sum()
            else:
                total_amp = 0.0

            # convert amplitude to detachment fraction this time step
            # simple saturating form: 1 - exp(-alpha * total_amp)
            p_detach = 1.0 - np.exp(-alpha * total_amp)
            p_detach = np.clip(p_detach, 0.0, 1.0)

            # deterministically reduce bacteria fraction (for speed and reproducibility)
            # apply to counts
            detached = np.random.binomial(bacteria, p_detach) if bacteria > 0 else 0
            bacteria = max(0, bacteria - detached)

            # fouling reduces rejection: combine base rejection with fouling penalty proportional to fraction of bacteria still attached
            bacteria_frac = bacteria / initial_bacteria
            rejection = params['base_rejection'] * (1.0 - fouling_sensitivity * bacteria_frac)
            # limit to [0,1]
            rejection = np.clip(rejection, 0.0, 1.0)

        final_rejections[i] = rejection
        final_bacteria_frac[i] = bacteria / initial_bacteria

    return final_rejections, final_bacteria_frac


# Run simulations
start_time = time.time()
try:
    lam_rej, lam_bfrac = simulate_group(n_per_group, lam_params)
    ctrl_rej, ctrl_bfrac = simulate_group(n_per_group, ctrl_params)

    # Compute summary metrics
    metrics = {}
    metrics['salt_rejection_laminated_mean'] = float(np.mean(lam_rej))
    metrics['salt_rejection_laminated_std'] = float(np.std(lam_rej, ddof=1))
    metrics['salt_rejection_control_mean'] = float(np.mean(ctrl_rej))
    metrics['salt_rejection_control_std'] = float(np.std(ctrl_rej, ddof=1))

    metrics['bacterial_survival_laminated_mean'] = float(np.mean(lam_bfrac))
    metrics['bacterial_survival_laminated_std'] = float(np.std(lam_bfrac, ddof=1))
    metrics['bacterial_survival_control_mean'] = float(np.mean(ctrl_bfrac))
    metrics['bacterial_survival_control_std'] = float(np.std(ctrl_bfrac, ddof=1))

    # Statistical tests: try scipy t-test, fallback to permutation test
    try:
        from scipy import stats
        tstat1, pval_rej = stats.ttest_ind(lam_rej, ctrl_rej, equal_var=False)
        tstat2, pval_bact = stats.ttest_ind(lam_bfrac, ctrl_bfrac, equal_var=False)
    except Exception:
        # permutation test for difference-in-means
        def perm_test(x, y, n_perm=2000, seed=SEED):
            rng = np.random.RandomState(seed)
            obs = np.mean(x) - np.mean(y)
            pooled = np.concatenate([x, y])
            count = 0
            for _ in range(n_perm):
                rng.shuffle(pooled)
                new_x = pooled[: len(x)]
                new_y = pooled[len(x):]
                if abs(np.mean(new_x) - np.mean(new_y)) >= abs(obs):
                    count += 1
            return max(1, count) / (n_perm + 1)

        pval_rej = perm_test(lam_rej, ctrl_rej, n_perm=2000)
        pval_bact = perm_test(lam_bfrac, ctrl_bfrac, n_perm=2000)

    metrics['p_value_rejection'] = float(pval_rej)
    metrics['p_value_survival'] = float(pval_bact)

    # Save diagnostic plot
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.boxplot([lam_rej, ctrl_rej], labels=['Laminated', 'Control'])
    plt.title('Final Salt Rejection')
    plt.ylabel('Rejection fraction')

    plt.subplot(1, 2, 2)
    plt.boxplot([lam_bfrac, ctrl_bfrac], labels=['Laminated', 'Control'])
    plt.title('Final Bacterial Survival Fraction')

    plt.tight_layout()
    plotname = 'plot.png'
    plt.savefig(plotname, dpi=150)
    plt.close()

    metrics['plot'] = plotname
    metrics['runtime_seconds'] = float(time.time() - start_time)
    metrics['n_per_group'] = int(n_per_group)
    metrics['time_steps'] = int(T)

    # Print final JSON to stdout as required
    print(json.dumps(metrics))

except Exception as e:
    # If any error occurs, print a JSON with an 'error' key
    err = {'error': str(e)}
    print(json.dumps(err))
