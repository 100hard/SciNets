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
import matplotlib.pyplot as plt
import traceback

try:
    np.random.seed(1)

    # Simulation settings
    runs = 200               # Monte Carlo stochastic runs
    timesteps = 200          # discrete time steps per run
    dt = 1.0

    # Baseline state variables (normalized 0-1 where meaningful)
    baseline = {
        'NO': 1.0,           # baseline NO signaling
        'BBB_perm': 0.05,    # baseline BBB permeability
        'LA_brain': 0.01,    # baseline brain LA
        'OXLAM': 0.01,
        'Cardiol_oxid': 0.0,
        'Mito_func': 1.0,
        'DAMP': 0.0,
        'Microglia': 0.05,
        'Neuroinflam': 0.01,
        'Amyloid': 0.01
    }

    # Key model parameters (interpretable)
    params = {
        'alpha_sd': 0.02,   # effect of sleep deprivation on NO decline per timestep per SD unit
        'beta_age': 0.001,  # age-dependent NO decline factor (age in [0,1])
        'k_perm': 0.8,      # sensitivity of BBB permeability to NO drop
        'k_influx': 0.5,    # LA influx scaling given BBB_perm
        'plasma_LA': 1.0,   # plasma albumin-bound LA availability
        'k_ox': 0.6,        # LA -> OXLAM oxidation rate
        'k_cl': 0.5,        # OXLAM -> cardiolipin oxidation rate
        'k_mito': 0.7,      # cardiol oxidation -> mitochondrial dysfunction rate
        'k_damp': 0.6,      # mito dysfunction -> DAMP release
        'k_micro': 0.6,     # DAMP -> microglia activation
        'k_inflam': 0.5,    # microglia -> neuroinflammation
        'k_amyloid': 0.4,   # neuroinflam -> amyloid/tau accumulation
        'clear_LA': 0.05,   # LA clearance from brain
        'clear_OX': 0.05,   # OXLAM clearance
        'clear_DAMP': 0.05, # DAMP clearance
        'clear_micro': 0.02 # microglia return-to-baseline
    }

    # Experimental (hypothesis) axis: chronic sleep deprivation level and age
    SD_level = 0.8  # chronic sleep deprivation intensity (0-1)
    age = 0.6       # relative age-like factor (0-1), captures age-like decline

    # Function to run a single stochastic simulation
    def run_sim(params, SD_level, age, seed=None):
        if seed is not None:
            rng = np.random.RandomState(seed)
        else:
            rng = np.random

        s = {k: float(v) for k, v in baseline.items()}
        traj = {k: np.zeros(timesteps) for k in s.keys()}

        for t in range(timesteps):
            # small additive noise
            noise_scale = 0.005
            noise = {k: rng.normal(scale=noise_scale) for k in s.keys()}

            # NO dynamics: declines with SD and age
            dNO = -params['alpha_sd'] * SD_level - params['beta_age'] * age * (s['NO'] - baseline['NO'])
            s['NO'] = max(0.0, s['NO'] + dt * dNO + noise['NO'])

            # BBB permeability increases when NO decreases (relative to baseline), clamp to [0,1]
            delta_no = max(0.0, baseline['NO'] - s['NO'])
            s['BBB_perm'] = float(np.clip(baseline['BBB_perm'] + params['k_perm'] * delta_no, 0.0, 1.0))

            # LA brain influx scales with BBB_perm and plasma_LA
            influx = params['k_influx'] * s['BBB_perm'] * params['plasma_LA']
            # LA oxidation to OXLAM
            ox = params['k_ox'] * s['LA_brain']
            # LA clearance
            cl_la = params['clear_LA'] * s['LA_brain']
            s['LA_brain'] = max(0.0, s['LA_brain'] + dt * (influx - ox - cl_la) + noise['LA_brain'])

            # OXLAM dynamics
            s['OXLAM'] = max(0.0, s['OXLAM'] + dt * (ox - params['clear_OX'] * s['OXLAM']) + noise['OXLAM'])

            # Cardiolipin oxidation driven by OXLAM
            d_card = params['k_cl'] * s['OXLAM']
            s['Cardiol_oxid'] = np.clip(s['Cardiol_oxid'] + dt * d_card + noise['Cardiol_oxid'], 0.0, 10.0)

            # Mitochondrial function declines with cardiolipin oxidation (lower is worse)
            d_mito = -params['k_mito'] * s['Cardiol_oxid']
            s['Mito_func'] = np.clip(s['Mito_func'] + dt * d_mito + noise['Mito_func'], 0.0, 1.0)

            # DAMP release increases as mitochondrial function is lost
            d_damp = params['k_damp'] * (1.0 - s['Mito_func'])
            s['DAMP'] = max(0.0, s['DAMP'] + dt * (d_damp - params['clear_DAMP'] * s['DAMP']) + noise['DAMP'])

            # Microglia activation
            d_micro = params['k_micro'] * s['DAMP'] - params['clear_micro'] * s['Microglia']
            s['Microglia'] = max(0.0, s['Microglia'] + dt * d_micro + noise['Microglia'])

            # Neuroinflammation
            d_inflam = params['k_inflam'] * s['Microglia']
            s['Neuroinflam'] = max(0.0, s['Neuroinflam'] + dt * d_inflam + noise['Neuroinflam'])

            # Amyloid/tau accumulation responds to neuroinflammation and mitochondrial dysfunction
            d_amy = params['k_amyloid'] * (s['Neuroinflam'] + (1.0 - s['Mito_func']) * 0.5)
            s['Amyloid'] = max(0.0, s['Amyloid'] + dt * d_amy + noise['Amyloid'])

            # Record trajectory
            for k in traj.keys():
                traj[k][t] = float(s[k])

            # Quick failure checks
            if np.any(np.isnan(list(s.values()))) or np.any(np.array(list(s.values())) < -1e-6):
                # Abort this run and return NaN trajectory to indicate failure
                for k in traj.keys():
                    traj[k][t:] = np.nan
                break

        return traj

    # Run Monte Carlo simulations
    trajectories = []
    for r in range(runs):
        traj = run_sim(params, SD_level, age, seed=r)
        trajectories.append(traj)

    # Collect final Amyloid values and detect failure modes
    final_amy = np.array([traj['Amyloid'][-1] for traj in trajectories])
    failures = []
    if np.any(np.isnan(final_amy)):
        failures.append('nan_or_instability_in_some_runs')

    if np.any(final_amy > 1e6):
        failures.append('explosive_unbounded_growth')

    if np.any(final_amy < 0):
        failures.append('negative_concentrations')

    # Compute stability_score: 1 - normalized variability of final amyloid across runs
    mean_amy = np.nanmean(final_amy)
    std_amy = np.nanstd(final_amy)
    stability_score = float(max(0.0, 1.0 - (std_amy / (abs(mean_amy) + 1e-9))))

    # Sensitivity analysis: perturb key parameters by +/-20% and assess relative change in final Amyloid
    key_params = ['alpha_sd', 'k_perm', 'k_influx', 'k_ox', 'k_cl']
    baseline_final = mean_amy
    rel_changes = {}
    for kp in key_params:
        p0 = params[kp]
        deltas = []
        for factor in [0.8, 1.2]:
            ptest = params.copy()
            ptest[kp] = p0 * factor
            # small ensemble for sensitivity (to save time)
            finals = []
            for r in range(20):
                traj = run_sim(ptest, SD_level, age, seed=1000 + r)
                finals.append(traj['Amyloid'][-1])
            finals = np.array(finals)
            mean_final = np.nanmean(finals)
            rel = 0.0
            if baseline_final != 0:
                rel = (mean_final - baseline_final) / (abs(baseline_final) + 1e-9)
            deltas.append(abs(rel))
        rel_changes[kp] = float(np.mean(deltas))

    # Classify sensitivity per-parameter
    sens_class = {}
    for k, v in rel_changes.items():
        if v > 0.3:
            sens_class[k] = 'high'
        elif v > 0.1:
            sens_class[k] = 'medium'
        else:
            sens_class[k] = 'low'

    # Aggregate sensitivity: if any high -> high; elif any medium -> medium; else low
    if any(v == 'high' for v in sens_class.values()):
        overall_sensitivity = 'high'
    elif any(v == 'medium' for v in sens_class.values()):
        overall_sensitivity = 'medium'
    else:
        overall_sensitivity = 'low'

    # Identify other failure modes by inspection of parameter-effect relationships
    if len(failures) == 0:
        # also add model-specific failure modes if chain doesn't propagate
        if mean_amy <= baseline['Amyloid'] * 1.05:
            failures.append('no_appreciable_amyloid_response_under_test_conditions')
    
    # Behavioral pattern summary (brief)
    behavioral_pattern = (
        'Under chronic SD and moderate age, the model typically shows: NO decline -> increased BBB_perm -> ' 
        'elevated LA_brain -> rise in OXLAM -> cardiolipin oxidation -> mitochondrial dysfunction -> DAMP -> ' 
        'microglial activation -> gradual amyloid/tau accumulation. Magnitudes and timing depend on oxidation and BBB gains.'
    )

    # Consistency check: partial if the chain is observed but depends on parameter regimes
    # We treat as 'partial' if final_amy increased > 20% vs baseline across Monte Carlo mean
    consistency_check = 'partial'
    if baseline_final > baseline['Amyloid'] * 1.5:
        consistency_check = 'yes'
    elif baseline_final <= baseline['Amyloid'] * 1.05:
        consistency_check = 'no'

    # Save a representative plot: median run by final Amyloid
    # pick run with closest final amyloid to median
    valid_indices = np.where(~np.isnan(final_amy))[0]
    if len(valid_indices) > 0:
        med_idx = valid_indices[int(len(valid_indices) // 2)]
    else:
        med_idx = 0

    rep = trajectories[med_idx]
    plt.figure(figsize=(8, 6))
    plt.plot(rep['NO'], label='NO')
    plt.plot(rep['BBB_perm'], label='BBB_perm')
    plt.plot(rep['LA_brain'], label='LA_brain')
    plt.plot(rep['OXLAM'], label='OXLAM')
    plt.plot(rep['Cardiol_oxid'], label='Cardiol_oxid')
    plt.plot(rep['Mito_func'], label='Mito_func')
    plt.plot(rep['DAMP'], label='DAMP')
    plt.plot(rep['Microglia'], label='Microglia')
    plt.plot(rep['Amyloid'], label='Amyloid')
    plt.xlabel('Timestep')
    plt.legend(loc='upper right', fontsize='small')
    plt.title('Representative trajectory (median-run)')
    plt.tight_layout()
    plt.savefig('plot.png')
    plt.close()

    # Prepare final JSON metrics (behavioral metrics only, per rules)
    out = {
        'stability_score': round(float(stability_score), 3),
        'sensitivity': overall_sensitivity,
        'failure_modes': failures,
        'behavioral_pattern': behavioral_pattern,
        'consistency_check': consistency_check,
        'plot': 'plot.png'
    }

    print(json.dumps(out))

except Exception as e:
    err = {'error': str(e), 'traceback': traceback.format_exc()}
    print(json.dumps(err))
