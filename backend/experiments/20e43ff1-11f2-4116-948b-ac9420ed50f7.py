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
import sys

try:
    # Simple ASCII-only exploratory dynamical model
    np.random.seed(0)

    params = {
        'k_sws_recovery': 0.3,
        'k_ne_suppress': 0.8,
        'k_sd': 0.6,
        'k_ise_on': 0.6,
        'k_ise_decay': 0.2,
        'k_aqp4_on': 0.5,
        'k_aqp4_decay': 0.1,
        'k_aqp4_dep_by_ab': 0.2,
        'k_clear_base': 0.15,
        'k_clear_aqp4': 1.0,
        'ab_prod': 0.05,
        'k_ab_seed': 0.02,
        'k_seed_clear': 0.01,
        'k_seed_increase_ne': 0.6,
        'k_ne_decay': 0.3,
        'k_ne_base': 0.3
    }

    T = 50            # timesteps, small
    dt = 0.2
    ensemble = 3      # small ensemble
    scenarios = {
        'control': {'SD_strength': 0.0, 'SD_period': (0, 0)},
        'SD': {'SD_strength': 1.0, 'SD_period': (0, T)}
    }

    rng = np.random.default_rng(1)

    def run_sim(p, SD_strength=0.0, SD_period=(0,0), noise_scale=0.0, rng=None):
        SWS, NE, ISE, AQP4, Abeta, Abeta_seeded = 0.8, 0.5, 0.7, 0.8, 0.5, 0.05
        traj = {'SWS':[], 'NE':[], 'Abeta':[], 'AQP4':[]}
        for t in range(T):
            sd_on = SD_strength if (SD_period[0] <= t <= SD_period[1]) else 0.0
            noise = (rng.normal(scale=noise_scale) if rng is not None else 0.0)
            dNE = (p['k_ne_base'] + p['k_seed_increase_ne']*Abeta_seeded - 0.5*SWS - p['k_ne_decay']*NE)*dt
            NE = max(0.0, NE + dNE + noise*0.01)
            dSWS = (p['k_sws_recovery']*(1.0 - SWS) - p['k_ne_suppress']*NE*SWS - p['k_sd']*sd_on*SWS)*dt
            SWS = np.clip(SWS + dSWS + noise*0.01, 0.0, 1.0)
            dISE = (p['k_ise_on']*SWS - p['k_ise_decay']*ISE)*dt
            ISE = np.clip(ISE + dISE + noise*0.01, 0.0, 1.0)
            dAQP4 = (p['k_aqp4_on']*ISE - p['k_aqp4_decay']*AQP4 - p['k_aqp4_dep_by_ab']*Abeta*AQP4)*dt
            AQP4 = np.clip(AQP4 + dAQP4 + noise*0.01, 0.0, 1.0)
            clearance = max(0.0, p['k_clear_base'] + p['k_clear_aqp4']*AQP4)
            dAbeta = (p['ab_prod'] - clearance*Abeta - p['k_ab_seed']*Abeta)*dt
            Abeta = max(0.0, Abeta + dAbeta + noise*0.01)
            dSeed = (p['k_ab_seed']*Abeta - p['k_seed_clear']*Abeta_seeded)*dt
            Abeta_seeded = max(0.0, Abeta_seeded + dSeed + noise*0.0005)
            traj['SWS'].append(SWS)
            traj['NE'].append(NE)
            traj['Abeta'].append(Abeta)
            traj['AQP4'].append(AQP4)
        return traj

    # Run small ensembles
    results = {}
    for name, sc in scenarios.items():
        runs = []
        for i in range(ensemble):
            # small parameter jitter +/-10 percent simulated via normal around 1
            pert = {k: max(1e-8, v * rng.normal(1.0, 0.10)) for k,v in params.items()}
            traj = run_sim(pert, SD_strength=sc['SD_strength'], SD_period=sc['SD_period'], noise_scale=0.03, rng=rng)
            runs.append(traj)
        results[name] = runs

    # Summaries
    summary = {}
    for name, runs in results.items():
        ab_final = np.array([r['Abeta'][-1] for r in runs])
        sws_final = np.array([r['SWS'][-1] for r in runs])
        ne_final = np.array([r['NE'][-1] for r in runs])
        summary[name] = {
            'ab_mean': float(ab_final.mean()),
            'ab_std': float(ab_final.std()),
            'sws_mean': float(sws_final.mean()),
            'ne_mean': float(ne_final.mean()),
            'ab_vals': ab_final.tolist()
        }

    # stability_score from CV of Abeta in SD scenario
    sd_ab = summary['SD']['ab_mean']
    sd_std = summary['SD']['ab_std']
    cv = (sd_std / sd_ab) if sd_ab>1e-8 else 10.0
    stability_score = float(np.clip(1.0/(1.0+cv), 0.0, 1.0))

    # sensitivity: vary two params +/-25 percent
    key_params = ['k_seed_increase_ne', 'k_aqp4_dep_by_ab']
    baseline = run_sim(params, SD_strength=scenarios['SD']['SD_strength'], SD_period=scenarios['SD']['SD_period'], noise_scale=0.0, rng=np.random.default_rng(2))
    baseline_ab = baseline['Abeta'][-1]
    frac_changes = []
    for p in key_params:
        for factor in [0.75, 1.25]:
            p2 = params.copy()
            p2[p] = p2[p]*factor
            r = run_sim(p2, SD_strength=scenarios['SD']['SD_strength'], SD_period=scenarios['SD']['SD_period'], noise_scale=0.0, rng=np.random.default_rng(3))
            frac_changes.append(abs(r['Abeta'][-1] - baseline_ab)/(baseline_ab+1e-8))
    max_frac = max(frac_changes)
    if max_frac>0.5:
        sensitivity='high'
    elif max_frac>0.2:
        sensitivity='medium'
    else:
        sensitivity='low'

    # failure modes
    failure_modes = []
    for name, runs in results.items():
        for r in runs:
            if r['Abeta'][-1] > 2.0:
                failure_modes.append('runaway_Abeta')
            if np.isnan(r['Abeta'][-1]):
                failure_modes.append('nan')
    if not failure_modes:
        failure_modes = ['none_observed']

    # behavioral pattern
    bp = (
        f"Control Abeta={summary['control']['ab_mean']:.3f}, SWS={summary['control']['sws_mean']:.3f}, NE={summary['control']['ne_mean']:.3f}. "
        f"SD Abeta={summary['SD']['ab_mean']:.3f}, SWS={summary['SD']['sws_mean']:.3f}, NE={summary['SD']['ne_mean']:.3f}."
    )

    # consistency check: directional expectations
    cond1 = summary['SD']['ne_mean'] > summary['control']['ne_mean']
    cond2 = summary['SD']['sws_mean'] < summary['control']['sws_mean']
    cond3 = summary['SD']['ab_mean'] >= summary['control']['ab_mean']
    consistency = 'partial'
    if cond1 and cond2 and cond3:
        consistency = 'yes'

    # small plot of mean Abeta trajectories
    mean_ctrl = np.mean(np.array([r['Abeta'] for r in results['control']]), axis=0)
    mean_sd = np.mean(np.array([r['Abeta'] for r in results['SD']]), axis=0)
    plt.figure(figsize=(6,3))
    plt.plot(np.arange(T)*dt, mean_ctrl, label='control')
    plt.plot(np.arange(T)*dt, mean_sd, label='SD')
    plt.xlabel('time')
    plt.ylabel('mean Abeta')
    plt.legend(fontsize='small')
    plt.tight_layout()
    plt.savefig('plot.png')
    plt.close()

    out = {
        'stability_score': round(stability_score,3),
        'sensitivity': sensitivity,
        'failure_modes': sorted(list(set(failure_modes))),
        'behavioral_pattern': bp,
        'consistency_check': consistency,
        'plot': 'plot.png'
    }
    print(json.dumps(out))

except Exception as e:
    # Return a JSON with the error key as required
    msg = {'error': str(e)}
    print(json.dumps(msg))
    sys.exit(0)
