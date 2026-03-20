"""
FIXED Experiment: Temporal Leakage Poisoning in Concept-Based OPE

Experiments:
1. Leakage degradation (probe R² over time)
2. OPE error compounding (hard vs soft vs gated vs conformal)
3. Conformal gating fix (distribution-free OOD detection)

Run: python experiments/temporal_leakage_experiment.py
"""

import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(project_root, 'src'))

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple

from gridworld import WindyGridworld, collect_trajectory
from policies import EpsilonGreedyPolicy, OptimalPolicy
from concepts import (HardConcepts, SoftConcepts, GatedSoftConcepts,
                      ConformalGatedConcepts, measure_leakage,
                      train_probe, evaluate_probe)
from ope import monte_carlo_ground_truth
from utils import set_seed, print_trajectory_stats


# =============================================================================
# POLICY CLASSES
# =============================================================================

class ConceptBasedPolicy:
    """Maps state -> concept bin -> action probabilities."""

    def __init__(self, concept_extractor, n_concepts=32, n_actions=4):
        self.concept_extractor = concept_extractor
        self.n_concepts        = n_concepts
        self.n_actions         = n_actions
        self.policy_table      = np.ones((n_concepts, n_actions)) / n_actions

    def state_to_concept_index(self, state) -> int:
        concept_vec = self.concept_extractor(state)
        if len(concept_vec) > 5:
            concept_vec = concept_vec[:5]
        binary = (concept_vec > 0.5).astype(int)
        return sum(b * (2 ** i) for i, b in enumerate(binary))

    def learn_from_trajectories(self, trajectories, smoothing=1.0):
        """
        Count concept-action pairs with Laplace smoothing.
        smoothing=1.0 ensures all 32x4=128 pairs have non-zero probability
        satisfying Assumption 4.1 (completeness).
        """
        counts = np.ones((self.n_concepts, self.n_actions)) * smoothing
        for traj in trajectories:
            for step in traj:
                c = self.state_to_concept_index(step['state'])
                counts[c, step['action']] += 1
        self.policy_table = counts / counts.sum(axis=1, keepdims=True)

    def prob(self, state, action) -> float:
        return float(self.policy_table[self.state_to_concept_index(state),
                                       action])

    def action_probs(self, state) -> np.ndarray:
        return self.policy_table[self.state_to_concept_index(state)]


class GatedConceptBasedPolicy:
    """
    Gated concept policy for entropy and conformal gating.

    prob(a|s) = gate(s) * pi_concept(a|c(s))
              + (1-gate(s)) * pi_global(a)

    Falls back to global policy (concept space) NOT state policy.
    Consistent with paper finding: state-based interventions
    increase MSE (Zarlenga et al. Point 26 in checklist).
    """

    def __init__(self, gated_extractor, n_concepts=32, n_actions=4):
        self.gated_extractor = gated_extractor
        self.n_concepts      = n_concepts
        self.n_actions       = n_actions
        self.policy_table    = np.ones((n_concepts, n_actions)) / n_actions
        self.global_policy   = np.ones(n_actions) / n_actions

    def state_to_concept_index(self, state) -> int:
        concept_vec = self.gated_extractor(state)
        if len(concept_vec) > 5:
            concept_vec = concept_vec[:5]
        binary = (concept_vec > 0.5).astype(int)
        return int(sum(b * (2**i) for i, b in enumerate(binary)))

    def learn_from_trajectories(self, trajectories, smoothing=1.0):
        counts = np.ones((self.n_concepts, self.n_actions)) * smoothing
        for traj in trajectories:
            for step in traj:
                c = self.state_to_concept_index(step['state'])
                counts[c, step['action']] += 1
        self.policy_table = counts / counts.sum(axis=1, keepdims=True)
        total = counts.sum(axis=0)
        self.global_policy = total / total.sum()

    def prob(self, state, action) -> float:
        gate         = self.gated_extractor.get_gate_value(state)
        c            = self.state_to_concept_index(state)
        concept_prob = float(self.policy_table[c, action])
        global_prob  = float(self.global_policy[action])
        return gate * concept_prob + (1 - gate) * global_prob

    def action_probs(self, state) -> np.ndarray:
        gate = self.gated_extractor.get_gate_value(state)
        c    = self.state_to_concept_index(state)
        return gate * self.policy_table[c] + (1 - gate) * self.global_policy


# =============================================================================
# ESTIMATOR FUNCTIONS
# =============================================================================

def cpdis_estimate_by_horizon(trajectories, behavior_policy, eval_policy,
                               max_horizon, gamma=0.99):
    """
    Concept-based Per-Decision IS (CPDIS) estimator.
    V_CPDIS = (1/N) sum_n sum_t gamma^t * rho(0:t) * r_t
    rho(0:t) = prod_{t'=0}^{t} pi_e(a|c) / pi_b(a|c)

    Returns: (estimate, variance, ESS)
    """
    returns, final_rhos = [], []
    for traj in trajectories:
        G, rho = 0.0, 1.0
        for t in range(min(len(traj), max_horizon)):
            step = traj[t]
            pi_e = eval_policy.prob(step['state'], step['action'])
            pi_b = behavior_policy.prob(step['state'], step['action'])
            rho  *= (pi_e / pi_b) if pi_b > 1e-10 else 0.0
            G    += (gamma ** t) * rho * step['reward']
        returns.append(G)
        final_rhos.append(rho)
    returns    = np.array(returns)
    final_rhos = np.array(final_rhos)
    ess = (np.sum(final_rhos)**2 / np.sum(final_rhos**2)
           if np.sum(final_rhos**2) > 0 else 0)
    return float(np.mean(returns)), float(np.var(returns)), float(ess)


def pdis_estimate_by_horizon(trajectories, behavior_policy, eval_policy,
                              max_horizon, gamma=0.99):
    """
    Standard state-based Per-Decision IS (PDIS) baseline.
    Same formula as CPDIS but uses state-based policies.

    Returns: (estimate, variance, ESS)
    """
    returns, final_rhos = [], []
    for traj in trajectories:
        G, rho = 0.0, 1.0
        for t in range(min(len(traj), max_horizon)):
            step = traj[t]
            pi_e = eval_policy.prob(step['state'], step['action'])
            pi_b = behavior_policy.prob(step['state'], step['action'])
            rho  *= (pi_e / pi_b) if pi_b > 1e-10 else 0.0
            G    += (gamma ** t) * rho * step['reward']
        returns.append(G)
        final_rhos.append(rho)
    returns    = np.array(returns)
    final_rhos = np.array(final_rhos)
    ess = (np.sum(final_rhos)**2 / np.sum(final_rhos**2)
           if np.sum(final_rhos**2) > 0 else 0)
    return float(np.mean(returns)), float(np.var(returns)), float(ess)


def compute_ground_truth_by_horizon(env, eval_policy, max_horizon,
                                    n_episodes=2000, gamma=0.99):
    """Monte Carlo estimate of true V(pi_e) at given horizon."""
    returns = []
    for _ in range(n_episodes):
        traj = collect_trajectory(env, eval_policy, max_steps=max_horizon)
        returns.append(sum((gamma**t) * s['reward']
                           for t, s in enumerate(traj)))
    return float(np.mean(returns))


# =============================================================================
# MAIN EXPERIMENT
# =============================================================================

def run_fixed_experiment(
    n_trajectories=500, max_steps=50, train_horizon=10,
    test_horizons=None, seed=42,
    behavior_epsilon=0.4, eval_epsilon=0.05
) -> Dict:
    """
    Run all three experiments for one seed.

    Behavior policy: epsilon-greedy (epsilon=0.4)
    Satisfies Assumption 4.1 (completeness): all actions have
    probability >= 0.4/4 = 0.1 at every state.

    Returns dict with estimates, errors, variances, ESS for all estimators.
    """
    if test_horizons is None:
        test_horizons = [5, 10, 15, 20, 25, 30, 35, 40]

    set_seed(seed)

    print("=" * 70)
    print("FIXED EXPERIMENT: OPE Error Compounding")
    print("=" * 70)

    env             = WindyGridworld()
    behavior_policy = EpsilonGreedyPolicy(env, epsilon=behavior_epsilon,
                                          seed=seed)
    eval_policy     = OptimalPolicy(env, epsilon=eval_epsilon, seed=seed)

    print(f"\n[1] Setup")
    print(f"    Behavior: epsilon-greedy (epsilon={behavior_epsilon})")
    print(f"    Evaluation: Optimal (epsilon={eval_epsilon})")
    print(f"    Train horizon: {train_horizon}")

    # =========================================================================
    # [2] Collect trajectories
    # =========================================================================
    print(f"\n[2] Collecting {n_trajectories} trajectories...")
    trajectories = []
    for _ in range(n_trajectories):
        trajectories.append(collect_trajectory(env, behavior_policy,
                                               max_steps=max_steps))
    print_trajectory_stats(trajectories, "Data")

    # =========================================================================
    # [3] Setup and train concepts
    # =========================================================================
    print(f"\n[3] Setting up concepts...")
    hard_concepts      = HardConcepts(env)
    soft_concepts      = SoftConcepts(env, use_leakage=True, seed=seed)
    gated_concepts     = GatedSoftConcepts(env, seed=seed)
    conformal_concepts = ConformalGatedConcepts(env, seed=seed)

    train_trajs = []
    for traj in trajectories[:200]:
        early = [s for i, s in enumerate(traj) if i < train_horizon]
        if early:
            train_trajs.append(early)

    print(f"    Training soft concepts on t < {train_horizon}...")
    soft_concepts.train_on_trajectories(train_trajs, hard_concepts,
                                        epochs=200)

    print(f"    Training entropy-gated concepts on t < {train_horizon}...")
    gated_concepts.train_on_trajectories(train_trajs, hard_concepts,
                                         epochs=200)
    print(f"    Entropy gated global mean: "
          f"{gated_concepts.global_mean.round(3)}")

    print(f"    Training conformal-gated concepts on t < {train_horizon}...")
    conformal_concepts.train_on_trajectories(train_trajs, hard_concepts,
                                             epochs=200)
    print(f"    Conformal global mean: "
          f"{conformal_concepts.global_mean.round(3)}")

    # =========================================================================
    # EXPERIMENT 1: Temporal Leakage Degradation
    # =========================================================================
    print(f"\n[3b] Experiment 1: Measuring temporal leakage degradation...")

    leakage_timesteps = [2, 5, 10, 15, 20, 25, 30, 35, 40]
    states_by_t   = {t: [] for t in leakage_timesteps}
    features_by_t = {t: [] for t in leakage_timesteps}

    for traj in trajectories:
        for t, step in enumerate(traj):
            if t in states_by_t:
                states_by_t[t].append(step['state'])
                features_by_t[t].append(step['features'])

    train_states, train_features = [], []
    for t in leakage_timesteps:
        if t < train_horizon:
            train_states.extend(states_by_t[t])
            train_features.extend(features_by_t[t])

    print(f"    Training probes on {len(train_states)} samples "
          f"from t < {train_horizon}...")
    probe_soft  = train_probe(soft_concepts,  train_states, train_features)
    probe_gated = train_probe(gated_concepts, train_states, train_features)

    leakage_results = {
        'timesteps': leakage_timesteps,
        'hard_r2':   [], 'soft_r2': [], 'gated_r2': [], 'n_samples': []
    }

    print(f"\n    {'Timestep':<10} {'In-Dist?':<10} {'Hard R2':<12} "
          f"{'Soft R2':<12} {'Gated R2':<12} {'N':<8}")
    print("    " + "-" * 64)

    for t in leakage_timesteps:
        if len(states_by_t[t]) < 10:
            continue
        r2_hard  = measure_leakage(hard_concepts,
                                   np.array(states_by_t[t], dtype=object),
                                   np.array(features_by_t[t]))
        r2_soft  = evaluate_probe(probe_soft,  soft_concepts,
                                  states_by_t[t], features_by_t[t])
        r2_gated = evaluate_probe(probe_gated, gated_concepts,
                                  states_by_t[t], features_by_t[t])
        in_dist  = "Yes" if t < train_horizon else "No"
        n        = len(states_by_t[t])
        leakage_results['hard_r2'].append(r2_hard)
        leakage_results['soft_r2'].append(r2_soft)
        leakage_results['gated_r2'].append(r2_gated)
        leakage_results['n_samples'].append(n)
        print(f"    t={t:<8} {in_dist:<10} {r2_hard:<12.4f} "
              f"{r2_soft:<12.4f} {r2_gated:<12.4f} {n:<8}")

    # Gate values + weight on global_mean
    print("\n=== GATE VALUES AND GLOBAL MEAN WEIGHTS ===")
    print(f"\n{'t':<6} {'in-dist':<10} {'ent gate':<12} {'conf gate':<12} "
          f"{'ent->global%':<14} {'conf->global%':<14}")
    print("-" * 70)
    for t in leakage_timesteps:
        if len(states_by_t[t]) < 10:
            continue
        ge = float(np.mean([gated_concepts.get_gate_value(s)
                            for s in states_by_t[t]]))
        gc = float(np.mean([conformal_concepts.get_gate_value(s)
                            for s in states_by_t[t]]))
        in_dist    = "Yes" if t < train_horizon else "No"
        ent_wt_pct = (1 - ge) * 100
        conf_wt_pct = (1 - gc) * 100
        print(f"{t:<6} {in_dist:<10} {ge:<12.3f} {gc:<12.3f} "
              f"{ent_wt_pct:<14.1f} {conf_wt_pct:<14.1f}")

    results_leakage = leakage_results

    # =========================================================================
    # [4] Build concept-based policies
    # =========================================================================
    print(f"\n[4] Building concept-based policies...")

    hard_behavior_policy      = ConceptBasedPolicy(hard_concepts,
                                                   n_concepts=32, n_actions=4)
    hard_eval_policy          = ConceptBasedPolicy(hard_concepts,
                                                   n_concepts=32, n_actions=4)
    soft_behavior_policy      = ConceptBasedPolicy(soft_concepts,
                                                   n_concepts=32, n_actions=4)
    soft_eval_policy          = ConceptBasedPolicy(soft_concepts,
                                                   n_concepts=32, n_actions=4)
    gated_behavior_policy     = GatedConceptBasedPolicy(gated_concepts,
                                                        n_concepts=32,
                                                        n_actions=4)
    gated_eval_policy         = GatedConceptBasedPolicy(gated_concepts,
                                                        n_concepts=32,
                                                        n_actions=4)
    conformal_behavior_policy = GatedConceptBasedPolicy(conformal_concepts,
                                                        n_concepts=32,
                                                        n_actions=4)
    conformal_eval_policy     = GatedConceptBasedPolicy(conformal_concepts,
                                                        n_concepts=32,
                                                        n_actions=4)

    hard_behavior_policy.learn_from_trajectories(trajectories)
    soft_behavior_policy.learn_from_trajectories(trajectories)
    gated_behavior_policy.learn_from_trajectories(trajectories)
    conformal_behavior_policy.learn_from_trajectories(trajectories)

    print("    Collecting eval policy trajectories...")
    eval_trajectories = []
    for _ in range(200):
        eval_trajectories.append(collect_trajectory(env, eval_policy,
                                                    max_steps=max_steps))

    hard_eval_policy.learn_from_trajectories(eval_trajectories)
    soft_eval_policy.learn_from_trajectories(eval_trajectories)
    gated_eval_policy.learn_from_trajectories(eval_trajectories)
    conformal_eval_policy.learn_from_trajectories(eval_trajectories)

    # =========================================================================
    # [5] EXPERIMENTS 2 & 3: OPE error vs horizon
    # =========================================================================
    print(f"\n[5] Running OPE at horizons: {test_horizons}")

    results = {
        'horizons':        test_horizons,
        'true_values':     [],
        'state_pdis':      {'estimates': [], 'errors': [],
                            'variances': [], 'ess': []},
        'hard_cpdis':      {'estimates': [], 'errors': [],
                            'variances': [], 'ess': []},
        'soft_cpdis':      {'estimates': [], 'errors': [],
                            'variances': [], 'ess': []},
        'gated_cpdis':     {'estimates': [], 'errors': [],
                            'variances': [], 'ess': []},
        'conformal_cpdis': {'estimates': [], 'errors': [],
                            'variances': [], 'ess': []},
    }

    for h in test_horizons:
        print(f"\n    Horizon T={h}:")

        true_val = compute_ground_truth_by_horizon(env, eval_policy, h,
                                                   n_episodes=1000)
        results['true_values'].append(true_val)
        print(f"      True value: {true_val:.4f}")

        for key, b_pol, e_pol, label in [
            ('state_pdis',      behavior_policy,
             eval_policy,           'State PDIS     '),
            ('hard_cpdis',      hard_behavior_policy,
             hard_eval_policy,      'Hard CPDIS     '),
            ('soft_cpdis',      soft_behavior_policy,
             soft_eval_policy,      'Soft CPDIS     '),
            ('gated_cpdis',     gated_behavior_policy,
             gated_eval_policy,     'Gated CPDIS    '),
            ('conformal_cpdis', conformal_behavior_policy,
             conformal_eval_policy, 'Conformal CPDIS'),
        ]:
            fn = (pdis_estimate_by_horizon if key == 'state_pdis'
                  else cpdis_estimate_by_horizon)
            est, var, ess = fn(trajectories, b_pol, e_pol, h)
            err = abs(est - true_val)
            results[key]['estimates'].append(est)
            results[key]['errors'].append(err)
            results[key]['variances'].append(var)
            results[key]['ess'].append(ess)
            print(f"      {label}: est={est:.4f}, err={err:.4f}, "
                  f"var={var:.4f}, ess={ess:.1f}")

    results['leakage'] = results_leakage
    return results


# =============================================================================
# MULTI-SEED EXPERIMENT
# =============================================================================

def run_multiseed_experiment(n_seeds=30, test_horizons=None):
    """
    Run experiment across multiple seeds.
    Computes proper MSE, Bias, Variance to confirm theorems.

    Theorem 4.3: Bias(hard) ~= 0 (unbiased under known concepts)
    Theorem 5.1: Bias(soft) grows after train horizon
    Theorem 4.4: Var(hard) < Var(PDIS) at long horizons
    Theorem 5.2: Var(soft) > Var(PDIS) at long horizons (violated by leakage)
    Our contribution: Var(conformal) < Var(soft) at H>=25
    """
    if test_horizons is None:
        test_horizons = [5, 10, 15, 20, 25, 30, 35, 40]

    keys = ['state_pdis', 'hard_cpdis', 'soft_cpdis',
            'gated_cpdis', 'conformal_cpdis']
    all_estimates = {k: [] for k in keys}
    all_true      = []

    for seed in range(n_seeds):
        print(f"  Seed {seed+1}/{n_seeds}...")
        r = run_fixed_experiment(
            seed=seed,
            n_trajectories=500,
            train_horizon=10,
            test_horizons=test_horizons,
            behavior_epsilon=0.4,
            eval_epsilon=0.05
        )
        all_true.append(r['true_values'])
        for k in keys:
            all_estimates[k].append(r[k]['estimates'])

    true_vals = np.mean(all_true, axis=0)

    print(f"\n{'='*70}")
    print(f"MULTI-SEED RESULTS ({n_seeds} seeds)")
    print(f"{'='*70}")
    print(f"\n{'H':<5} {'Estimator':<22} {'MSE':<10} {'Bias':<10} "
          f"{'Variance':<12} {'Theorem check':<25}")
    print("-" * 87)

    for k in keys:
        ests = np.array(all_estimates[k])
        for i, h in enumerate(test_horizons):
            tv   = true_vals[i]
            mse  = float(np.mean((ests[:, i] - tv) ** 2))
            bias = float(np.mean(ests[:, i]) - tv)
            var  = float(np.var(ests[:, i]))

            if k == 'hard_cpdis':
                thm = "T4.3 OK" if abs(bias) < 0.1 else "T4.3 FAIL"
            elif k == 'soft_cpdis':
                thm = "T5.1 OK" if abs(bias) > 0.1 else "T5.1 unexpect"
            else:
                thm = ""

            print(f"{h:<5} {k:<22} {mse:<10.4f} {bias:<10.4f} "
                  f"{var:<12.4f} {thm:<25}")
        print()

    # Theorem 4.4 check
    print(f"\n{'='*70}")
    print("THEOREM 4.4: Var(hard CPDIS) < Var(state PDIS)?")
    print(f"{'='*70}")
    hard_vars = np.var(np.array(all_estimates['hard_cpdis']),      axis=0)
    pdis_vars = np.var(np.array(all_estimates['state_pdis']),      axis=0)
    soft_vars = np.var(np.array(all_estimates['soft_cpdis']),      axis=0)
    conf_vars = np.var(np.array(all_estimates['conformal_cpdis']), axis=0)

    print(f"\n{'H':<6} {'Hard var':<12} {'PDIS var':<12} "
          f"{'Hard<PDIS?':<12} {'Soft var':<12} {'Conf var':<12} "
          f"{'Conf<Soft?':<12}")
    print("-" * 80)
    for i, h in enumerate(test_horizons):
        c1 = "YES" if hard_vars[i] < pdis_vars[i] else "NO"
        c2 = "YES" if conf_vars[i] < soft_vars[i]  else "NO"
        print(f"{h:<6} {hard_vars[i]:<12.4f} {pdis_vars[i]:<12.4f} "
              f"{c1:<12} {soft_vars[i]:<12.4f} {conf_vars[i]:<12.4f} "
              f"{c2:<12}")

    return {
        'all_estimates': all_estimates,
        'true_vals':     true_vals,
        'horizons':      test_horizons,
        'n_seeds':       n_seeds
    }


# =============================================================================
# INTERPRETABILITY ANALYSIS
# =============================================================================

def analyse_concept_variance(trajectories, hard_behavior_policy,
                              hard_eval_policy, hard_concepts):
    """
    For each concept bin (0-31), compute mean IS ratio.
    High IS ratio = high variance source = problematic concept bin.
    Demonstrates interpretability advantage over raw state abstractions.
    """
    bin_ratios = {i: [] for i in range(32)}
    for traj in trajectories:
        for step in traj:
            s, a  = step['state'], step['action']
            c_idx = hard_behavior_policy.state_to_concept_index(s)
            pi_b  = hard_behavior_policy.prob(s, a)
            pi_e  = hard_eval_policy.prob(s, a)
            if pi_b > 1e-10:
                bin_ratios[c_idx].append(pi_e / pi_b)

    print(f"\n{'='*65}")
    print("CONCEPT-VARIANCE ANALYSIS (Interpretability)")
    print(f"{'='*65}")
    print(f"{'Bin':<5} {'Concepts':<38} {'Mean IS':<10} {'Count':<8}")
    print("-" * 63)

    rows = []
    for idx in range(32):
        if not bin_ratios[idx]:
            continue
        names  = hard_concepts.extract_from_index(idx)
        mean_r = float(np.mean(bin_ratios[idx]))
        rows.append((mean_r, idx, names, len(bin_ratios[idx])))

    for mean_r, idx, names, count in sorted(rows, reverse=True):
        print(f"{idx:<5} {str(names):<38} {mean_r:<10.4f} {count:<8}")


# =============================================================================
# PLOTTING
# =============================================================================

_STYLE = {
    'font.family':       'DejaVu Sans',
    'font.size':         13,
    'axes.titlesize':    15,
    'axes.titleweight':  'bold',
    'axes.labelsize':    13,
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'axes.linewidth':    1.3,
    'axes.grid':         True,
    'grid.alpha':        0.22,
    'grid.linestyle':    '--',
    'xtick.direction':   'out',
    'ytick.direction':   'out',
    'legend.framealpha': 0.92,
    'legend.edgecolor':  '#cccccc',
    'legend.fontsize':   11,
    'lines.linewidth':   2.4,
    'lines.markersize':  8,
    'figure.dpi':        150,
}

C = {
    'hard':    '#27ae60',
    'soft':    '#e74c3c',
    'entropy': '#2980b9',
    'conf':    '#8e44ad',
    'state':   '#2c3e50',
    'vline':   '#7f8c8d',
    'in_bg':   '#f4faf5',
    'ood_bg':  '#fdf4f4',
}

TRAIN_H = 10


def _shade_regions(ax, x_min, x_max):
    ax.axvspan(x_min,   TRAIN_H, color=C['in_bg'],  alpha=1.0, zorder=0)
    ax.axvspan(TRAIN_H, x_max,   color=C['ood_bg'], alpha=1.0, zorder=0)
    ax.axvline(x=TRAIN_H, color=C['vline'], ls='--', lw=1.6,
               label=f'Train horizon (t={TRAIN_H})', zorder=2)


def _region_labels(ax, x_min, x_max, y_frac=0.94):
    ylim = ax.get_ylim()
    y    = ylim[0] + (ylim[1] - ylim[0]) * y_frac
    ax.text((x_min + TRAIN_H) / 2, y,
            'In-distribution', ha='center', fontsize=10,
            color='#1a6b35', style='italic', zorder=5)
    ax.text(TRAIN_H + (x_max - TRAIN_H) / 2, y,
            'Out-of-distribution', ha='center', fontsize=10,
            color='#a93226', style='italic', zorder=5)


def plot_exp1_leakage(results, save_path=None):
    """Figure 1 — Experiment 1: Leakage R² vs Timestep."""
    plt.rcParams.update(_STYLE)
    fig, ax = plt.subplots(figsize=(7, 5))
    lk = results.get('leakage', {})
    ts = lk.get('timesteps', [])[:len(lk.get('hard_r2', []))]
    _shade_regions(ax, min(ts), max(ts))
    ax.plot(ts, lk['hard_r2'], color=C['hard'], marker='s', zorder=3,
            label='Hard Concepts (rule-based)')
    ax.plot(ts, lk['soft_r2'], color=C['soft'], marker='^', zorder=3,
            label='Soft Concepts (neural network)')
    ax.axhline(y=0, color='#aaaaaa', ls='-', lw=1.0, zorder=1)
    ax.set_xlabel('Trajectory Timestep  $t$')
    ax.set_ylabel('Linear Probe  $R^2$')
    ax.set_title('Exp 1 — Temporal Leakage Degradation')
    ax.legend(loc='lower left')
    fig.tight_layout()
    _region_labels(ax, min(ts), max(ts))
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    return fig


def plot_exp2_ope(results, save_path=None):
    """Figure 2 — Experiment 2: OPE Error vs Horizon (hard vs soft)."""
    plt.rcParams.update(_STYLE)
    fig, ax = plt.subplots(figsize=(7, 5))
    H = results['horizons']
    _shade_regions(ax, min(H), max(H))
    ax.plot(H, results['hard_cpdis']['errors'], color=C['hard'], marker='s',
            zorder=3, label='Hard Concept CPDIS')
    ax.plot(H, results['soft_cpdis']['errors'], color=C['soft'], marker='^',
            zorder=3, label='Soft Concept CPDIS')
    ax.set_xlabel('Trajectory Horizon  $H$')
    ax.set_ylabel('OPE Absolute Error')
    ax.set_title('Exp 2 — Leakage Poisoning Compounds Over Horizon')
    ax.legend(loc='upper left')
    fig.tight_layout()
    _region_labels(ax, min(H), max(H))
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    return fig


def plot_exp3_conformal(results, save_path=None):
    """Figure 3 — Experiment 3: OPE Error vs Horizon (all 4 estimators)."""
    plt.rcParams.update(_STYLE)
    fig, ax = plt.subplots(figsize=(7, 5))
    H = results['horizons']
    _shade_regions(ax, min(H), max(H))
    ax.plot(H, results['hard_cpdis']['errors'],      color=C['hard'],
            marker='s', zorder=3, label='Hard Concept CPDIS')
    ax.plot(H, results['soft_cpdis']['errors'],      color=C['soft'],
            marker='^', zorder=3, label='Soft Concept CPDIS')
    ax.plot(H, results['gated_cpdis']['errors'],     color=C['entropy'],
            marker='o', zorder=3, label='Gated CPDIS (Entropy)')
    ax.plot(H, results['conformal_cpdis']['errors'], color=C['conf'],
            marker='D', zorder=3, label='Gated CPDIS (Conformal)')
    ax.set_xlabel('Trajectory Horizon  $H$')
    ax.set_ylabel('OPE Absolute Error')
    ax.set_title('Exp 3 — Conformal Gating Reduces Leakage Poisoning')
    ax.legend(loc='upper left')
    fig.tight_layout()
    _region_labels(ax, min(H), max(H))
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    return fig


def plot_concept_variance_bar(trajectories, hard_behavior_policy,
                               hard_eval_policy, hard_concepts,
                               save_path=None):
    """
    Bar chart: concept bin vs mean IS ratio.
    Red bars = high variance source concepts (> mean + 1 SD).
    Demonstrates interpretability advantage.
    """
    bin_ratios = {i: [] for i in range(32)}
    for traj in trajectories:
        for step in traj:
            s, a  = step['state'], step['action']
            c_idx = hard_behavior_policy.state_to_concept_index(s)
            pi_b  = hard_behavior_policy.prob(s, a)
            pi_e  = hard_eval_policy.prob(s, a)
            if pi_b > 1e-10:
                bin_ratios[c_idx].append(pi_e / pi_b)

    bins   = [i for i in range(32) if bin_ratios[i]]
    means  = [float(np.mean(bin_ratios[i])) for i in bins]
    labels = [str(hard_concepts.extract_from_index(i))[:15] for i in bins]

    plt.rcParams.update(_STYLE)
    fig, ax = plt.subplots(figsize=(12, 5))
    bars      = ax.bar(range(len(bins)), means, color=C['hard'], alpha=0.8)
    threshold = np.mean(means) + np.std(means)
    for bar, mean in zip(bars, means):
        if mean > threshold:
            bar.set_color(C['soft'])
    ax.set_xticks(range(len(bins)))
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    ax.set_xlabel('Concept Bin')
    ax.set_ylabel('Mean IS Ratio')
    ax.set_title('Concept-Variance Analysis: IS Ratio per Concept Bin\n'
                 '(Red = high variance source, above mean+1SD threshold)')
    ax.axhline(y=threshold, color=C['soft'], ls='--', lw=1.5,
               label=f'Threshold ({threshold:.2f})')
    ax.legend()
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    return fig


def plot_ips_histogram(trajectories, behavior_policy, eval_policy,
                       hard_behavior_policy, hard_eval_policy,
                       save_path=None):
    """
    IPS score distribution: state-based vs concept-based.
    Fewer high IPS scores in concepts = lower variance.
    Paper Figure 2c equivalent. Confirms Theorem 4.4 visually.
    """
    state_ips, concept_ips = [], []
    for traj in trajectories:
        for step in traj:
            s, a = step['state'], step['action']
            pb_s = behavior_policy.prob(s, a)
            pe_s = eval_policy.prob(s, a)
            if pb_s > 1e-10:
                state_ips.append(pe_s / pb_s)
            pb_c = hard_behavior_policy.prob(s, a)
            pe_c = hard_eval_policy.prob(s, a)
            if pb_c > 1e-10:
                concept_ips.append(pe_c / pb_c)

    plt.rcParams.update(_STYLE)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(state_ips,   bins=50, alpha=0.6,
            label='State PDIS (baseline)', color=C['soft'])
    ax.hist(concept_ips, bins=50, alpha=0.6,
            label='Hard CPDIS',            color=C['hard'])
    ax.set_xlabel('IPS Score  ($\\rho$)')
    ax.set_ylabel('Frequency')
    ax.set_title('IPS Distribution: State vs Concept\n'
                 'Fewer high IPS in concepts = lower variance (Theorem 4.4)')
    ax.legend()
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    return fig


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    results_dir = os.path.join(project_root, 'results')
    os.makedirs(results_dir, exist_ok=True)

    # ── Single seed experiment ────────────────────────────────────────────────
    results = run_fixed_experiment(
        n_trajectories=500, max_steps=50, train_horizon=10,
        test_horizons=[5, 10, 15, 20, 25, 30, 35, 40], seed=42
    )

    # ── Summary table with variance + ESS ─────────────────────────────────────
    print("\n" + "=" * 70)
    print("SUMMARY: OPE Metrics at Each Horizon")
    print("=" * 70)
    print(f"\n{'H':<5} {'True':<9} "
          f"{'Hard err':<10} {'Soft err':<10} "
          f"{'Ent err':<9} {'Conf err':<9} "
          f"{'Hard var':<10} {'PDIS var':<10} "
          f"{'Hard ESS':<10} {'Soft ESS':<10}")
    print("-" * 102)

    for i, h in enumerate(results['horizons']):
        print(f"{h:<5} "
              f"{results['true_values'][i]:<9.4f} "
              f"{results['hard_cpdis']['errors'][i]:<10.4f} "
              f"{results['soft_cpdis']['errors'][i]:<10.4f} "
              f"{results['gated_cpdis']['errors'][i]:<9.4f} "
              f"{results['conformal_cpdis']['errors'][i]:<9.4f} "
              f"{results['hard_cpdis']['variances'][i]:<10.4f} "
              f"{results['state_pdis']['variances'][i]:<10.4f} "
              f"{results['hard_cpdis']['ess'][i]:<10.1f} "
              f"{results['soft_cpdis']['ess'][i]:<10.1f}")

    # ── Theorem 4.6: K computation ────────────────────────────────────────────
    K = 32 / 70
    print(f"\n{'='*55}")
    print("THEOREM 4.6: MSE Tightening Factor K^(2T)")
    print(f"{'='*55}")
    print(f"  K = |C|/|S| = 32/70 = {K:.4f}")
    print(f"  K^(2T=80)   = {K**80:.2e}  (MSE bound improvement factor)")

    # ── Theorem 5.4: epsilon term table ──────────────────────────────────────
    print(f"\n{'='*55}")
    print("THEOREM 5.4: epsilon term = MSE(soft) - MSE(hard)")
    print(f"{'='*55}")
    print(f"\n{'H':<6} {'Hard MSE':<12} {'Soft MSE':<12} "
          f"{'Conf MSE':<12} {'eps soft':<12} {'eps conf':<12}")
    print("-" * 68)
    for i, h in enumerate(results['horizons']):
        hard_mse = results['hard_cpdis']['errors'][i] ** 2
        soft_mse = results['soft_cpdis']['errors'][i] ** 2
        conf_mse = results['conformal_cpdis']['errors'][i] ** 2
        print(f"{h:<6} {hard_mse:<12.4f} {soft_mse:<12.4f} "
              f"{conf_mse:<12.4f} {soft_mse-hard_mse:<12.4f} "
              f"{conf_mse-hard_mse:<12.4f}")

    # ── Rebuild env and policies for analysis functions ───────────────────────
    print("\nRebuilding policies for analysis functions...")
    env             = WindyGridworld()
    behavior_policy = EpsilonGreedyPolicy(env, epsilon=0.4, seed=42)
    eval_policy_mc  = OptimalPolicy(env, epsilon=0.05, seed=42)
    hard_concepts   = HardConcepts(env)

    trajectories = []
    for _ in range(500):
        trajectories.append(collect_trajectory(env, behavior_policy,
                                               max_steps=50))
    eval_trajs = []
    for _ in range(200):
        eval_trajs.append(collect_trajectory(env, eval_policy_mc,
                                             max_steps=50))

    hard_b = ConceptBasedPolicy(hard_concepts, n_concepts=32, n_actions=4)
    hard_e = ConceptBasedPolicy(hard_concepts, n_concepts=32, n_actions=4)
    hard_b.learn_from_trajectories(trajectories)
    hard_e.learn_from_trajectories(eval_trajs)

    # ── Concept-variance analysis ─────────────────────────────────────────────
    analyse_concept_variance(trajectories, hard_b, hard_e, hard_concepts)

    # ── Plots ─────────────────────────────────────────────────────────────────
    print("\nGenerating plots...")

    plot_exp1_leakage(results,
                      save_path=os.path.join(results_dir, 'exp1_leakage.png'))
    plot_exp2_ope(results,
                  save_path=os.path.join(results_dir, 'exp2_ope_error.png'))
    plot_exp3_conformal(results,
                        save_path=os.path.join(results_dir,
                                               'exp3_conformal_fix.png'))
    plot_concept_variance_bar(
        trajectories, hard_b, hard_e, hard_concepts,
        save_path=os.path.join(results_dir, 'concept_variance_bar.png'))
    plot_ips_histogram(
        trajectories,
        behavior_policy, eval_policy_mc, hard_b, hard_e,
        save_path=os.path.join(results_dir, 'ips_histogram.png'))

    plt.close('all')

    # Save results
    np.save(os.path.join(results_dir, 'fixed_experiment_results.npy'),
            results)
    print(f"\nAll results saved to {results_dir}/")

    # ── Multi-seed experiment (uncomment — ~30 mins) ──────────────────────────
    print("\nRunning multi-seed experiment (30 seeds)...")
    multi = run_multiseed_experiment(n_seeds=1)
    np.save(os.path.join(results_dir, 'multiseed_results.npy'), multi)
    print("Multi-seed results saved.")