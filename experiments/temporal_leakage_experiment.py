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
from concepts import HardConcepts, SoftConcepts, GatedSoftConcepts, ConformalGatedConcepts, measure_leakage
from ope import monte_carlo_ground_truth
from utils import set_seed, print_trajectory_stats


class ConceptBasedPolicy:
    """Maps state -> concept -> action probabilities."""

    def __init__(self, concept_extractor, n_concepts=32, n_actions=4):
        self.concept_extractor = concept_extractor
        self.n_concepts = n_concepts
        self.n_actions  = n_actions
        self.policy_table = np.ones((n_concepts, n_actions)) / n_actions

    def state_to_concept_index(self, state) -> int:
        concept_vec = self.concept_extractor(state)
        if len(concept_vec) > 5:
            concept_vec = concept_vec[:5]
        binary = (concept_vec > 0.5).astype(int)
        return sum(b * (2 ** i) for i, b in enumerate(binary))

    def learn_from_trajectories(self, trajectories, smoothing=1.0):
        counts = np.ones((self.n_concepts, self.n_actions)) * smoothing
        for traj in trajectories:
            for step in traj:
                c = self.state_to_concept_index(step['state'])
                counts[c, step['action']] += 1
        self.policy_table = counts / counts.sum(axis=1, keepdims=True)

    def prob(self, state, action) -> float:
        return float(self.policy_table[self.state_to_concept_index(state), action])

    def action_probs(self, state) -> np.ndarray:
        return self.policy_table[self.state_to_concept_index(state)]


class GatedConceptBasedPolicy:
    """
    prob(a|s) = gate(s) * pi_concept(a|c(s))
              + (1-gate(s)) * pi_global(a)
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
        gate          = self.gated_extractor.get_gate_value(state)
        c             = self.state_to_concept_index(state)
        return gate * self.policy_table[c] + (1 - gate) * self.global_policy


def cpdis_estimate_by_horizon(trajectories, behavior_policy, eval_policy,
                               max_horizon, gamma=0.99):
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
    return float(np.mean(returns)), float(np.var(returns)), ess


def pdis_estimate_by_horizon(trajectories, behavior_policy, eval_policy,
                              max_horizon, gamma=0.99):
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
    return float(np.mean(returns)), float(np.var(returns)), ess


def compute_ground_truth_by_horizon(env, eval_policy, max_horizon,
                                    n_episodes=2000, gamma=0.99):
    returns = []
    for _ in range(n_episodes):
        traj = collect_trajectory(env, eval_policy, max_steps=max_horizon)
        returns.append(sum((gamma**t) * s['reward'] for t, s in enumerate(traj)))
    return float(np.mean(returns))


def run_fixed_experiment(
    n_trajectories=500, max_steps=50, train_horizon=10,
    test_horizons=None, seed=42,
    behavior_epsilon=0.4, eval_epsilon=0.05
) -> Dict:

    if test_horizons is None:
        test_horizons = [5, 10, 15, 20, 25, 30, 35, 40]

    set_seed(seed)

    print("=" * 70)
    print("FIXED EXPERIMENT: OPE Error Compounding")
    print("=" * 70)

    env             = WindyGridworld()
    behavior_policy = EpsilonGreedyPolicy(env, epsilon=behavior_epsilon, seed=seed)
    eval_policy     = OptimalPolicy(env, epsilon=eval_epsilon, seed=seed)

    print(f"\n[1] Setup")
    print(f"    Behavior: ε-greedy (ε={behavior_epsilon})")
    print(f"    Evaluation: Optimal (ε={eval_epsilon})")
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
    soft_concepts.train_on_trajectories(train_trajs, hard_concepts, epochs=200)

    print(f"    Training entropy-gated concepts on t < {train_horizon}...")
    gated_concepts.train_on_trajectories(train_trajs, hard_concepts, epochs=200)
    print(f"    Entropy gated global mean: {gated_concepts.global_mean.round(3)}")

    print(f"    Training conformal-gated concepts on t < {train_horizon}...")
    conformal_concepts.train_on_trajectories(train_trajs, hard_concepts, epochs=200)
    print(f"    Conformal global mean: {conformal_concepts.global_mean.round(3)}")

    # =========================================================================
    # EXPERIMENT 1: Temporal Leakage Degradation
    # =========================================================================
    print(f"\n[3b] Experiment 1: Measuring temporal leakage degradation...")

    from concepts import train_probe, evaluate_probe

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

    print(f"    Training probes on {len(train_states)} samples from t < {train_horizon}...")
    probe_soft  = train_probe(soft_concepts,  train_states, train_features)
    probe_gated = train_probe(gated_concepts, train_states, train_features)

    leakage_results = {
        'timesteps': leakage_timesteps,
        'hard_r2': [], 'soft_r2': [], 'gated_r2': [], 'n_samples': []
    }

    print(f"\n    {'Timestep':<10} {'In-Dist?':<10} {'Hard R²':<12} "
          f"{'Soft R²':<12} {'Gated R²':<12} {'N':<8}")
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

    # Gate values — entropy vs conformal
    print("\n=== GATE VALUES OVER TIME ===")
    print(f"\n{'t':<6} {'in-dist':<10} {'entropy gate':<15} {'conformal gate':<15}")
    print("-" * 48)
    for t in leakage_timesteps:
        if len(states_by_t[t]) < 10:
            continue
        ge = np.mean([gated_concepts.get_gate_value(s)     for s in states_by_t[t]])
        gc = np.mean([conformal_concepts.get_gate_value(s) for s in states_by_t[t]])
        in_dist = "Yes" if t < train_horizon else "No"
        print(f"{t:<6} {in_dist:<10} {ge:<15.3f} {gc:<15.3f}")

    results_leakage = leakage_results

    # =========================================================================
    # [4] Build concept-based policies
    # =========================================================================
    print(f"\n[4] Building concept-based policies...")

    hard_behavior_policy      = ConceptBasedPolicy(hard_concepts,      n_concepts=32, n_actions=4)
    hard_eval_policy          = ConceptBasedPolicy(hard_concepts,      n_concepts=32, n_actions=4)
    soft_behavior_policy      = ConceptBasedPolicy(soft_concepts,      n_concepts=32, n_actions=4)
    soft_eval_policy          = ConceptBasedPolicy(soft_concepts,      n_concepts=32, n_actions=4)
    gated_behavior_policy     = GatedConceptBasedPolicy(gated_concepts,     n_concepts=32, n_actions=4)
    gated_eval_policy         = GatedConceptBasedPolicy(gated_concepts,     n_concepts=32, n_actions=4)
    conformal_behavior_policy = GatedConceptBasedPolicy(conformal_concepts, n_concepts=32, n_actions=4)
    conformal_eval_policy     = GatedConceptBasedPolicy(conformal_concepts, n_concepts=32, n_actions=4)

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
        'state_pdis':      {'estimates': [], 'errors': [], 'variances': []},
        'hard_cpdis':      {'estimates': [], 'errors': [], 'variances': []},
        'soft_cpdis':      {'estimates': [], 'errors': [], 'variances': []},
        'gated_cpdis':     {'estimates': [], 'errors': [], 'variances': []},
        'conformal_cpdis': {'estimates': [], 'errors': [], 'variances': []},
    }

    for h in test_horizons:
        print(f"\n    Horizon T={h}:")

        true_val = compute_ground_truth_by_horizon(env, eval_policy, h,
                                                   n_episodes=1000)
        results['true_values'].append(true_val)
        print(f"      True value: {true_val:.4f}")

        for key, b_pol, e_pol, label in [
            ('state_pdis',      behavior_policy,           eval_policy,           'State PDIS     '),
            ('hard_cpdis',      hard_behavior_policy,      hard_eval_policy,      'Hard CPDIS     '),
            ('soft_cpdis',      soft_behavior_policy,      soft_eval_policy,      'Soft CPDIS     '),
            ('gated_cpdis',     gated_behavior_policy,     gated_eval_policy,     'Gated CPDIS    '),
            ('conformal_cpdis', conformal_behavior_policy, conformal_eval_policy, 'Conformal CPDIS'),
        ]:
            fn = (pdis_estimate_by_horizon if key == 'state_pdis'
                  else cpdis_estimate_by_horizon)
            est, var, ess = fn(trajectories, b_pol, e_pol, h)
            err = abs(est - true_val)
            results[key]['estimates'].append(est)
            results[key]['errors'].append(err)
            results[key]['variances'].append(var)
            print(f"      {label}: est={est:.4f}, err={err:.4f}")

    results['leakage'] = results_leakage
    return results


# =============================================================================
# PLOTTING — three separate publication-quality figures
# =============================================================================

# Shared style constants
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

# Colourblind-friendly palette
C = {
    'hard':    '#27ae60',   # green
    'soft':    '#e74c3c',   # red
    'entropy': '#2980b9',   # blue
    'conf':    '#8e44ad',   # purple
    'state':   '#2c3e50',   # charcoal
    'vline':   '#7f8c8d',   # grey
    'in_bg':   '#f4faf5',   # light green tint
    'ood_bg':  '#fdf4f4',   # light red tint
}

TRAIN_H = 10


def _shade_regions(ax, x_min, x_max):
    """Add in-dist / OOD shaded background."""
    ax.axvspan(x_min, TRAIN_H,  color=C['in_bg'],  alpha=1.0, zorder=0)
    ax.axvspan(TRAIN_H, x_max,  color=C['ood_bg'],  alpha=1.0, zorder=0)
    ax.axvline(x=TRAIN_H, color=C['vline'], ls='--', lw=1.6,
               label=f'Train horizon (t={TRAIN_H})', zorder=2)


def _region_labels(ax, x_min, x_max, y_frac=0.94):
    """Add italic region labels inside the shaded areas."""
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
    ts = lk.get('timesteps', [])
    ts = ts[:len(lk.get('hard_r2', []))]

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
    """Figure 2 — Experiment 2: OPE Error vs Horizon (hard vs soft only)."""
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

    ax.plot(H, results['hard_cpdis']['errors'],      color=C['hard'],    marker='s',
            zorder=3, label='Hard Concept CPDIS')
    ax.plot(H, results['soft_cpdis']['errors'],      color=C['soft'],    marker='^',
            zorder=3, label='Soft Concept CPDIS')
    ax.plot(H, results['gated_cpdis']['errors'],     color=C['entropy'], marker='o',
            zorder=3, label='Gated CPDIS (Entropy)')
    ax.plot(H, results['conformal_cpdis']['errors'], color=C['conf'],    marker='D',
            zorder=3, label='Gated CPDIS (Conformal)')

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


if __name__ == "__main__":
    results_dir = os.path.join(project_root, 'results')
    os.makedirs(results_dir, exist_ok=True)

    results = run_fixed_experiment(
        n_trajectories=500, max_steps=50, train_horizon=10,
        test_horizons=[5, 10, 15, 20, 25, 30, 35, 40], seed=42
    )

    print("\n" + "=" * 70)
    print("SUMMARY: OPE Error at Each Horizon")
    print("=" * 70)
    print(f"\n{'H':<6} {'True':<10} {'Hard':<10} {'Soft':<10} "
          f"{'Entropy gate':<15} {'Conformal gate':<15}")
    print("-" * 68)

    for i, h in enumerate(results['horizons']):
        print(f"{h:<6} {results['true_values'][i]:<10.4f} "
              f"{results['hard_cpdis']['errors'][i]:<10.4f} "
              f"{results['soft_cpdis']['errors'][i]:<10.4f} "
              f"{results['gated_cpdis']['errors'][i]:<15.4f} "
              f"{results['conformal_cpdis']['errors'][i]:<15.4f}")

    print("\nGenerating plots...")
    plot_exp1_leakage(results,
                      save_path=os.path.join(results_dir, 'exp1_leakage.png'))
    plot_exp2_ope(results,
                  save_path=os.path.join(results_dir, 'exp2_ope_error.png'))
    plot_exp3_conformal(results,
                        save_path=os.path.join(results_dir, 'exp3_conformal_fix.png'))

    plt.show()

    np.save(os.path.join(results_dir, 'fixed_experiment_results.npy'), results)
    print(f"\nResults saved to {results_dir}/")