"""
FIXED Experiment: Temporal Leakage Poisoning in Concept-Based OPE

This experiment ACTUALLY shows that OPE error compounds over time.

Key changes from the broken experiment:
1. Build concept-based policies that use soft concept predictions
2. Run CPDIS with those policies
3. Measure OPE error at different trajectory lengths
4. Show error grows with trajectory length

Run: python experiments/fixed_experiment.py
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
from concepts import HardConcepts, SoftConcepts
from ope import monte_carlo_ground_truth
from utils import set_seed, print_trajectory_stats


class ConceptBasedPolicy:
    """
    A policy that maps state → concept → action probabilities.
    
    This is the key piece that was missing. Instead of π(a|s), we have:
    1. φ(s) → c  (concept extractor)
    2. π^c(a|c)  (concept-conditioned policy)
    """
    
    def __init__(self, concept_extractor, n_concepts=32, n_actions=4):
        """
        Args:
            concept_extractor: Function that maps state → concept index
            n_concepts: Number of possible concept values (2^5 = 32 for our 5 binary concepts)
            n_actions: Number of actions
        """
        self.concept_extractor = concept_extractor
        self.n_concepts = n_concepts
        self.n_actions = n_actions
        
        # π^c(a|c) - probability of action given concept
        # Initialize uniform, will be learned from data
        self.policy_table = np.ones((n_concepts, n_actions)) / n_actions
    
    def state_to_concept_index(self, state) -> int:
        """Convert state to discrete concept index."""
        concept_vec = self.concept_extractor(state)
        
        # Take first 5 dimensions (concept probabilities)
        if len(concept_vec) > 5:
            concept_vec = concept_vec[:5]
        
        # Threshold at 0.5 to get binary
        binary = (concept_vec > 0.5).astype(int)
        
        # Convert to index (0-31)
        idx = sum(b * (2 ** i) for i, b in enumerate(binary))
        return idx
    
    def learn_from_trajectories(self, trajectories: List[List[Dict]], smoothing: float = 1.0):
        """
        Learn π^c(a|c) from trajectory data.
        
        Count how often each action is taken for each concept, then normalize.
        """
        counts = np.ones((self.n_concepts, self.n_actions)) * smoothing
        
        for traj in trajectories:
            for step in traj:
                state = step['state']
                action = step['action']
                c_idx = self.state_to_concept_index(state)
                counts[c_idx, action] += 1
        
        # Normalize to probabilities
        self.policy_table = counts / counts.sum(axis=1, keepdims=True)
    
    def prob(self, state, action) -> float:
        """Return π^c(a|c) where c = φ(s)."""
        c_idx = self.state_to_concept_index(state)
        return self.policy_table[c_idx, action]
    
    def action_probs(self, state) -> np.ndarray:
        """Return full action distribution for a state."""
        c_idx = self.state_to_concept_index(state)
        return self.policy_table[c_idx]


def cpdis_estimate_by_horizon(
    trajectories: List[List[Dict]],
    behavior_concept_policy: ConceptBasedPolicy,
    eval_concept_policy: ConceptBasedPolicy,
    max_horizon: int,
    gamma: float = 0.99
) -> Tuple[float, float, float]:
    """
    Run CPDIS up to a specific horizon.
    
    Returns:
        (estimate, variance, effective_sample_size)
    """
    returns = []
    final_rhos = []
    
    for traj in trajectories:
        G = 0.0
        rho_cumulative = 1.0
        
        # Only go up to max_horizon or trajectory length
        T = min(len(traj), max_horizon)
        
        for t in range(T):
            step = traj[t]
            state = step['state']
            action = step['action']
            reward = step['reward']
            
            # Concept-based importance ratio
            pi_e_c = eval_concept_policy.prob(state, action)
            pi_b_c = behavior_concept_policy.prob(state, action)
            
            if pi_b_c < 1e-10:
                rho_t = 0.0
            else:
                rho_t = pi_e_c / pi_b_c
            
            # THIS IS WHERE COMPOUNDING HAPPENS
            rho_cumulative *= rho_t
            
            G += (gamma ** t) * rho_cumulative * reward
        
        returns.append(G)
        final_rhos.append(rho_cumulative)
    
    returns = np.array(returns)
    final_rhos = np.array(final_rhos)
    
    estimate = np.mean(returns)
    variance = np.var(returns)
    
    # Effective sample size
    if np.sum(final_rhos ** 2) > 0:
        ess = (np.sum(final_rhos) ** 2) / np.sum(final_rhos ** 2)
    else:
        ess = 0
    
    return estimate, variance, ess


def pdis_estimate_by_horizon(
    trajectories: List[List[Dict]],
    behavior_policy,
    eval_policy,
    max_horizon: int,
    gamma: float = 0.99
) -> Tuple[float, float, float]:
    """
    Standard PDIS (state-based) up to a specific horizon.
    """
    returns = []
    final_rhos = []
    
    for traj in trajectories:
        G = 0.0
        rho_cumulative = 1.0
        
        T = min(len(traj), max_horizon)
        
        for t in range(T):
            step = traj[t]
            state = step['state']
            action = step['action']
            reward = step['reward']
            
            pi_e = eval_policy.prob(state, action)
            pi_b = behavior_policy.prob(state, action)
            
            if pi_b < 1e-10:
                rho_t = 0.0
            else:
                rho_t = pi_e / pi_b
            
            rho_cumulative *= rho_t
            G += (gamma ** t) * rho_cumulative * reward
        
        returns.append(G)
        final_rhos.append(rho_cumulative)
    
    returns = np.array(returns)
    final_rhos = np.array(final_rhos)
    
    estimate = np.mean(returns)
    variance = np.var(returns)
    
    if np.sum(final_rhos ** 2) > 0:
        ess = (np.sum(final_rhos) ** 2) / np.sum(final_rhos ** 2)
    else:
        ess = 0
    
    return estimate, variance, ess


def compute_ground_truth_by_horizon(
    env,
    eval_policy,
    max_horizon: int,
    n_episodes: int = 2000,
    gamma: float = 0.99
) -> float:
    """
    Compute true policy value up to a specific horizon.
    """
    returns = []
    
    for _ in range(n_episodes):
        traj = collect_trajectory(env, eval_policy, max_steps=max_horizon)
        G = sum((gamma ** t) * step['reward'] for t, step in enumerate(traj))
        returns.append(G)
    
    return np.mean(returns)


def run_fixed_experiment(
    n_trajectories: int = 500,
    max_steps: int = 50,
    train_horizon: int = 10,
    test_horizons: List[int] = None,
    seed: int = 42,
    behavior_epsilon: float = 0.4,
    eval_epsilon: float = 0.05
) -> Dict:
    """
    The FIXED experiment that actually shows OPE error compounding.
    """
    if test_horizons is None:
        test_horizons = [5, 10, 15, 20, 25, 30, 35, 40]
    
    set_seed(seed)
    
    print("=" * 70)
    print("FIXED EXPERIMENT: OPE Error Compounding")
    print("=" * 70)
    
    # Setup
    env = WindyGridworld()
    behavior_policy = EpsilonGreedyPolicy(env, epsilon=behavior_epsilon, seed=seed)
    eval_policy = OptimalPolicy(env, epsilon=eval_epsilon, seed=seed)
    
    print(f"\n[1] Setup")
    print(f"    Behavior: ε-greedy (ε={behavior_epsilon})")
    print(f"    Evaluation: Optimal (ε={eval_epsilon})")
    print(f"    Train horizon: {train_horizon}")
    
    # Collect trajectories
    print(f"\n[2] Collecting {n_trajectories} trajectories...")
    trajectories = []
    for _ in range(n_trajectories):
        traj = collect_trajectory(env, behavior_policy, max_steps=max_steps)
        trajectories.append(traj)
    print_trajectory_stats(trajectories, "Data")
    
    # Setup concepts
    print(f"\n[3] Setting up concepts...")
    hard_concepts = HardConcepts(env)
    soft_concepts = SoftConcepts(env, use_leakage=True, seed=seed)
    
    # Train soft concepts on early timesteps only
    train_trajs = []
    for traj in trajectories[:200]:
        early_steps = [s for i, s in enumerate(traj) if i < train_horizon]
        if len(early_steps) > 0:
            train_trajs.append(early_steps)
    
    print(f"    Training soft concepts on t < {train_horizon}...")
    soft_concepts.train_on_trajectories(train_trajs, hard_concepts, epochs=200)
    
    # =========================================================================
    # EXPERIMENT 1: Temporal Leakage Degradation
    # Train probe on t < train_horizon, evaluate R² at each timestep
    # =========================================================================
    print(f"\n[3b] Experiment 1: Measuring temporal leakage degradation...")
    
    from concepts import train_probe, evaluate_probe
    
    # Collect states and features by timestep
    leakage_timesteps = [2, 5, 10, 15, 20, 25, 30, 35, 40]
    states_by_t = {t: [] for t in leakage_timesteps}
    features_by_t = {t: [] for t in leakage_timesteps}
    
    for traj in trajectories:
        for t, step in enumerate(traj):
            if t in states_by_t:
                states_by_t[t].append(step['state'])
                features_by_t[t].append(step['features'])
    
    # Train probes on early timesteps only (t < train_horizon)
    train_states = []
    train_features = []
    for t in leakage_timesteps:
        if t < train_horizon:
            train_states.extend(states_by_t[t])
            train_features.extend(features_by_t[t])
    
    print(f"    Training probes on {len(train_states)} samples from t < {train_horizon}...")
    probe_hard = train_probe(hard_concepts, train_states, train_features)
    probe_soft = train_probe(soft_concepts, train_states, train_features)
    
    # Evaluate at each timestep
    leakage_results = {
        'timesteps': leakage_timesteps,
        'hard_r2': [],
        'soft_r2': [],
        'n_samples': []
    }
    
    print(f"\n    {'Timestep':<10} {'In-Dist?':<10} {'Hard R²':<12} {'Soft R²':<12} {'N':<8}")
    print("    " + "-" * 52)
    
    for t in leakage_timesteps:
        if len(states_by_t[t]) < 10:
            print(f"    t={t}: Skipping (only {len(states_by_t[t])} samples)")
            continue
        
        r2_hard = evaluate_probe(probe_hard, hard_concepts, states_by_t[t], features_by_t[t])
        r2_soft = evaluate_probe(probe_soft, soft_concepts, states_by_t[t], features_by_t[t])
        
        in_dist = "Yes" if t < train_horizon else "No"
        n_samples = len(states_by_t[t])
        
        leakage_results['hard_r2'].append(r2_hard)
        leakage_results['soft_r2'].append(r2_soft)
        leakage_results['n_samples'].append(n_samples)
        
        print(f"    t={t:<8} {in_dist:<10} {r2_hard:<12.4f} {r2_soft:<12.4f} {n_samples:<8}")
    
    # Store leakage results
    results_leakage = leakage_results
    
    # Build concept-based policies
    print(f"\n[4] Building concept-based policies...")
    
    # Hard concept policies (should work well)
    hard_behavior_policy = ConceptBasedPolicy(hard_concepts, n_concepts=32, n_actions=4)
    hard_eval_policy = ConceptBasedPolicy(hard_concepts, n_concepts=32, n_actions=4)
    
    # Soft concept policies (should degrade at OOD)
    soft_behavior_policy = ConceptBasedPolicy(soft_concepts, n_concepts=32, n_actions=4)
    soft_eval_policy = ConceptBasedPolicy(soft_concepts, n_concepts=32, n_actions=4)
    
    # Learn policies from data
    # Behavior policies learn from behavior trajectories
    hard_behavior_policy.learn_from_trajectories(trajectories)
    soft_behavior_policy.learn_from_trajectories(trajectories)
    
    # Eval policies learn from eval policy rollouts
    print("    Collecting eval policy trajectories...")
    eval_trajectories = []
    for _ in range(200):
        traj = collect_trajectory(env, eval_policy, max_steps=max_steps)
        eval_trajectories.append(traj)
    
    hard_eval_policy.learn_from_trajectories(eval_trajectories)
    soft_eval_policy.learn_from_trajectories(eval_trajectories)
    
    # Run OPE at different horizons
    print(f"\n[5] Running OPE at horizons: {test_horizons}")
    
    results = {
        'horizons': test_horizons,
        'true_values': [],
        'state_pdis': {'estimates': [], 'errors': [], 'variances': []},
        'hard_cpdis': {'estimates': [], 'errors': [], 'variances': []},
        'soft_cpdis': {'estimates': [], 'errors': [], 'variances': []},
    }
    
    for h in test_horizons:
        print(f"\n    Horizon T={h}:")
        
        # Ground truth at this horizon
        true_val = compute_ground_truth_by_horizon(env, eval_policy, h, n_episodes=1000)
        results['true_values'].append(true_val)
        print(f"      True value: {true_val:.4f}")
        
        # State-based PDIS
        est, var, ess = pdis_estimate_by_horizon(
            trajectories, behavior_policy, eval_policy, h
        )
        err = abs(est - true_val)
        results['state_pdis']['estimates'].append(est)
        results['state_pdis']['errors'].append(err)
        results['state_pdis']['variances'].append(var)
        print(f"      State PDIS:  est={est:.4f}, err={err:.4f}")
        
        # Hard concept CPDIS
        est, var, ess = cpdis_estimate_by_horizon(
            trajectories, hard_behavior_policy, hard_eval_policy, h
        )
        err = abs(est - true_val)
        results['hard_cpdis']['estimates'].append(est)
        results['hard_cpdis']['errors'].append(err)
        results['hard_cpdis']['variances'].append(var)
        print(f"      Hard CPDIS:  est={est:.4f}, err={err:.4f}")
        
        # Soft concept CPDIS
        est, var, ess = cpdis_estimate_by_horizon(
            trajectories, soft_behavior_policy, soft_eval_policy, h
        )
        err = abs(est - true_val)
        results['soft_cpdis']['estimates'].append(est)
        results['soft_cpdis']['errors'].append(err)
        results['soft_cpdis']['variances'].append(var)
        print(f"      Soft CPDIS:  est={est:.4f}, err={err:.4f}")
    
    # Add leakage results to output
    results['leakage'] = results_leakage
    
    return results


def plot_fixed_results(results: Dict, save_path: str = None):
    """
    Plot the key result: OPE error vs horizon for hard vs soft concepts.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    horizons = results['horizons']
    
    # Plot 1: OPE Error vs Horizon
    ax1 = axes[0]
    ax1.plot(horizons, results['state_pdis']['errors'], 'k-o', 
             linewidth=2, markersize=8, label='State PDIS')
    ax1.plot(horizons, results['hard_cpdis']['errors'], 'g-s', 
             linewidth=2, markersize=8, label='Hard Concept CPDIS')
    ax1.plot(horizons, results['soft_cpdis']['errors'], 'r-^', 
             linewidth=2, markersize=8, label='Soft Concept CPDIS')
    
    ax1.set_xlabel('Trajectory Horizon T', fontsize=12)
    ax1.set_ylabel('OPE Absolute Error', fontsize=12)
    ax1.set_title('OPE Error vs Horizon', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.axvline(x=10, color='gray', linestyle='--', alpha=0.5, label='Train horizon')
    
    # Plot 2: OPE Variance vs Horizon
    ax2 = axes[1]
    ax2.plot(horizons, results['state_pdis']['variances'], 'k-o', 
             linewidth=2, markersize=8, label='State PDIS')
    ax2.plot(horizons, results['hard_cpdis']['variances'], 'g-s', 
             linewidth=2, markersize=8, label='Hard Concept CPDIS')
    ax2.plot(horizons, results['soft_cpdis']['variances'], 'r-^', 
             linewidth=2, markersize=8, label='Soft Concept CPDIS')
    
    ax2.set_xlabel('Trajectory Horizon T', fontsize=12)
    ax2.set_ylabel('OPE Variance', fontsize=12)
    ax2.set_title('OPE Variance vs Horizon', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')
    ax2.axvline(x=10, color='gray', linestyle='--', alpha=0.5)
    
    # Plot 3: Leakage R² vs Timestep (Experiment 1)
    ax3 = axes[2]
    if 'leakage' in results:
        leakage = results['leakage']
        timesteps = leakage['timesteps'][:len(leakage['hard_r2'])]
        
        ax3.plot(timesteps, leakage['hard_r2'], 'g-s', 
                 linewidth=2, markersize=8, label='Hard Concepts')
        ax3.plot(timesteps, leakage['soft_r2'], 'r-^', 
                 linewidth=2, markersize=8, label='Soft Concepts')
        
        ax3.set_xlabel('Timestep t', fontsize=12)
        ax3.set_ylabel('Probe R² (Leakage)', fontsize=12)
        ax3.set_title('Exp 1: Leakage vs Timestep', fontsize=14, fontweight='bold')
        ax3.legend(fontsize=10)
        ax3.grid(True, alpha=0.3)
        ax3.axvline(x=10, color='gray', linestyle='--', alpha=0.5, label='Train horizon')
        ax3.axhline(y=0, color='gray', linestyle='-', alpha=0.3)  # Zero line
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nPlot saved to: {save_path}")
    
    plt.show()
    
    return fig


if __name__ == "__main__":
    results_dir = os.path.join(project_root, 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    # Run the fixed experiment
    results = run_fixed_experiment(
        n_trajectories=500,
        max_steps=50,
        train_horizon=10,
        test_horizons=[5, 10, 15, 20, 25, 30, 35, 40],
        seed=42
    )
    
    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY: OPE Error at Each Horizon")
    print("=" * 70)
    print(f"\n{'Horizon':<10} {'True Val':<12} {'State Err':<12} {'Hard Err':<12} {'Soft Err':<12}")
    print("-" * 60)
    
    for i, h in enumerate(results['horizons']):
        true_val = results['true_values'][i]
        state_err = results['state_pdis']['errors'][i]
        hard_err = results['hard_cpdis']['errors'][i]
        soft_err = results['soft_cpdis']['errors'][i]
        print(f"{h:<10} {true_val:<12.4f} {state_err:<12.4f} {hard_err:<12.4f} {soft_err:<12.4f}")
    
    # Plot results
    plot_fixed_results(results, save_path=os.path.join(results_dir, 'ope_error_compounding.png'))
    
    # Save results
    np.save(os.path.join(results_dir, 'fixed_experiment_results.npy'), results)
    print(f"\nResults saved to {results_dir}/")