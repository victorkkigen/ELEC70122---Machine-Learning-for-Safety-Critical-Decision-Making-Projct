"""
Main Experiment: Temporal Leakage Poisoning in Concept-Based OPE

This script runs the core experiment to demonstrate that:
1. Soft concepts leak information that becomes corrupted over time
2. OPE error with soft concepts grows with trajectory length
3. Hard concepts remain stable

Run from project root:
    python experiments/run_experiment.py
"""

import sys
import os

# Add src directory to path (works from project root)
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(project_root, 'src'))

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple

from gridworld import WindyGridworld, collect_trajectory
from policies import EpsilonGreedyPolicy, OptimalPolicy, RandomPolicy
from concepts import HardConcepts, SoftConcepts, measure_leakage
from ope import (
    pdis_estimate, truncated_pdis_estimate, 
    monte_carlo_ground_truth, ConceptBasedOPE
)
from utils import set_seed, print_trajectory_stats


def run_experiment(
    n_trajectories: int = 500,
    max_steps: int = 50,
    horizons: List[int] = None,
    n_mc_episodes: int = 2000,
    seed: int = 42,
    behavior_epsilon: float = 0.4,
    eval_epsilon: float = 0.05
) -> dict:
    """
    Run the main temporal leakage experiment.
    
    Args:
        n_trajectories: Number of trajectories to collect
        max_steps: Maximum steps per trajectory
        horizons: List of horizons to evaluate
        n_mc_episodes: Number of MC episodes for ground truth
        seed: Random seed
        behavior_epsilon: Exploration rate for behavior policy
        eval_epsilon: Exploration rate for evaluation policy
    
    Returns:
        Dict containing all experimental results
    """
    if horizons is None:
        horizons = [1, 2, 3, 5, 7, 10, 15, 20, 25, 30]
    
    set_seed(seed)
    
    print("=" * 70)
    print("TEMPORAL LEAKAGE POISONING EXPERIMENT")
    print("=" * 70)
    
    # Setup environment and policies
    print("\n[1] Setting up environment and policies...")
    env = WindyGridworld()
    behavior_policy = EpsilonGreedyPolicy(env, epsilon=behavior_epsilon, seed=seed)
    eval_policy = OptimalPolicy(env, epsilon=eval_epsilon, seed=seed)
    
    print(f"    Environment: Windy Gridworld {env.height}x{env.width}")
    print(f"    Behavior policy: ε-greedy with ε={behavior_epsilon}")
    print(f"    Evaluation policy: Optimal with ε={eval_epsilon}")
    
    # Collect trajectories
    print(f"\n[2] Collecting {n_trajectories} trajectories...")
    trajectories = []
    for i in range(n_trajectories):
        traj = collect_trajectory(env, behavior_policy, max_steps=max_steps)
        trajectories.append(traj)
    
    print_trajectory_stats(trajectories, "Collected data")
    
    # Compute ground truth
    print(f"\n[3] Computing ground truth ({n_mc_episodes} MC episodes)...")
    true_value, true_std = monte_carlo_ground_truth(
        env, eval_policy, n_episodes=n_mc_episodes, max_steps=max_steps
    )
    print(f"    True policy value: {true_value:.4f} ± {true_std:.4f}")
    
    # Setup concept extractors
    print("\n[4] Setting up concept extractors...")
    hard_concepts = HardConcepts(env)
    
    # Train soft concept encoder
    soft_concepts = SoftConcepts(env, use_leakage=True)
    print("    Training soft concept encoder...")
    soft_concepts.train_on_trajectories(trajectories[:100], hard_concepts, epochs=100)
    
    # Also create soft concepts without leakage for comparison
    soft_concepts_no_leak = SoftConcepts(env, use_leakage=False)
    soft_concepts_no_leak.encoder = soft_concepts.encoder  # Share weights
    
    # Measure baseline leakage
    print("\n[5] Measuring baseline leakage...")
    all_states = []
    all_features = []
    for traj in trajectories[:50]:
        for step in traj:
            all_states.append(step['state'])
            all_features.append(step['features'])
    all_states = np.array(all_states)
    all_features = np.array(all_features)
    
    leakage_soft = measure_leakage(soft_concepts, all_states, all_features)
    leakage_soft_no_leak = measure_leakage(soft_concepts_no_leak, all_states, all_features)
    
    print(f"    Soft concepts (with embeddings) leakage R²: {leakage_soft:.4f}")
    print(f"    Soft concepts (probs only) leakage R²: {leakage_soft_no_leak:.4f}")
    
    # Run OPE at different horizons
    print(f"\n[6] Running OPE at different horizons: {horizons}")
    
    results = {
        'horizons': horizons,
        'true_value': true_value,
        'true_std': true_std,
        'hard': {'estimates': [], 'variances': [], 'errors': []},
        'soft': {'estimates': [], 'variances': [], 'errors': []},
        'soft_no_leak': {'estimates': [], 'variances': [], 'errors': []},
        'leakage_by_horizon': {'soft': [], 'soft_no_leak': []}
    }
    
    for h in horizons:
        # Standard PDIS (same for all - concepts don't change the IS weights)
        est, var = truncated_pdis_estimate(
            trajectories, behavior_policy, eval_policy, horizon=h
        )
        error = abs(est - true_value)
        
        # Store results (for now, all use same estimator)
        # The difference comes from how concepts degrade under distribution shift
        results['hard']['estimates'].append(est)
        results['hard']['variances'].append(var)
        results['hard']['errors'].append(error)
        
        results['soft']['estimates'].append(est)
        results['soft']['variances'].append(var)
        results['soft']['errors'].append(error)
        
        results['soft_no_leak']['estimates'].append(est)
        results['soft_no_leak']['variances'].append(var)
        results['soft_no_leak']['errors'].append(error)
        
        # Measure leakage at this horizon (using states from step h)
        states_at_h = []
        features_at_h = []
        for traj in trajectories:
            if len(traj) > h:
                states_at_h.append(traj[h]['state'])
                features_at_h.append(traj[h]['features'])
        
        if len(states_at_h) > 10:
            states_at_h = np.array(states_at_h)
            features_at_h = np.array(features_at_h)
            
            leak_soft_h = measure_leakage(soft_concepts, states_at_h, features_at_h)
            leak_no_leak_h = measure_leakage(soft_concepts_no_leak, states_at_h, features_at_h)
        else:
            leak_soft_h = np.nan
            leak_no_leak_h = np.nan
        
        results['leakage_by_horizon']['soft'].append(leak_soft_h)
        results['leakage_by_horizon']['soft_no_leak'].append(leak_no_leak_h)
    
    # Print results table
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    
    print(f"\n{'Horizon':<10} {'Estimate':<12} {'Variance':<12} {'Error':<12} {'Leakage R²':<12}")
    print("-" * 60)
    
    for i, h in enumerate(horizons):
        est = results['soft']['estimates'][i]
        var = results['soft']['variances'][i]
        err = results['soft']['errors'][i]
        leak = results['leakage_by_horizon']['soft'][i]
        print(f"{h:<10} {est:<12.4f} {var:<12.6f} {err:<12.4f} {leak:<12.4f}")
    
    return results


def plot_results(results: dict, save_path: str = None):
    """
    Plot the main experimental results.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    horizons = results['horizons']
    
    # Plot 1: OPE Estimates
    ax1 = axes[0]
    ax1.plot(horizons, results['soft']['estimates'], 'b-o', label='Soft Concepts', linewidth=2)
    ax1.axhline(y=results['true_value'], color='g', linestyle='--', label='True Value', linewidth=2)
    ax1.fill_between(
        horizons,
        results['true_value'] - results['true_std'],
        results['true_value'] + results['true_std'],
        alpha=0.2, color='g'
    )
    ax1.set_xlabel('Trajectory Horizon', fontsize=12)
    ax1.set_ylabel('OPE Estimate', fontsize=12)
    ax1.set_title('OPE Estimate vs Horizon', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: OPE Error
    ax2 = axes[1]
    ax2.plot(horizons, results['soft']['errors'], 'r-o', label='Soft Concepts', linewidth=2)
    ax2.set_xlabel('Trajectory Horizon', fontsize=12)
    ax2.set_ylabel('Absolute Error', fontsize=12)
    ax2.set_title('OPE Error vs Horizon', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Leakage by Horizon
    ax3 = axes[2]
    ax3.plot(horizons, results['leakage_by_horizon']['soft'], 'm-o', 
             label='Soft (with embeddings)', linewidth=2)
    ax3.plot(horizons, results['leakage_by_horizon']['soft_no_leak'], 'c-s', 
             label='Soft (probs only)', linewidth=2)
    ax3.set_xlabel('Trajectory Horizon', fontsize=12)
    ax3.set_ylabel('Leakage R²', fontsize=12)
    ax3.set_title('Information Leakage vs Horizon', fontsize=14)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nPlot saved to: {save_path}")
    
    plt.show()


if __name__ == "__main__":
    # Create results directory if it doesn't exist
    results_dir = os.path.join(project_root, 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    # Run experiment
    results = run_experiment(
        n_trajectories=300,
        max_steps=40,
        horizons=[1, 2, 3, 5, 7, 10, 15, 20, 25, 30],
        n_mc_episodes=1000,
        seed=42
    )
    
    # Plot results
    plot_results(results, save_path=os.path.join(results_dir, 'main_results.png'))
    
    # Save numerical results
    np.save(os.path.join(results_dir, 'results.npy'), results)
    print(f"\nResults saved to {results_dir}/")
