"""
Corrected Experiment: Temporal Leakage Poisoning in Concept-Based OPE

The key insight we want to demonstrate:
1. States at later timesteps are more OOD (distribution shift grows over time)
2. Soft concept predictions DEGRADE at later timesteps due to leakage poisoning
3. This causes CPDIS error to compound multiplicatively

The CORRECT way to measure this:
- Train soft concepts on states from early timesteps (t < T_train)
- Measure concept quality at EACH timestep separately
- Show that concept quality degrades as t increases (OOD effect)
- Show that this causes per-timestep IS ratio errors to grow

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
from policies import EpsilonGreedyPolicy, OptimalPolicy, ConceptPolicy
from concepts import HardConcepts, SoftConcepts, SoftConceptEncoder, measure_leakage
from ope import pdis_estimate, cpdis_estimate, monte_carlo_ground_truth
from utils import set_seed, print_trajectory_stats, compute_state_distribution, kl_divergence


def measure_concept_quality_at_timestep(
    concept_extractor,
    trajectories: List[List[Dict]],
    timestep: int,
    hard_concepts: HardConcepts
) -> Dict:
    """
    Measure how well soft concepts match hard concepts at a specific timestep.
    
    Returns:
        - accuracy: fraction of concept predictions matching hard labels
        - leakage_r2: R² of probe from embeddings to raw features
        - n_samples: number of states at this timestep
    """
    states = []
    features = []
    hard_labels = []
    soft_preds = []
    
    for traj in trajectories:
        if len(traj) > timestep:
            state = traj[timestep]['state']
            feat = traj[timestep]['features']
            
            states.append(state)
            features.append(feat)
            hard_labels.append(hard_concepts(state))
            
            # Get soft concept probabilities (first 5 dims)
            soft_emb = concept_extractor(state)
            if len(soft_emb) > 5:
                soft_preds.append(soft_emb[:5])
            else:
                soft_preds.append(soft_emb)
    
    if len(states) < 10:
        return {'accuracy': np.nan, 'leakage_r2': np.nan, 'n_samples': len(states)}
    
    states = np.array(states)
    features = np.array(features)
    hard_labels = np.array(hard_labels)
    soft_preds = np.array(soft_preds)
    
    # Accuracy: threshold soft predictions at 0.5
    soft_binary = (soft_preds > 0.5).astype(float)
    accuracy = np.mean(soft_binary == hard_labels)
    
    # Leakage
    leakage = measure_leakage(concept_extractor, states, features)
    
    return {
        'accuracy': accuracy,
        'leakage_r2': leakage,
        'n_samples': len(states)
    }


def measure_distribution_shift_at_timestep(
    trajectories: List[List[Dict]],
    reference_timestep: int,
    target_timestep: int,
    env: WindyGridworld
) -> float:
    """
    Measure KL divergence between state distributions at two timesteps.
    """
    ref_states = []
    target_states = []
    
    for traj in trajectories:
        if len(traj) > reference_timestep:
            ref_states.append(traj[reference_timestep]['state'])
        if len(traj) > target_timestep:
            target_states.append(traj[target_timestep]['state'])
    
    if len(ref_states) < 10 or len(target_states) < 10:
        return np.nan
    
    ref_dist = compute_state_distribution(ref_states, env.height, env.width)
    target_dist = compute_state_distribution(target_states, env.height, env.width)
    
    return kl_divergence(target_dist, ref_dist)


def compute_per_timestep_is_error(
    trajectories: List[List[Dict]],
    behavior_policy,
    eval_policy,
    timestep: int
) -> Tuple[float, float]:
    """
    Compute the average IS ratio and its variance at a specific timestep.
    
    Returns:
        (mean_ratio, var_ratio)
    """
    ratios = []
    
    for traj in trajectories:
        if len(traj) > timestep:
            state = traj[timestep]['state']
            action = traj[timestep]['action']
            
            pi_e = eval_policy.prob(state, action)
            pi_b = behavior_policy.prob(state, action)
            
            if pi_b > 1e-10:
                ratio = pi_e / pi_b
                ratios.append(ratio)
    
    if len(ratios) < 10:
        return np.nan, np.nan
    
    return np.mean(ratios), np.var(ratios)


def run_temporal_leakage_experiment(
    n_trajectories: int = 500,
    max_steps: int = 50,
    train_horizon: int = 10,  # Train concepts on states from t < train_horizon
    test_timesteps: List[int] = None,
    n_mc_episodes: int = 2000,
    seed: int = 42,
    behavior_epsilon: float = 0.4,
    eval_epsilon: float = 0.05
) -> Dict:
    """
    Run the temporal leakage experiment.
    
    Key idea:
    1. Train soft concepts on EARLY timesteps only (t < train_horizon)
    2. Evaluate concept quality at ALL timesteps
    3. Show that quality degrades at later timesteps (OOD)
    4. Show this causes OPE error to grow
    """
    if test_timesteps is None:
        test_timesteps = [0, 2, 5, 10, 15, 20, 25, 30, 35, 40]
    
    set_seed(seed)
    
    print("=" * 70)
    print("TEMPORAL LEAKAGE EXPERIMENT (CORRECTED)")
    print("=" * 70)
    
    # Setup
    env = WindyGridworld()
    behavior_policy = EpsilonGreedyPolicy(env, epsilon=behavior_epsilon, seed=seed)
    eval_policy = OptimalPolicy(env, epsilon=eval_epsilon, seed=seed)
    
    print(f"\n[1] Setup")
    print(f"    Behavior: ε-greedy (ε={behavior_epsilon})")
    print(f"    Evaluation: Optimal (ε={eval_epsilon})")
    print(f"    Train horizon: {train_horizon} (concepts trained on t < {train_horizon})")
    
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
    
    # Train soft concepts on EARLY timesteps only
    soft_concepts = SoftConcepts(env, use_leakage=True, seed=seed)
    
    # Collect training data from early timesteps
    early_states = []
    early_features = []
    early_labels = []
    
    for traj in trajectories[:200]:
        for t, step in enumerate(traj):
            if t < train_horizon:
                early_states.append(step['state'])
                early_features.append(step['features'])
                early_labels.append(hard_concepts(step['state']))
    
    print(f"    Training on {len(early_states)} states from t < {train_horizon}")
    
    # Create training trajectories (fake format for existing train function)
    train_trajs = [[{'state': s, 'features': f} for s, f in 
                    zip(early_states[i:i+10], early_features[i:i+10])] 
                   for i in range(0, len(early_states)-10, 10)]
    
    soft_concepts.train_on_trajectories(train_trajs, hard_concepts, epochs=200, verbose=False)
    
    # Measure at each timestep
    print(f"\n[4] Measuring at timesteps: {test_timesteps}")
    
    results = {
        'timesteps': test_timesteps,
        'train_horizon': train_horizon,
        'concept_accuracy': [],
        'leakage_r2': [],
        'distribution_shift': [],
        'is_ratio_mean': [],
        'is_ratio_var': [],
        'n_samples': []
    }
    
    for t in test_timesteps:
        # Concept quality
        quality = measure_concept_quality_at_timestep(
            soft_concepts, trajectories, t, hard_concepts
        )
        results['concept_accuracy'].append(quality['accuracy'])
        results['leakage_r2'].append(quality['leakage_r2'])
        results['n_samples'].append(quality['n_samples'])
        
        # Distribution shift (relative to t=0)
        shift = measure_distribution_shift_at_timestep(
            trajectories, reference_timestep=0, target_timestep=t, env=env
        )
        results['distribution_shift'].append(shift)
        
        # IS ratio statistics
        is_mean, is_var = compute_per_timestep_is_error(
            trajectories, behavior_policy, eval_policy, t
        )
        results['is_ratio_mean'].append(is_mean)
        results['is_ratio_var'].append(is_var)
    
    # Print results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    
    print(f"\n{'t':<6} {'Accuracy':<10} {'Leakage':<10} {'Dist Shift':<12} {'IS Var':<12} {'N':<6}")
    print("-" * 60)
    
    for i, t in enumerate(test_timesteps):
        acc = results['concept_accuracy'][i]
        leak = results['leakage_r2'][i]
        shift = results['distribution_shift'][i]
        is_var = results['is_ratio_var'][i]
        n = results['n_samples'][i]
        
        acc_str = f"{acc:.4f}" if not np.isnan(acc) else "N/A"
        leak_str = f"{leak:.4f}" if not np.isnan(leak) else "N/A"
        shift_str = f"{shift:.4f}" if not np.isnan(shift) else "N/A"
        is_var_str = f"{is_var:.4f}" if not np.isnan(is_var) else "N/A"
        
        in_dist = "✓" if t < train_horizon else "OOD"
        print(f"{t:<6} {acc_str:<10} {leak_str:<10} {shift_str:<12} {is_var_str:<12} {n:<6} {in_dist}")
    
    return results


def plot_temporal_leakage_results(results: Dict, save_path: str = None):
    """
    Create visualization of temporal leakage effect.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    timesteps = results['timesteps']
    train_horizon = results['train_horizon']
    
    # Plot 1: Concept Accuracy over Time
    ax1 = axes[0, 0]
    ax1.plot(timesteps, results['concept_accuracy'], 'b-o', linewidth=2, markersize=8)
    ax1.axvline(x=train_horizon, color='r', linestyle='--', label=f'Train horizon (t={train_horizon})')
    ax1.axhline(y=1.0, color='g', linestyle=':', alpha=0.5)
    ax1.set_xlabel('Timestep t', fontsize=12)
    ax1.set_ylabel('Concept Accuracy', fontsize=12)
    ax1.set_title('Concept Prediction Accuracy vs Timestep', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0.5, 1.05])
    
    # Plot 2: Distribution Shift over Time
    ax2 = axes[0, 1]
    ax2.plot(timesteps, results['distribution_shift'], 'r-s', linewidth=2, markersize=8)
    ax2.axvline(x=train_horizon, color='r', linestyle='--', label=f'Train horizon')
    ax2.set_xlabel('Timestep t', fontsize=12)
    ax2.set_ylabel('KL Divergence from t=0', fontsize=12)
    ax2.set_title('Distribution Shift over Time', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Leakage R² over Time
    ax3 = axes[1, 0]
    ax3.plot(timesteps, results['leakage_r2'], 'm-^', linewidth=2, markersize=8)
    ax3.axvline(x=train_horizon, color='r', linestyle='--', label=f'Train horizon')
    ax3.set_xlabel('Timestep t', fontsize=12)
    ax3.set_ylabel('Leakage R²', fontsize=12)
    ax3.set_title('Information Leakage vs Timestep', fontsize=14)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: IS Ratio Variance over Time
    ax4 = axes[1, 1]
    ax4.plot(timesteps, results['is_ratio_var'], 'g-d', linewidth=2, markersize=8)
    ax4.axvline(x=train_horizon, color='r', linestyle='--', label=f'Train horizon')
    ax4.set_xlabel('Timestep t', fontsize=12)
    ax4.set_ylabel('IS Ratio Variance', fontsize=12)
    ax4.set_title('Importance Sampling Variance vs Timestep', fontsize=14)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle('Temporal Leakage Poisoning Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nPlot saved to: {save_path}")
    
    plt.show()


def run_hard_vs_soft_comparison(
    n_trajectories: int = 500,
    max_steps: int = 50,
    train_horizon: int = 10,
    test_timesteps: List[int] = None,
    seed: int = 42
) -> Dict:
    """
    Compare hard concepts vs soft concepts at each timestep.
    
    This is the key experiment: show that soft concepts degrade at OOD timesteps
    while hard concepts remain stable.
    """
    if test_timesteps is None:
        test_timesteps = [0, 5, 10, 15, 20, 25, 30]
    
    set_seed(seed)
    
    env = WindyGridworld()
    behavior = EpsilonGreedyPolicy(env, epsilon=0.4, seed=seed)
    eval_policy = OptimalPolicy(env, epsilon=0.05, seed=seed)
    
    # Collect trajectories
    trajectories = []
    for _ in range(n_trajectories):
        traj = collect_trajectory(env, behavior, max_steps=max_steps)
        trajectories.append(traj)
    
    # Setup concepts
    hard_concepts = HardConcepts(env)
    soft_concepts = SoftConcepts(env, use_leakage=True, seed=seed)
    
    # Train soft concepts on early timesteps
    train_trajs = []
    for traj in trajectories[:200]:
        early_steps = [s for i, s in enumerate(traj) if i < train_horizon]
        if len(early_steps) > 0:
            train_trajs.append(early_steps)
    
    soft_concepts.train_on_trajectories(train_trajs, hard_concepts, epochs=200)
    
    # Measure concept quality for both
    results = {
        'timesteps': test_timesteps,
        'hard_accuracy': [],
        'soft_accuracy': [],
        'hard_leakage': [],
        'soft_leakage': []
    }
    
    for t in test_timesteps:
        # Hard concepts (should be stable)
        hard_quality = measure_concept_quality_at_timestep(
            hard_concepts, trajectories, t, hard_concepts
        )
        results['hard_accuracy'].append(hard_quality['accuracy'])
        results['hard_leakage'].append(hard_quality['leakage_r2'])
        
        # Soft concepts (should degrade at OOD)
        soft_quality = measure_concept_quality_at_timestep(
            soft_concepts, trajectories, t, hard_concepts
        )
        results['soft_accuracy'].append(soft_quality['accuracy'])
        results['soft_leakage'].append(soft_quality['leakage_r2'])
    
    return results


if __name__ == "__main__":
    results_dir = os.path.join(project_root, 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    # Run main experiment
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: Temporal Leakage Analysis")
    print("=" * 70)
    
    results = run_temporal_leakage_experiment(
        n_trajectories=500,
        max_steps=50,
        train_horizon=10,
        test_timesteps=[0, 2, 5, 10, 15, 20, 25, 30, 35, 40],
        seed=42
    )
    
    plot_temporal_leakage_results(
        results, 
        save_path=os.path.join(results_dir, 'temporal_leakage.png')
    )
    
    # Run hard vs soft comparison
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: Hard vs Soft Concepts")
    print("=" * 70)
    
    comparison = run_hard_vs_soft_comparison(
        n_trajectories=500,
        max_steps=50,
        train_horizon=10,
        test_timesteps=[0, 5, 10, 15, 20, 25, 30],
        seed=42
    )
    
    print("\nHard vs Soft Concept Accuracy:")
    print(f"{'t':<6} {'Hard':<10} {'Soft':<10} {'Diff':<10}")
    print("-" * 40)
    for i, t in enumerate(comparison['timesteps']):
        hard = comparison['hard_accuracy'][i]
        soft = comparison['soft_accuracy'][i]
        diff = hard - soft if not (np.isnan(hard) or np.isnan(soft)) else np.nan
        print(f"{t:<6} {hard:.4f}     {soft:.4f}     {diff:+.4f}")
    
    # Save results
    np.save(os.path.join(results_dir, 'temporal_results.npy'), results)
    np.save(os.path.join(results_dir, 'comparison_results.npy'), comparison)
    
    print(f"\nAll results saved to {results_dir}/")
