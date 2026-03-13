"""
Off-Policy Evaluation Estimators

Implements Per-Decision Importance Sampling (PDIS) with concept-based modifications.

Key insight: IS ratios multiply over time
    ρ_{0:T} = ∏_{t=0}^{T} ρ_t

This means any bias introduced at each step compounds multiplicatively.
"""

import numpy as np
from typing import List, Callable, Optional, Tuple
from dataclasses import dataclass


@dataclass
class OPEResult:
    """Result of off-policy evaluation."""
    estimate: float
    variance: float
    effective_sample_size: float
    per_step_estimates: List[float]
    per_step_variances: List[float]


def compute_importance_weights(
    trajectory: List[dict],
    behavior_policy,
    eval_policy,
    gamma: float = 0.99
) -> np.ndarray:
    """
    Compute per-decision importance weights for a trajectory.
    
    Returns array of cumulative weights: ρ_{0:t} for each t
    """
    T = len(trajectory)
    weights = np.ones(T)
    cumulative_weight = 1.0
    
    for t, step in enumerate(trajectory):
        state = step['state']
        action = step['action']
        
        prob_b = behavior_policy.get_prob(state, action)
        prob_e = eval_policy.get_prob(state, action)
        
        # Avoid division by zero
        prob_b = max(prob_b, 1e-10)
        
        ratio = prob_e / prob_b
        cumulative_weight *= ratio
        weights[t] = cumulative_weight
    
    return weights


def pdis_estimate(
    trajectories: List[List[dict]],
    behavior_policy,
    eval_policy,
    gamma: float = 0.99,
    max_horizon: Optional[int] = None
) -> OPEResult:
    """
    Per-Decision Importance Sampling estimator.
    
    V̂^{PDIS} = (1/n) Σ_i Σ_t γ^t ρ_{0:t}^{(i)} r_t^{(i)}
    
    Args:
        trajectories: List of trajectory data
        behavior_policy: Policy that collected the data
        eval_policy: Policy we want to evaluate
        gamma: Discount factor
        max_horizon: Maximum horizon to consider (None = use full trajectories)
    
    Returns:
        OPEResult with estimate and diagnostics
    """
    n_trajectories = len(trajectories)
    
    # Determine horizon
    if max_horizon is None:
        max_horizon = max(len(traj) for traj in trajectories)
    
    # Compute weighted returns for each trajectory
    weighted_returns = []
    per_step_weighted_rewards = [[] for _ in range(max_horizon)]
    
    for traj in trajectories:
        weights = compute_importance_weights(traj, behavior_policy, eval_policy, gamma)
        
        traj_return = 0.0
        for t, step in enumerate(traj):
            if t >= max_horizon:
                break
            
            weighted_reward = (gamma ** t) * weights[t] * step['reward']
            traj_return += weighted_reward
            per_step_weighted_rewards[t].append(weighted_reward)
        
        weighted_returns.append(traj_return)
    
    weighted_returns = np.array(weighted_returns)
    
    # Compute estimate and variance
    estimate = np.mean(weighted_returns)
    variance = np.var(weighted_returns) / n_trajectories
    
    # Effective sample size (diagnostic for weight degeneracy)
    all_weights = []
    for traj in trajectories:
        w = compute_importance_weights(traj, behavior_policy, eval_policy, gamma)
        all_weights.extend(w[:max_horizon])
    all_weights = np.array(all_weights)
    ess = (np.sum(all_weights) ** 2) / np.sum(all_weights ** 2) if len(all_weights) > 0 else 0
    
    # Per-step estimates (for temporal analysis)
    per_step_estimates = []
    per_step_variances = []
    for t in range(max_horizon):
        if len(per_step_weighted_rewards[t]) > 0:
            step_est = np.mean(per_step_weighted_rewards[t])
            step_var = np.var(per_step_weighted_rewards[t]) / len(per_step_weighted_rewards[t])
        else:
            step_est = 0.0
            step_var = 0.0
        per_step_estimates.append(step_est)
        per_step_variances.append(step_var)
    
    return OPEResult(
        estimate=estimate,
        variance=variance,
        effective_sample_size=ess,
        per_step_estimates=per_step_estimates,
        per_step_variances=per_step_variances
    )


def truncated_pdis_estimate(
    trajectories: List[List[dict]],
    behavior_policy,
    eval_policy,
    horizon: int,
    gamma: float = 0.99
) -> Tuple[float, float]:
    """
    Compute PDIS estimate truncated at a specific horizon.
    
    This is used to analyze how error changes with trajectory length.
    
    Returns:
        (estimate, variance)
    """
    result = pdis_estimate(
        trajectories, behavior_policy, eval_policy, gamma, max_horizon=horizon
    )
    return result.estimate, result.variance


class ConceptBasedOPE:
    """
    Concept-Based Off-Policy Evaluation.
    
    Uses concepts instead of raw states for computing importance weights.
    This can reduce variance but introduces potential for leakage poisoning.
    """
    
    def __init__(
        self,
        concept_extractor,
        behavior_policy,
        eval_policy,
        gamma: float = 0.99
    ):
        self.concept_extractor = concept_extractor
        self.behavior_policy = behavior_policy
        self.eval_policy = eval_policy
        self.gamma = gamma
    
    def estimate(
        self,
        trajectories: List[List[dict]],
        max_horizon: Optional[int] = None
    ) -> OPEResult:
        """
        Compute concept-based OPE estimate.
        
        The key difference: we use concept representations for analysis,
        but importance weights are still computed from state-action probabilities.
        
        The concepts affect the analysis by:
        1. Potentially learning concept-conditional policies
        2. Using concept-based variance reduction techniques
        """
        # For now, use standard PDIS
        # The concept layer affects how we analyze the results, not the estimator itself
        return pdis_estimate(
            trajectories,
            self.behavior_policy,
            self.eval_policy,
            self.gamma,
            max_horizon
        )
    
    def compute_concept_trajectory(
        self,
        trajectory: List[dict]
    ) -> np.ndarray:
        """Extract concepts for each state in trajectory."""
        concepts = []
        for step in trajectory:
            c = self.concept_extractor(step['state'])
            concepts.append(c)
        return np.array(concepts)
    
    def analyze_temporal_leakage(
        self,
        trajectories: List[List[dict]],
        horizons: List[int]
    ) -> dict:
        """
        Analyze how OPE error changes with trajectory length.
        
        This is the core experiment for detecting temporal leakage poisoning.
        
        Returns:
            Dict with estimates and errors at each horizon
        """
        results = {
            'horizons': horizons,
            'estimates': [],
            'variances': [],
            'concept_stats': []
        }
        
        for h in horizons:
            estimate, variance = truncated_pdis_estimate(
                trajectories,
                self.behavior_policy,
                self.eval_policy,
                horizon=h,
                gamma=self.gamma
            )
            results['estimates'].append(estimate)
            results['variances'].append(variance)
            
            # Compute concept statistics at this horizon
            concept_stats = self._compute_concept_stats_at_horizon(trajectories, h)
            results['concept_stats'].append(concept_stats)
        
        return results
    
    def _compute_concept_stats_at_horizon(
        self,
        trajectories: List[List[dict]],
        horizon: int
    ) -> dict:
        """Compute statistics about concepts at a specific horizon."""
        concepts_at_h = []
        
        for traj in trajectories:
            if len(traj) > horizon:
                step = traj[horizon]
                c = self.concept_extractor(step['state'])
                concepts_at_h.append(c)
        
        if len(concepts_at_h) == 0:
            return {'mean': None, 'std': None, 'n_samples': 0}
        
        concepts_at_h = np.array(concepts_at_h)
        
        return {
            'mean': np.mean(concepts_at_h, axis=0),
            'std': np.std(concepts_at_h, axis=0),
            'n_samples': len(concepts_at_h)
        }


def monte_carlo_ground_truth(
    env,
    policy,
    n_episodes: int = 1000,
    max_steps: int = 100,
    gamma: float = 0.99
) -> Tuple[float, float]:
    """
    Compute ground truth policy value using Monte Carlo rollouts.
    
    This is our target for OPE - we want to estimate this value
    using only data collected by a different (behavior) policy.
    
    Returns:
        (mean_return, std_return)
    """
    returns = []
    
    for _ in range(n_episodes):
        state = env.reset()
        episode_return = 0.0
        
        for t in range(max_steps):
            action = policy(state)
            next_state, reward, done, _ = env.step(action)
            episode_return += (gamma ** t) * reward
            
            if done:
                break
            state = next_state
        
        returns.append(episode_return)
    
    returns = np.array(returns)
    return np.mean(returns), np.std(returns)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, '/home/claude/temporal-leakage-ope/src')
    
    from gridworld import WindyGridworld, collect_trajectory
    from policies import EpsilonGreedyPolicy, OptimalPolicy
    from concepts import HardConcepts, SoftConcepts
    
    print("Testing OPE Estimators")
    print("=" * 60)
    
    # Setup
    env = WindyGridworld()
    behavior_policy = EpsilonGreedyPolicy(env, epsilon=0.4, seed=42)
    eval_policy = OptimalPolicy(env, epsilon=0.05, seed=42)
    
    # Collect trajectories using behavior policy
    print("\nCollecting trajectories with behavior policy...")
    np.random.seed(42)
    trajectories = []
    for i in range(200):
        traj = collect_trajectory(env, behavior_policy, max_steps=50)
        trajectories.append(traj)
    
    avg_len = np.mean([len(t) for t in trajectories])
    print(f"Collected {len(trajectories)} trajectories, avg length: {avg_len:.1f}")
    
    # Compute ground truth
    print("\nComputing ground truth (Monte Carlo)...")
    true_value, true_std = monte_carlo_ground_truth(
        env, eval_policy, n_episodes=1000, max_steps=50
    )
    print(f"True value: {true_value:.4f} ± {true_std:.4f}")
    
    # Standard PDIS estimate
    print("\nStandard PDIS estimate...")
    result = pdis_estimate(trajectories, behavior_policy, eval_policy)
    print(f"PDIS estimate: {result.estimate:.4f}")
    print(f"PDIS variance: {result.variance:.4f}")
    print(f"Effective sample size: {result.effective_sample_size:.1f}")
    print(f"Error: {abs(result.estimate - true_value):.4f}")
    
    # Truncated estimates at different horizons
    print("\n" + "=" * 60)
    print("Truncated PDIS at different horizons")
    print("=" * 60)
    
    horizons = [1, 2, 5, 10, 15, 20, 30]
    print(f"\n{'Horizon':<10} {'Estimate':<12} {'Variance':<12} {'Error':<12}")
    print("-" * 50)
    
    for h in horizons:
        est, var = truncated_pdis_estimate(
            trajectories, behavior_policy, eval_policy, horizon=h
        )
        # Note: truncated estimate should be compared to truncated ground truth
        # For simplicity, we compare to full ground truth here
        error = abs(est - true_value)
        print(f"{h:<10} {est:<12.4f} {var:<12.4f} {error:<12.4f}")
    
    # Concept-based OPE
    print("\n" + "=" * 60)
    print("Concept-Based OPE Analysis")
    print("=" * 60)
    
    # Hard concepts
    hard_concepts = HardConcepts(env)
    hard_ope = ConceptBasedOPE(hard_concepts, behavior_policy, eval_policy)
    
    print("\nHard concepts temporal analysis:")
    hard_results = hard_ope.analyze_temporal_leakage(trajectories, horizons)
    
    print(f"\n{'Horizon':<10} {'Estimate':<12} {'N samples':<12}")
    print("-" * 40)
    for i, h in enumerate(horizons):
        est = hard_results['estimates'][i]
        n = hard_results['concept_stats'][i]['n_samples']
        print(f"{h:<10} {est:<12.4f} {n:<12}")
    
    # Soft concepts (untrained - will show baseline behavior)
    soft_concepts = SoftConcepts(env, use_leakage=True)
    soft_ope = ConceptBasedOPE(soft_concepts, behavior_policy, eval_policy)
    
    print("\nSoft concepts temporal analysis:")
    soft_results = soft_ope.analyze_temporal_leakage(trajectories, horizons)
    
    print(f"\n{'Horizon':<10} {'Estimate':<12} {'N samples':<12}")
    print("-" * 40)
    for i, h in enumerate(horizons):
        est = soft_results['estimates'][i]
        n = soft_results['concept_stats'][i]['n_samples']
        print(f"{h:<10} {est:<12.4f} {n:<12}")
