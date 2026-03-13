"""
Off-Policy Evaluation Estimators

Includes:
- Standard PDIS (Per-Decision Importance Sampling)
- CPDIS (Concept-Based Per-Decision Importance Sampling)
- Monte Carlo ground truth estimation
- Truncated variants for horizon analysis
"""

import numpy as np
from typing import Tuple, List, Dict, Optional, Callable


def pdis_estimate(
    trajectories: List[List[Dict]], 
    behavior_policy, 
    eval_policy,
    gamma: float = 0.99
) -> Tuple[float, float]:
    """
    Per-Decision Importance Sampling estimate.
    
    V̂^PDIS = (1/N) Σ_n Σ_t γ^t ρ_{0:t} r_t
    
    where ρ_{0:t} = Π_{t'=0}^{t} π_e(a_t'|s_t') / π_b(a_t'|s_t')
    
    Args:
        trajectories: List of trajectory lists
        behavior_policy: Policy used to collect data (has .prob(s, a))
        eval_policy: Policy being evaluated (has .prob(s, a))
        gamma: Discount factor
    
    Returns:
        (estimate, variance) tuple
    """
    N = len(trajectories)
    returns = []
    
    for traj in trajectories:
        G = 0.0
        rho_cumulative = 1.0
        
        for t, step in enumerate(traj):
            state = step['state']
            action = step['action']
            reward = step['reward']
            
            # Importance ratio
            pi_e = eval_policy.prob(state, action)
            pi_b = behavior_policy.prob(state, action)
            
            if pi_b < 1e-10:
                rho_t = 0.0
            else:
                rho_t = pi_e / pi_b
            
            rho_cumulative *= rho_t
            
            # Add weighted reward
            G += (gamma ** t) * rho_cumulative * reward
        
        returns.append(G)
    
    returns = np.array(returns)
    estimate = np.mean(returns)
    variance = np.var(returns) / N  # Variance of the mean
    
    return estimate, variance


def truncated_pdis_estimate(
    trajectories: List[List[Dict]], 
    behavior_policy, 
    eval_policy,
    horizon: int,
    gamma: float = 0.99
) -> Tuple[float, float]:
    """
    PDIS truncated at a specific horizon.
    
    Only considers first `horizon` steps of each trajectory.
    """
    N = len(trajectories)
    returns = []
    
    for traj in trajectories:
        G = 0.0
        rho_cumulative = 1.0
        
        for t, step in enumerate(traj[:horizon]):
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
    
    returns = np.array(returns)
    estimate = np.mean(returns)
    variance = np.var(returns) / N
    
    return estimate, variance


def cpdis_estimate(
    trajectories: List[List[Dict]],
    concept_fn: Callable,
    behavior_concept_policy,
    eval_concept_policy,
    gamma: float = 0.99
) -> Tuple[float, float]:
    """
    Concept-Based Per-Decision Importance Sampling (CPDIS).
    
    V̂^CPDIS = (1/N) Σ_n Σ_t γ^t ρ^c_{0:t} r_t
    
    where ρ^c_{0:t} = Π_{t'=0}^{t} π^c_e(a_t'|c_t') / π^c_b(a_t'|c_t')
    
    Key difference from PDIS: policies conditioned on concepts, not states.
    
    Args:
        trajectories: List of trajectory lists
        concept_fn: Function mapping state -> concept (φ(s) -> c)
        behavior_concept_policy: π^c_b with .prob(state, action) (uses concept internally)
        eval_concept_policy: π^c_e with .prob(state, action) (uses concept internally)
        gamma: Discount factor
    
    Returns:
        (estimate, variance) tuple
    """
    N = len(trajectories)
    returns = []
    
    for traj in trajectories:
        G = 0.0
        rho_cumulative = 1.0
        
        for t, step in enumerate(traj):
            state = step['state']
            action = step['action']
            reward = step['reward']
            
            # Get concept
            concept = concept_fn(state)
            
            # Importance ratio over concepts
            pi_e_c = eval_concept_policy.prob(state, action)
            pi_b_c = behavior_concept_policy.prob(state, action)
            
            if pi_b_c < 1e-10:
                rho_t = 0.0
            else:
                rho_t = pi_e_c / pi_b_c
            
            rho_cumulative *= rho_t
            G += (gamma ** t) * rho_cumulative * reward
        
        returns.append(G)
    
    returns = np.array(returns)
    estimate = np.mean(returns)
    variance = np.var(returns) / N
    
    return estimate, variance


def truncated_cpdis_estimate(
    trajectories: List[List[Dict]],
    concept_fn: Callable,
    behavior_concept_policy,
    eval_concept_policy,
    horizon: int,
    gamma: float = 0.99
) -> Tuple[float, float]:
    """
    CPDIS truncated at a specific horizon.
    """
    N = len(trajectories)
    returns = []
    
    for traj in trajectories:
        G = 0.0
        rho_cumulative = 1.0
        
        for t, step in enumerate(traj[:horizon]):
            state = step['state']
            action = step['action']
            reward = step['reward']
            
            concept = concept_fn(state)
            
            pi_e_c = eval_concept_policy.prob(state, action)
            pi_b_c = behavior_concept_policy.prob(state, action)
            
            if pi_b_c < 1e-10:
                rho_t = 0.0
            else:
                rho_t = pi_e_c / pi_b_c
            
            rho_cumulative *= rho_t
            G += (gamma ** t) * rho_cumulative * reward
        
        returns.append(G)
    
    returns = np.array(returns)
    estimate = np.mean(returns)
    variance = np.var(returns) / N
    
    return estimate, variance


def monte_carlo_ground_truth(
    env, 
    policy,
    n_episodes: int = 1000,
    max_steps: int = 100,
    gamma: float = 0.99
) -> Tuple[float, float]:
    """
    Compute ground truth policy value via Monte Carlo rollouts.
    
    Returns:
        (mean_return, std_return) tuple
    """
    # Import here to avoid circular imports
    import sys
    import os
    src_dir = os.path.dirname(os.path.abspath(__file__))
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)
    from gridworld import collect_trajectory
    
    returns = []
    
    for _ in range(n_episodes):
        traj = collect_trajectory(env, policy, max_steps=max_steps)
        
        G = 0.0
        for t, step in enumerate(traj):
            G += (gamma ** t) * step['reward']
        
        returns.append(G)
    
    returns = np.array(returns)
    return np.mean(returns), np.std(returns)


def effective_sample_size(trajectories, behavior_policy, eval_policy) -> float:
    """
    Compute effective sample size (ESS) for importance sampling.
    
    ESS = (Σ ρ)² / Σ ρ²
    
    Lower ESS indicates higher variance.
    """
    all_rhos = []
    
    for traj in trajectories:
        rho_cumulative = 1.0
        
        for step in traj:
            state = step['state']
            action = step['action']
            
            pi_e = eval_policy.prob(state, action)
            pi_b = behavior_policy.prob(state, action)
            
            if pi_b > 1e-10:
                rho_cumulative *= (pi_e / pi_b)
            else:
                rho_cumulative = 0.0
                break
        
        all_rhos.append(rho_cumulative)
    
    all_rhos = np.array(all_rhos)
    
    sum_rho = np.sum(all_rhos)
    sum_rho_sq = np.sum(all_rhos ** 2)
    
    if sum_rho_sq < 1e-10:
        return 0.0
    
    return (sum_rho ** 2) / sum_rho_sq


class ConceptBasedOPE:
    """
    Wrapper class for concept-based OPE experiments.
    
    Handles learning concept policies and running CPDIS.
    """
    
    def __init__(self, concept_extractor, env, n_concepts: int = 5):
        """
        Args:
            concept_extractor: HardConcepts or SoftConcepts instance
            env: WindyGridworld environment
            n_concepts: Number of discrete concept values
        """
        self.concept_extractor = concept_extractor
        self.env = env
        self.n_concepts = n_concepts
        
        self.behavior_concept_policy = None
        self.eval_concept_policy = None
    
    def concept_fn(self, state) -> int:
        """Map state to discrete concept index."""
        concept_vec = self.concept_extractor(state)
        
        if hasattr(self.concept_extractor, 'to_index'):
            return self.concept_extractor.to_index(state)
        else:
            # For soft concepts, discretize
            return int(np.argmax(concept_vec[:5]))  # First 5 dims are probs
    
    def learn_concept_policies(
        self, 
        trajectories: List[List[Dict]],
        behavior_policy,
        eval_policy,
        smoothing: float = 1.0
    ):
        """
        Learn concept-conditioned policies from trajectory data.
        """
        from policies import ConceptPolicy
        
        # Concept-based behavior policy
        self.behavior_concept_policy = ConceptPolicy(
            self.concept_fn, behavior_policy, 
            n_actions=4, n_concepts=32  # 2^5 for hard concepts
        )
        self.behavior_concept_policy.learn_from_data(trajectories, smoothing)
        
        # For evaluation policy, simulate what it would do
        # Generate some trajectories with eval policy to learn its concept distribution
        eval_trajectories = []
        from gridworld import collect_trajectory
        for _ in range(100):
            traj = collect_trajectory(self.env, eval_policy, max_steps=50)
            eval_trajectories.append(traj)
        
        self.eval_concept_policy = ConceptPolicy(
            self.concept_fn, eval_policy,
            n_actions=4, n_concepts=32
        )
        self.eval_concept_policy.learn_from_data(eval_trajectories, smoothing)
    
    def estimate(
        self, 
        trajectories: List[List[Dict]],
        gamma: float = 0.99
    ) -> Tuple[float, float]:
        """
        Run CPDIS estimation.
        """
        if self.behavior_concept_policy is None:
            raise ValueError("Must call learn_concept_policies first")
        
        return cpdis_estimate(
            trajectories,
            self.concept_fn,
            self.behavior_concept_policy,
            self.eval_concept_policy,
            gamma
        )


if __name__ == "__main__":
    from gridworld import WindyGridworld, collect_trajectory
    from policies import EpsilonGreedyPolicy, OptimalPolicy
    from concepts import HardConcepts
    
    # Setup
    env = WindyGridworld()
    behavior = EpsilonGreedyPolicy(env, epsilon=0.4, seed=42)
    evaluation = OptimalPolicy(env, epsilon=0.05, seed=42)
    
    # Collect trajectories
    print("Collecting trajectories...")
    trajectories = []
    for _ in range(200):
        traj = collect_trajectory(env, behavior, max_steps=50)
        trajectories.append(traj)
    
    # Ground truth
    print("\nComputing ground truth...")
    true_value, true_std = monte_carlo_ground_truth(env, evaluation, n_episodes=1000)
    print(f"  True value: {true_value:.4f} ± {true_std:.4f}")
    
    # Standard PDIS
    print("\nRunning standard PDIS...")
    pdis_est, pdis_var = pdis_estimate(trajectories, behavior, evaluation)
    print(f"  PDIS estimate: {pdis_est:.4f} (variance: {pdis_var:.6f})")
    print(f"  Error: {abs(pdis_est - true_value):.4f}")
    
    # ESS
    ess = effective_sample_size(trajectories, behavior, evaluation)
    print(f"  Effective sample size: {ess:.1f} / {len(trajectories)}")
    
    # Concept-based OPE
    print("\nRunning Concept-based OPE...")
    hard_concepts = HardConcepts(env)
    concept_ope = ConceptBasedOPE(hard_concepts, env)
    concept_ope.learn_concept_policies(trajectories, behavior, evaluation)
    
    cpdis_est, cpdis_var = concept_ope.estimate(trajectories)
    print(f"  CPDIS estimate: {cpdis_est:.4f} (variance: {cpdis_var:.6f})")
    print(f"  Error: {abs(cpdis_est - true_value):.4f}")
    
    # Compare truncated estimates at different horizons
    print("\nTruncated PDIS at different horizons:")
    for h in [5, 10, 20, 30]:
        est, var = truncated_pdis_estimate(trajectories, behavior, evaluation, horizon=h)
        print(f"  Horizon {h}: estimate = {est:.4f}, error = {abs(est - true_value):.4f}")
