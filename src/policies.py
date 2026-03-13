"""
Policies for Windy Gridworld

Includes:
- RandomPolicy: Uniform random actions
- EpsilonGreedyPolicy: ε-greedy based on value function
- OptimalPolicy: Near-optimal via value iteration with optional ε exploration
"""

import numpy as np
from typing import Tuple, Dict, Optional


class RandomPolicy:
    """Uniform random policy."""
    
    def __init__(self, n_actions: int = 4, seed: int = None):
        self.n_actions = n_actions
        self.rng = np.random.RandomState(seed)
    
    def sample_action(self, state: Tuple[int, int]) -> int:
        return self.rng.randint(self.n_actions)
    
    def action_probs(self, state: Tuple[int, int]) -> np.ndarray:
        return np.ones(self.n_actions) / self.n_actions
    
    def prob(self, state: Tuple[int, int], action: int) -> float:
        return 1.0 / self.n_actions


class EpsilonGreedyPolicy:
    """
    ε-greedy policy based on value iteration.
    
    With probability ε: random action
    With probability 1-ε: greedy action
    """
    
    def __init__(self, env, epsilon: float = 0.1, gamma: float = 0.99, seed: int = None):
        self.env = env
        self.epsilon = epsilon
        self.gamma = gamma
        self.rng = np.random.RandomState(seed)
        
        # Compute value function and policy via value iteration
        self.V, self.Q = self._value_iteration()
    
    def _value_iteration(self, max_iter: int = 1000, tol: float = 1e-6) -> Tuple[np.ndarray, np.ndarray]:
        """Run value iteration to compute V and Q."""
        n_states = self.env.n_states
        n_actions = self.env.n_actions
        
        V = np.zeros(n_states)
        Q = np.zeros((n_states, n_actions))
        
        for iteration in range(max_iter):
            V_old = V.copy()
            
            for s_idx in range(n_states):
                state = self.env.index_to_state(s_idx)
                
                if state == self.env.goal:
                    V[s_idx] = 0
                    Q[s_idx, :] = 0
                    continue
                
                for a in range(n_actions):
                    # Simulate taking action a from state
                    self.env.state = state
                    next_state, reward, done, _ = self.env.step(a)
                    next_idx = self.env.state_to_index(next_state)
                    
                    Q[s_idx, a] = reward + self.gamma * V[next_idx]
                
                V[s_idx] = np.max(Q[s_idx])
            
            if np.max(np.abs(V - V_old)) < tol:
                break
        
        return V, Q
    
    def sample_action(self, state: Tuple[int, int]) -> int:
        """Sample action according to ε-greedy."""
        if self.rng.random() < self.epsilon:
            return self.rng.randint(self.env.n_actions)
        else:
            s_idx = self.env.state_to_index(state)
            return np.argmax(self.Q[s_idx])
    
    def action_probs(self, state: Tuple[int, int]) -> np.ndarray:
        """Return probability distribution over actions."""
        s_idx = self.env.state_to_index(state)
        greedy_action = np.argmax(self.Q[s_idx])
        
        probs = np.ones(self.env.n_actions) * (self.epsilon / self.env.n_actions)
        probs[greedy_action] += (1 - self.epsilon)
        
        return probs
    
    def prob(self, state: Tuple[int, int], action: int) -> float:
        """Return probability of taking action in state."""
        return self.action_probs(state)[action]


class OptimalPolicy(EpsilonGreedyPolicy):
    """
    Near-optimal policy from value iteration.
    
    Same as EpsilonGreedyPolicy but with very low epsilon by default.
    """
    
    def __init__(self, env, epsilon: float = 0.05, gamma: float = 0.99, seed: int = None):
        super().__init__(env, epsilon=epsilon, gamma=gamma, seed=seed)


class ConceptPolicy:
    """
    Policy conditioned on concepts instead of states.
    
    π^c(a|c) where c = φ(s)
    
    This is learned by aggregating state-action statistics over concept groups.
    """
    
    def __init__(self, concept_fn, base_policy, n_actions: int = 4, n_concepts: int = 5):
        """
        Args:
            concept_fn: Function mapping state -> concept index
            base_policy: The underlying state-based policy to aggregate
            n_actions: Number of actions
            n_concepts: Number of discrete concept values
        """
        self.concept_fn = concept_fn
        self.base_policy = base_policy
        self.n_actions = n_actions
        self.n_concepts = n_concepts
        
        # Will be populated by learn_from_data
        self.concept_action_counts = None
        self.concept_action_probs = None
    
    def learn_from_data(self, trajectories, smoothing: float = 1.0):
        """
        Learn concept-based policy from trajectory data.
        
        Args:
            trajectories: List of trajectories
            smoothing: Laplace smoothing parameter
        """
        # Count (concept, action) pairs
        counts = np.zeros((self.n_concepts, self.n_actions)) + smoothing
        
        for traj in trajectories:
            for step in traj:
                state = step['state']
                action = step['action']
                concept = self.concept_fn(state)
                
                if isinstance(concept, np.ndarray):
                    concept = int(np.argmax(concept))
                
                counts[concept, action] += 1
        
        # Normalize to probabilities
        self.concept_action_counts = counts
        self.concept_action_probs = counts / counts.sum(axis=1, keepdims=True)
    
    def action_probs(self, state: Tuple[int, int]) -> np.ndarray:
        """Return action probabilities given state (via concept)."""
        concept = self.concept_fn(state)
        if isinstance(concept, np.ndarray):
            concept = int(np.argmax(concept))
        
        if self.concept_action_probs is None:
            # Fall back to base policy
            return self.base_policy.action_probs(state)
        
        return self.concept_action_probs[concept]
    
    def prob(self, state: Tuple[int, int], action: int) -> float:
        """Return probability of taking action in state."""
        return self.action_probs(state)[action]


if __name__ == "__main__":
    from gridworld import WindyGridworld, collect_trajectory
    
    env = WindyGridworld()
    
    # Test policies
    print("Testing EpsilonGreedyPolicy (ε=0.3)...")
    behavior = EpsilonGreedyPolicy(env, epsilon=0.3, seed=42)
    print(f"  Action probs at start {env.start}: {behavior.action_probs(env.start)}")
    
    print("\nTesting OptimalPolicy (ε=0.05)...")
    optimal = OptimalPolicy(env, epsilon=0.05, seed=42)
    print(f"  Action probs at start {env.start}: {optimal.action_probs(env.start)}")
    
    # Compare trajectory lengths
    print("\nCollecting trajectories...")
    
    behavior_lengths = []
    for _ in range(20):
        traj = collect_trajectory(env, behavior, max_steps=100)
        behavior_lengths.append(len(traj))
    
    optimal_lengths = []
    for _ in range(20):
        traj = collect_trajectory(env, optimal, max_steps=100)
        optimal_lengths.append(len(traj))
    
    print(f"  Behavior (ε=0.3): mean length = {np.mean(behavior_lengths):.1f}")
    print(f"  Optimal (ε=0.05): mean length = {np.mean(optimal_lengths):.1f}")
