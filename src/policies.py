"""
Policies for Off-Policy Evaluation

We need two policies:
1. Behavior Policy (π_b): Collects the training data (suboptimal, exploratory)
2. Evaluation Policy (π_e): The policy we want to evaluate (near-optimal)

The key is that π_e differs from π_b, creating distribution shift over time.
"""

import numpy as np
from typing import Tuple, Optional
from gridworld import WindyGridworld


class RandomPolicy:
    """Uniformly random policy - maximum exploration."""
    
    def __init__(self, n_actions: int = 4):
        self.n_actions = n_actions
    
    def __call__(self, state: np.ndarray) -> int:
        return np.random.randint(self.n_actions)
    
    def get_prob(self, state: np.ndarray, action: int) -> float:
        """Return probability of taking action in state."""
        return 1.0 / self.n_actions
    
    def get_all_probs(self, state: np.ndarray) -> np.ndarray:
        """Return probability distribution over all actions."""
        return np.ones(self.n_actions) / self.n_actions


class EpsilonGreedyPolicy:
    """
    Epsilon-greedy policy based on a value function.
    
    With probability epsilon: take random action
    With probability 1-epsilon: take greedy action toward goal
    """
    
    def __init__(
        self,
        env: WindyGridworld,
        epsilon: float = 0.1,
        seed: Optional[int] = None
    ):
        self.env = env
        self.epsilon = epsilon
        self.n_actions = env.n_actions
        self.rng = np.random.RandomState(seed)
        
        # Precompute greedy actions (move toward goal)
        self._compute_greedy_actions()
    
    def _compute_greedy_actions(self):
        """Compute greedy action for each state (simple heuristic: move toward goal)."""
        self.greedy_actions = {}
        
        for row in range(self.env.height):
            for col in range(self.env.width):
                state = (row, col)
                goal = self.env.goal
                
                # Determine best action to move toward goal
                # Account for wind: prefer moving right in wind zones
                row_diff = goal[0] - row
                col_diff = goal[1] - col
                
                # Priority: horizontal movement first (to get through wind), then vertical
                if col_diff > 0:
                    best_action = 1  # Right
                elif col_diff < 0:
                    best_action = 3  # Left
                elif row_diff > 0:
                    best_action = 2  # Down
                elif row_diff < 0:
                    best_action = 0  # Up
                else:
                    best_action = 0  # At goal, doesn't matter
                
                self.greedy_actions[state] = best_action
    
    def __call__(self, state: np.ndarray) -> int:
        """Select action using epsilon-greedy."""
        if self.rng.random() < self.epsilon:
            return self.rng.randint(self.n_actions)
        else:
            state_tuple = (int(state[0]), int(state[1]))
            return self.greedy_actions.get(state_tuple, 0)
    
    def get_prob(self, state: np.ndarray, action: int) -> float:
        """Return probability of taking action in state."""
        state_tuple = (int(state[0]), int(state[1]))
        greedy_action = self.greedy_actions.get(state_tuple, 0)
        
        if action == greedy_action:
            return 1.0 - self.epsilon + self.epsilon / self.n_actions
        else:
            return self.epsilon / self.n_actions
    
    def get_all_probs(self, state: np.ndarray) -> np.ndarray:
        """Return probability distribution over all actions."""
        probs = np.ones(self.n_actions) * (self.epsilon / self.n_actions)
        state_tuple = (int(state[0]), int(state[1]))
        greedy_action = self.greedy_actions.get(state_tuple, 0)
        probs[greedy_action] = 1.0 - self.epsilon + self.epsilon / self.n_actions
        return probs


class OptimalPolicy:
    """
    Near-optimal policy computed using value iteration.
    
    This gives us a strong evaluation policy that differs significantly
    from the behavior policy.
    """
    
    def __init__(
        self,
        env: WindyGridworld,
        gamma: float = 0.99,
        epsilon: float = 0.05,  # Small exploration for numerical stability
        seed: Optional[int] = None
    ):
        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon
        self.n_actions = env.n_actions
        self.rng = np.random.RandomState(seed)
        
        # Compute optimal value function and policy
        self._value_iteration()
    
    def _value_iteration(self, theta: float = 1e-6, max_iters: int = 1000):
        """Compute optimal policy using value iteration."""
        V = np.zeros((self.env.height, self.env.width))
        
        for iteration in range(max_iters):
            delta = 0
            
            for row in range(self.env.height):
                for col in range(self.env.width):
                    if (row, col) == self.env.goal:
                        continue
                    
                    v = V[row, col]
                    
                    # Compute Q-values for all actions
                    q_values = []
                    for action in range(self.n_actions):
                        q = self._compute_q_value(row, col, action, V)
                        q_values.append(q)
                    
                    V[row, col] = max(q_values)
                    delta = max(delta, abs(v - V[row, col]))
            
            if delta < theta:
                break
        
        # Extract greedy policy
        self.V = V
        self.greedy_actions = {}
        
        for row in range(self.env.height):
            for col in range(self.env.width):
                if (row, col) == self.env.goal:
                    self.greedy_actions[(row, col)] = 0
                    continue
                
                q_values = []
                for action in range(self.n_actions):
                    q = self._compute_q_value(row, col, action, V)
                    q_values.append(q)
                
                self.greedy_actions[(row, col)] = int(np.argmax(q_values))
    
    def _compute_q_value(self, row: int, col: int, action: int, V: np.ndarray) -> float:
        """Compute Q-value for state-action pair."""
        # Get action effect
        d_row, d_col = self.env.action_effects[action]
        new_col = max(0, min(self.env.width - 1, col + d_col))
        
        # Expected value over wind stochasticity
        wind_base = self.env.wind[min(new_col, len(self.env.wind) - 1)]
        
        if wind_base == 0 or not self.env.stochastic_wind:
            # Deterministic case
            new_row = row + d_row - wind_base
            new_row = max(0, min(self.env.height - 1, new_row))
            
            if (new_row, new_col) == self.env.goal:
                return 0.0
            else:
                return -1.0 + self.gamma * V[new_row, new_col]
        else:
            # Stochastic wind: average over possible outcomes
            q = 0.0
            for wind_delta, prob in [(-1, 0.2), (0, 0.6), (1, 0.2)]:
                wind = max(0, wind_base + wind_delta)
                new_row = row + d_row - wind
                new_row = max(0, min(self.env.height - 1, new_row))
                
                if (new_row, new_col) == self.env.goal:
                    q += prob * 0.0
                else:
                    q += prob * (-1.0 + self.gamma * V[new_row, new_col])
            return q
    
    def __call__(self, state: np.ndarray) -> int:
        """Select action using epsilon-greedy on optimal policy."""
        if self.rng.random() < self.epsilon:
            return self.rng.randint(self.n_actions)
        else:
            state_tuple = (int(state[0]), int(state[1]))
            return self.greedy_actions.get(state_tuple, 0)
    
    def get_prob(self, state: np.ndarray, action: int) -> float:
        """Return probability of taking action in state."""
        state_tuple = (int(state[0]), int(state[1]))
        greedy_action = self.greedy_actions.get(state_tuple, 0)
        
        if action == greedy_action:
            return 1.0 - self.epsilon + self.epsilon / self.n_actions
        else:
            return self.epsilon / self.n_actions
    
    def get_all_probs(self, state: np.ndarray) -> np.ndarray:
        """Return probability distribution over all actions."""
        probs = np.ones(self.n_actions) * (self.epsilon / self.n_actions)
        state_tuple = (int(state[0]), int(state[1]))
        greedy_action = self.greedy_actions.get(state_tuple, 0)
        probs[greedy_action] = 1.0 - self.epsilon + self.epsilon / self.n_actions
        return probs


def compute_importance_ratio(
    behavior_policy,
    eval_policy,
    state: np.ndarray,
    action: int
) -> float:
    """
    Compute importance sampling ratio: π_e(a|s) / π_b(a|s)
    """
    prob_b = behavior_policy.get_prob(state, action)
    prob_e = eval_policy.get_prob(state, action)
    
    # Clip to avoid division by zero
    prob_b = max(prob_b, 1e-10)
    
    return prob_e / prob_b


if __name__ == "__main__":
    # Test policies
    env = WindyGridworld()
    
    print("Testing Policies")
    print("=" * 50)
    
    # Test random policy
    random_policy = RandomPolicy()
    print("\nRandom Policy:")
    print(f"  Action probs: {random_policy.get_all_probs(np.array([3, 0]))}")
    
    # Test epsilon-greedy behavior policy
    behavior_policy = EpsilonGreedyPolicy(env, epsilon=0.3, seed=42)
    print("\nBehavior Policy (ε=0.3):")
    test_state = np.array([3, 0])  # Start state
    print(f"  At start {test_state}: probs = {behavior_policy.get_all_probs(test_state)}")
    print(f"  Greedy action: {behavior_policy.greedy_actions[(3, 0)]}")
    
    # Test optimal evaluation policy
    eval_policy = OptimalPolicy(env, epsilon=0.05, seed=42)
    print("\nEvaluation Policy (optimal, ε=0.05):")
    print(f"  At start {test_state}: probs = {eval_policy.get_all_probs(test_state)}")
    print(f"  Greedy action: {eval_policy.greedy_actions[(3, 0)]}")
    
    # Show value function
    print("\nOptimal Value Function:")
    for row in range(env.height):
        values = [f"{eval_policy.V[row, col]:6.2f}" for col in range(env.width)]
        print("  " + " ".join(values))
    
    # Test importance ratios
    print("\nImportance Ratios at start state:")
    for action in range(4):
        ratio = compute_importance_ratio(behavior_policy, eval_policy, test_state, action)
        action_names = ['Up', 'Right', 'Down', 'Left']
        print(f"  {action_names[action]}: {ratio:.3f}")
    
    # Run a trajectory comparison
    print("\n" + "=" * 50)
    print("Trajectory Comparison")
    print("=" * 50)
    
    from gridworld import collect_trajectory
    
    np.random.seed(42)
    
    # Behavior policy trajectory
    traj_b = collect_trajectory(env, behavior_policy, max_steps=50)
    print(f"\nBehavior policy: {len(traj_b)} steps")
    
    # Evaluation policy trajectory
    env.reset()
    traj_e = collect_trajectory(env, eval_policy, max_steps=50)
    print(f"Evaluation policy: {len(traj_e)} steps")
