"""
Windy Gridworld Environment

A standard 7x10 gridworld with wind affecting agent movement.
Used as the test environment for temporal leakage experiments.
"""

import numpy as np
from typing import Tuple, List, Dict, Optional


class WindyGridworld:
    """
    7x10 Windy Gridworld from Sutton & Barto.
    
    Wind pushes agent upward in certain columns.
    Goal: Navigate from start (3,0) to goal (3,7).
    """
    
    def __init__(self):
        self.height = 7
        self.width = 10
        self.start = (3, 0)
        self.goal = (3, 7)
        
        # Wind strength per column (pushes agent up)
        self.wind = np.array([0, 0, 0, 1, 1, 1, 2, 2, 1, 0])
        
        # Actions: 0=up, 1=right, 2=down, 3=left
        self.n_actions = 4
        self.action_effects = {
            0: (-1, 0),  # up
            1: (0, 1),   # right
            2: (1, 0),   # down
            3: (0, -1)   # left
        }
        self.action_names = ['up', 'right', 'down', 'left']
        
        self.state = None
        self.reset()
    
    def reset(self) -> Tuple[int, int]:
        """Reset to start position."""
        self.state = self.start
        return self.state
    
    def step(self, action: int) -> Tuple[Tuple[int, int], float, bool, dict]:
        """
        Take action, return (next_state, reward, done, info).
        """
        row, col = self.state
        
        # Apply action
        drow, dcol = self.action_effects[action]
        new_row = row + drow
        new_col = col + dcol
        
        # Apply wind (pushes up = negative row)
        if 0 <= new_col < self.width:
            new_row -= self.wind[new_col]
        
        # Clip to bounds
        new_row = np.clip(new_row, 0, self.height - 1)
        new_col = np.clip(new_col, 0, self.width - 1)
        
        self.state = (new_row, new_col)
        
        # Check if goal reached
        done = (self.state == self.goal)
        reward = 0.0 if done else -1.0
        
        info = {'wind': self.wind[new_col] if 0 <= new_col < self.width else 0}
        
        return self.state, reward, done, info
    
    def state_to_features(self, state: Tuple[int, int]) -> np.ndarray:
        """
        Convert state to feature vector.
        
        Features:
        - Normalized row position
        - Normalized col position
        - Distance to goal (normalized)
        - Wind at current column
        - One-hot encoded quadrant
        """
        row, col = state
        
        # Basic position features
        norm_row = row / (self.height - 1)
        norm_col = col / (self.width - 1)
        
        # Distance to goal
        dist_row = abs(row - self.goal[0]) / (self.height - 1)
        dist_col = abs(col - self.goal[1]) / (self.width - 1)
        dist = np.sqrt(dist_row**2 + dist_col**2)
        
        # Wind feature
        wind = self.wind[col] / 2.0  # Normalize
        
        # Quadrant (one-hot)
        quadrant = np.zeros(4)
        q_row = 0 if row < self.height // 2 else 1
        q_col = 0 if col < self.width // 2 else 1
        quadrant[q_row * 2 + q_col] = 1.0
        
        return np.array([norm_row, norm_col, dist, wind] + list(quadrant))
    
    def state_to_index(self, state: Tuple[int, int]) -> int:
        """Convert (row, col) to flat index."""
        return state[0] * self.width + state[1]
    
    def index_to_state(self, idx: int) -> Tuple[int, int]:
        """Convert flat index to (row, col)."""
        return (idx // self.width, idx % self.width)
    
    @property
    def n_states(self) -> int:
        return self.height * self.width
    
    def render(self) -> str:
        """Return string representation of grid."""
        grid = [['.' for _ in range(self.width)] for _ in range(self.height)]
        grid[self.goal[0]][self.goal[1]] = 'G'
        grid[self.start[0]][self.start[1]] = 'S'
        if self.state != self.start and self.state != self.goal:
            grid[self.state[0]][self.state[1]] = 'A'
        
        lines = [''.join(row) for row in grid]
        lines.append('Wind: ' + ''.join(str(w) for w in self.wind))
        return '\n'.join(lines)


def collect_trajectory(
    env: WindyGridworld, 
    policy, 
    max_steps: int = 100,
    include_features: bool = True
) -> List[Dict]:
    """
    Collect a single trajectory using the given policy.
    
    Returns list of dicts with keys:
    - state: (row, col)
    - action: int
    - reward: float
    - next_state: (row, col)
    - done: bool
    - features: np.ndarray (if include_features=True)
    """
    trajectory = []
    state = env.reset()
    
    for t in range(max_steps):
        action = policy.sample_action(state)
        next_state, reward, done, info = env.step(action)
        
        step_data = {
            'state': state,
            'action': action,
            'reward': reward,
            'next_state': next_state,
            'done': done,
            't': t
        }
        
        if include_features:
            step_data['features'] = env.state_to_features(state)
        
        trajectory.append(step_data)
        
        if done:
            break
        
        state = next_state
    
    return trajectory


if __name__ == "__main__":
    # Test the environment
    env = WindyGridworld()
    print("Environment:")
    print(env.render())
    print()
    
    # Test random trajectory
    class RandomPolicy:
        def __init__(self, n_actions):
            self.n_actions = n_actions
        def sample_action(self, state):
            return np.random.randint(self.n_actions)
    
    policy = RandomPolicy(4)
    traj = collect_trajectory(env, policy, max_steps=20)
    
    print(f"Trajectory length: {len(traj)}")
    print(f"Final state: {traj[-1]['next_state']}")
    print(f"Total reward: {sum(s['reward'] for s in traj)}")
