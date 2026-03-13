"""
Windy Gridworld Environment

A simple gridworld where certain columns have "wind" that pushes the agent upward.
This creates stochasticity that makes OPE challenging.

Grid layout (7x10):
- Start: (3, 0)
- Goal: (3, 7)
- Wind strength varies by column
"""

import numpy as np
from typing import Tuple, List, Optional


class WindyGridworld:
    """
    Windy Gridworld environment for OPE experiments.
    
    Actions:
        0: Up
        1: Right
        2: Down
        3: Left
    
    The wind pushes the agent upward by a certain amount depending on the column.
    """
    
    def __init__(
        self,
        height: int = 7,
        width: int = 10,
        start: Tuple[int, int] = (3, 0),
        goal: Tuple[int, int] = (3, 7),
        wind: Optional[List[int]] = None,
        stochastic_wind: bool = True
    ):
        self.height = height
        self.width = width
        self.start = start
        self.goal = goal
        
        # Wind strength per column (default from Sutton & Barto)
        if wind is None:
            self.wind = [0, 0, 0, 1, 1, 1, 2, 2, 1, 0]
        else:
            self.wind = wind
            
        self.stochastic_wind = stochastic_wind
        self.n_states = height * width
        self.n_actions = 4
        
        # Action effects: (row_delta, col_delta)
        self.action_effects = {
            0: (-1, 0),  # Up
            1: (0, 1),   # Right
            2: (1, 0),   # Down
            3: (0, -1)   # Left
        }
        
        self.state = None
        self.reset()
    
    def reset(self) -> np.ndarray:
        """Reset to start state."""
        self.state = self.start
        return self._get_obs()
    
    def _get_obs(self) -> np.ndarray:
        """Return current state as numpy array."""
        return np.array(self.state, dtype=np.float32)
    
    def _clip_state(self, row: int, col: int) -> Tuple[int, int]:
        """Clip state to grid boundaries."""
        row = max(0, min(self.height - 1, row))
        col = max(0, min(self.width - 1, col))
        return (row, col)
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        """
        Take action and return (next_state, reward, done, info).
        
        Args:
            action: Integer action (0=up, 1=right, 2=down, 3=left)
            
        Returns:
            obs: Next state observation
            reward: -1 for each step (we want to reach goal fast)
            done: True if goal reached
            info: Additional information
        """
        assert 0 <= action < self.n_actions, f"Invalid action {action}"
        
        row, col = self.state
        d_row, d_col = self.action_effects[action]
        
        # Apply action
        new_row = row + d_row
        new_col = col + d_col
        
        # Apply wind (pushes upward, i.e., decreases row)
        wind_strength = self.wind[min(col, len(self.wind) - 1)]
        
        if self.stochastic_wind and wind_strength > 0:
            # Stochastic wind: wind_strength ± 1 with some probability
            wind_noise = np.random.choice([-1, 0, 1], p=[0.2, 0.6, 0.2])
            wind_strength = max(0, wind_strength + wind_noise)
        
        new_row = new_row - wind_strength  # Wind pushes up (decreases row)
        
        # Clip to grid
        self.state = self._clip_state(new_row, new_col)
        
        # Check if goal reached
        done = (self.state == self.goal)
        reward = 0.0 if done else -1.0
        
        return self._get_obs(), reward, done, {"wind": wind_strength}
    
    def state_to_index(self, state: Tuple[int, int]) -> int:
        """Convert (row, col) to flat index."""
        return state[0] * self.width + state[1]
    
    def index_to_state(self, index: int) -> Tuple[int, int]:
        """Convert flat index to (row, col)."""
        return (index // self.width, index % self.width)
    
    def get_state_features(self, state: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Get feature representation of state for concept learning.
        
        Features:
            - Normalized row position
            - Normalized col position
            - Distance to goal (normalized)
            - Wind strength at current column
            - Is in wind zone (binary)
            - Distance to nearest wall
        """
        if state is None:
            state = self._get_obs()
        
        row, col = int(state[0]), int(state[1])
        
        # Normalized positions
        norm_row = row / (self.height - 1)
        norm_col = col / (self.width - 1)
        
        # Distance to goal (Manhattan, normalized)
        goal_dist = (abs(row - self.goal[0]) + abs(col - self.goal[1]))
        max_dist = self.height + self.width - 2
        norm_goal_dist = goal_dist / max_dist
        
        # Wind features
        wind_strength = self.wind[min(col, len(self.wind) - 1)] / 2.0  # Normalize
        in_wind_zone = 1.0 if self.wind[min(col, len(self.wind) - 1)] > 0 else 0.0
        
        # Distance to nearest wall (normalized)
        dist_to_wall = min(row, self.height - 1 - row, col, self.width - 1 - col)
        norm_wall_dist = dist_to_wall / (min(self.height, self.width) / 2)
        
        return np.array([
            norm_row,
            norm_col,
            norm_goal_dist,
            wind_strength,
            in_wind_zone,
            norm_wall_dist
        ], dtype=np.float32)
    
    def render(self) -> str:
        """Render grid as string."""
        grid = [['.' for _ in range(self.width)] for _ in range(self.height)]
        
        # Mark wind strength
        for col, w in enumerate(self.wind):
            if w > 0:
                for row in range(self.height):
                    grid[row][col] = str(w)
        
        # Mark start, goal, agent
        grid[self.start[0]][self.start[1]] = 'S'
        grid[self.goal[0]][self.goal[1]] = 'G'
        if self.state != self.start and self.state != self.goal:
            grid[self.state[0]][self.state[1]] = 'A'
        
        # Build string
        lines = [''.join(row) for row in grid]
        lines.append(f"Wind: {self.wind}")
        return '\n'.join(lines)


def collect_trajectory(
    env: WindyGridworld,
    policy,  # Callable that takes state and returns action
    max_steps: int = 100
) -> List[dict]:
    """
    Collect a single trajectory using the given policy.
    
    Returns:
        List of dicts with keys: state, action, reward, next_state, done, features
    """
    trajectory = []
    state = env.reset()
    
    for t in range(max_steps):
        features = env.get_state_features(state)
        action = policy(state)
        next_state, reward, done, info = env.step(action)
        
        trajectory.append({
            'state': state.copy(),
            'action': action,
            'reward': reward,
            'next_state': next_state.copy(),
            'done': done,
            'features': features.copy(),
            't': t
        })
        
        if done:
            break
            
        state = next_state
    
    return trajectory


if __name__ == "__main__":
    # Test the environment
    env = WindyGridworld()
    print("Windy Gridworld Environment")
    print("=" * 40)
    print(env.render())
    print()
    
    # Test random policy
    env.reset()
    print("Running random policy for 10 steps:")
    for i in range(10):
        action = np.random.randint(4)
        obs, reward, done, info = env.step(action)
        action_names = ['Up', 'Right', 'Down', 'Left']
        print(f"  Step {i+1}: {action_names[action]} -> State {tuple(obs.astype(int))}, Reward {reward}")
        if done:
            print("  Goal reached!")
            break
    
    print()
    print("State features at start:")
    env.reset()
    print(f"  {env.get_state_features()}")
