"""
Utility functions for the temporal leakage experiments.
"""

import numpy as np
from typing import List, Dict, Optional


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    np.random.seed(seed)


def compute_returns(trajectory: List[Dict], gamma: float = 0.99) -> float:
    """Compute discounted return for a trajectory."""
    G = 0.0
    for t, step in enumerate(trajectory):
        G += (gamma ** t) * step['reward']
    return G


def trajectory_statistics(trajectories: List[List[Dict]]) -> Dict:
    """
    Compute statistics over a set of trajectories.
    
    Returns dict with:
    - n_trajectories
    - mean_length, std_length, min_length, max_length
    - mean_return, std_return
    - completion_rate (fraction reaching goal)
    """
    lengths = [len(traj) for traj in trajectories]
    returns = [compute_returns(traj) for traj in trajectories]
    
    # Check for completion (last step has done=True)
    completions = sum(1 for traj in trajectories if traj[-1].get('done', False))
    
    return {
        'n_trajectories': len(trajectories),
        'mean_length': np.mean(lengths),
        'std_length': np.std(lengths),
        'min_length': np.min(lengths),
        'max_length': np.max(lengths),
        'mean_return': np.mean(returns),
        'std_return': np.std(returns),
        'completion_rate': completions / len(trajectories)
    }


def print_trajectory_stats(trajectories: List[List[Dict]], name: str = "Trajectories"):
    """Print formatted trajectory statistics."""
    stats = trajectory_statistics(trajectories)
    print(f"    {name}:")
    print(f"      Count: {stats['n_trajectories']}")
    print(f"      Length: {stats['mean_length']:.1f} ± {stats['std_length']:.1f} "
          f"(range: {stats['min_length']}-{stats['max_length']})")
    print(f"      Return: {stats['mean_return']:.2f} ± {stats['std_return']:.2f}")
    print(f"      Completion rate: {stats['completion_rate']*100:.1f}%")


def states_at_timestep(
    trajectories: List[List[Dict]], 
    timestep: int
) -> List[tuple]:
    """
    Extract all states that appear at a specific timestep across trajectories.
    
    Useful for analyzing distribution shift over time.
    """
    states = []
    for traj in trajectories:
        if len(traj) > timestep:
            states.append(traj[timestep]['state'])
    return states


def compute_state_distribution(
    states: List[tuple],
    grid_height: int = 7,
    grid_width: int = 10
) -> np.ndarray:
    """
    Compute empirical state distribution as a grid.
    """
    dist = np.zeros((grid_height, grid_width))
    
    for state in states:
        row, col = state
        dist[row, col] += 1
    
    if dist.sum() > 0:
        dist /= dist.sum()
    
    return dist


def kl_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-10) -> float:
    """
    Compute KL divergence D_KL(p || q).
    
    p and q should be normalized probability distributions.
    """
    p = p.flatten() + eps
    q = q.flatten() + eps
    
    p = p / p.sum()
    q = q / q.sum()
    
    return np.sum(p * np.log(p / q))


def measure_distribution_shift(
    trajectories: List[List[Dict]],
    reference_timestep: int = 0,
    target_timestep: int = 10,
    grid_height: int = 7,
    grid_width: int = 10
) -> float:
    """
    Measure distribution shift between two timesteps using KL divergence.
    """
    states_ref = states_at_timestep(trajectories, reference_timestep)
    states_target = states_at_timestep(trajectories, target_timestep)
    
    if len(states_ref) < 10 or len(states_target) < 10:
        return np.nan
    
    dist_ref = compute_state_distribution(states_ref, grid_height, grid_width)
    dist_target = compute_state_distribution(states_target, grid_height, grid_width)
    
    return kl_divergence(dist_target, dist_ref)


def visualize_trajectory(
    trajectory: List[Dict],
    grid_height: int = 7,
    grid_width: int = 10,
    show_rewards: bool = False
) -> str:
    """
    Create ASCII visualization of a trajectory.
    """
    grid = [['.' for _ in range(grid_width)] for _ in range(grid_height)]
    
    # Mark path
    for i, step in enumerate(trajectory):
        row, col = step['state']
        if i == 0:
            grid[row][col] = 'S'
        elif step.get('done', False):
            grid[row][col] = 'G'
        else:
            grid[row][col] = str(i % 10)
    
    lines = [''.join(row) for row in grid]
    
    if show_rewards:
        total_reward = sum(s['reward'] for s in trajectory)
        lines.append(f"Total reward: {total_reward:.2f}")
    
    return '\n'.join(lines)


if __name__ == "__main__":
    from gridworld import WindyGridworld, collect_trajectory
    from policies import EpsilonGreedyPolicy
    
    env = WindyGridworld()
    policy = EpsilonGreedyPolicy(env, epsilon=0.3, seed=42)
    
    # Collect trajectories
    trajectories = []
    for _ in range(50):
        traj = collect_trajectory(env, policy, max_steps=50)
        trajectories.append(traj)
    
    # Print statistics
    print_trajectory_stats(trajectories, "Test trajectories")
    
    # Measure distribution shift
    print("\nDistribution shift (KL divergence):")
    for t in [5, 10, 20, 30]:
        shift = measure_distribution_shift(trajectories, reference_timestep=0, target_timestep=t)
        print(f"  t=0 -> t={t}: {shift:.4f}")
    
    # Visualize a trajectory
    print("\nSample trajectory:")
    print(visualize_trajectory(trajectories[0], show_rewards=True))
