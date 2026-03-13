"""
Utility functions for the temporal leakage OPE project.
"""

import numpy as np
import torch
import random
from typing import List, Optional


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_returns(
    trajectory: List[dict],
    gamma: float = 0.99
) -> float:
    """Compute discounted return for a trajectory."""
    total_return = 0.0
    for t, step in enumerate(trajectory):
        total_return += (gamma ** t) * step['reward']
    return total_return


def split_trajectories(
    trajectories: List[List[dict]],
    train_ratio: float = 0.8,
    seed: int = 42
) -> tuple:
    """Split trajectories into train and test sets."""
    np.random.seed(seed)
    n = len(trajectories)
    indices = np.random.permutation(n)
    
    split_idx = int(n * train_ratio)
    train_indices = indices[:split_idx]
    test_indices = indices[split_idx:]
    
    train_trajs = [trajectories[i] for i in train_indices]
    test_trajs = [trajectories[i] for i in test_indices]
    
    return train_trajs, test_trajs


def trajectory_statistics(trajectories: List[List[dict]]) -> dict:
    """Compute statistics about a set of trajectories."""
    lengths = [len(t) for t in trajectories]
    returns = [compute_returns(t) for t in trajectories]
    
    return {
        'n_trajectories': len(trajectories),
        'mean_length': np.mean(lengths),
        'std_length': np.std(lengths),
        'min_length': np.min(lengths),
        'max_length': np.max(lengths),
        'mean_return': np.mean(returns),
        'std_return': np.std(returns)
    }


def print_trajectory_stats(trajectories: List[List[dict]], name: str = ""):
    """Print trajectory statistics."""
    stats = trajectory_statistics(trajectories)
    
    if name:
        print(f"\n{name}:")
    print(f"  N trajectories: {stats['n_trajectories']}")
    print(f"  Length: {stats['mean_length']:.1f} ± {stats['std_length']:.1f} "
          f"(min={stats['min_length']}, max={stats['max_length']})")
    print(f"  Return: {stats['mean_return']:.2f} ± {stats['std_return']:.2f}")
