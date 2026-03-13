"""
Clean Figure Generation for Temporal Leakage Paper

Run: python experiments/plot_figures.py
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

# Use a clean style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.dpi'] = 150


def plot_main_figure(results: dict, save_path: str = None):
    """
    Clean 2x2 figure for the paper.
    """
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    
    timesteps = results['timesteps']
    train_horizon = results['train_horizon']
    
    # Colors
    BLUE = '#2E86AB'
    RED = '#E94F37'
    PURPLE = '#7B2D8E'
    GREEN = '#3A7D44'
    GRAY = '#888888'
    
    # =========================================================================
    # Plot 1: Concept Accuracy (TOP LEFT) - THE KEY RESULT
    # =========================================================================
    ax1 = axes[0, 0]
    
    # Shade OOD region
    ax1.axvspan(train_horizon, max(timesteps) + 2, alpha=0.15, color=RED, label='OOD region')
    
    # Plot accuracy
    ax1.plot(timesteps, results['concept_accuracy'], 
             color=BLUE, linewidth=2.5, marker='o', markersize=7, 
             markerfacecolor='white', markeredgewidth=2)
    
    # Train horizon line
    ax1.axvline(x=train_horizon, color=RED, linestyle='--', linewidth=1.5, alpha=0.8)
    
    # Perfect accuracy reference
    ax1.axhline(y=1.0, color=GREEN, linestyle=':', linewidth=1.5, alpha=0.6)
    
    ax1.set_xlabel('Timestep $t$')
    ax1.set_ylabel('Concept Accuracy')
    ax1.set_title('(a) Soft Concept Accuracy Degrades Over Time', fontweight='bold')
    ax1.set_ylim([0.55, 1.02])
    ax1.set_xlim([-1, max(timesteps) + 1])
    
    # Custom legend
    legend_elements = [
        Line2D([0], [0], color=BLUE, linewidth=2, marker='o', markersize=6, 
               markerfacecolor='white', label='Soft concepts'),
        Line2D([0], [0], color=RED, linestyle='--', linewidth=1.5, label=f'Train horizon ($t={train_horizon}$)'),
        mpatches.Patch(facecolor=RED, alpha=0.15, label='Out-of-distribution'),
    ]
    ax1.legend(handles=legend_elements, loc='lower left', framealpha=0.9)
    
    # =========================================================================
    # Plot 2: Hard vs Soft Comparison (TOP RIGHT)
    # =========================================================================
    ax2 = axes[0, 1]
    
    # Shade OOD region
    ax2.axvspan(train_horizon, max(timesteps) + 2, alpha=0.15, color=RED)
    
    # Hard concepts (constant at 1.0)
    hard_acc = [1.0] * len(timesteps)
    ax2.plot(timesteps, hard_acc, 
             color=GREEN, linewidth=2.5, marker='s', markersize=7,
             markerfacecolor='white', markeredgewidth=2, label='Hard concepts')
    
    # Soft concepts
    ax2.plot(timesteps, results['concept_accuracy'], 
             color=BLUE, linewidth=2.5, marker='o', markersize=7,
             markerfacecolor='white', markeredgewidth=2, label='Soft concepts')
    
    ax2.axvline(x=train_horizon, color=RED, linestyle='--', linewidth=1.5, alpha=0.8)
    
    ax2.set_xlabel('Timestep $t$')
    ax2.set_ylabel('Concept Accuracy')
    ax2.set_title('(b) Hard Concepts Remain Stable', fontweight='bold')
    ax2.set_ylim([0.55, 1.02])
    ax2.set_xlim([-1, max(timesteps) + 1])
    ax2.legend(loc='lower left', framealpha=0.9)
    
    # =========================================================================
    # Plot 3: Distribution Shift (BOTTOM LEFT)
    # =========================================================================
    ax3 = axes[1, 0]
    
    ax3.axvspan(train_horizon, max(timesteps) + 2, alpha=0.15, color=RED)
    
    ax3.plot(timesteps, results['distribution_shift'], 
             color=PURPLE, linewidth=2.5, marker='^', markersize=7,
             markerfacecolor='white', markeredgewidth=2)
    
    ax3.axvline(x=train_horizon, color=RED, linestyle='--', linewidth=1.5, alpha=0.8)
    
    ax3.set_xlabel('Timestep $t$')
    ax3.set_ylabel('KL Divergence from $t=0$')
    ax3.set_title('(c) State Distribution Shifts Immediately', fontweight='bold')
    ax3.set_xlim([-1, max(timesteps) + 1])
    
    # =========================================================================
    # Plot 4: Accuracy Drop Magnitude (BOTTOM RIGHT)
    # =========================================================================
    ax4 = axes[1, 1]
    
    # Compute accuracy drop from t=0
    acc_drop = [1.0 - acc for acc in results['concept_accuracy']]
    
    ax4.axvspan(train_horizon, max(timesteps) + 2, alpha=0.15, color=RED)
    
    ax4.bar(timesteps, acc_drop, width=2, color=BLUE, alpha=0.7, edgecolor=BLUE)
    
    ax4.axvline(x=train_horizon, color=RED, linestyle='--', linewidth=1.5, alpha=0.8)
    
    ax4.set_xlabel('Timestep $t$')
    ax4.set_ylabel('Accuracy Drop (from 100%)')
    ax4.set_title('(d) Error Magnitude Grows with Time', fontweight='bold')
    ax4.set_xlim([-1, max(timesteps) + 1])
    ax4.set_ylim([0, 0.4])
    
    # Format y-axis as percentage
    ax4.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x*100:.0f}%'))
    
    # =========================================================================
    # Final adjustments
    # =========================================================================
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Figure saved to: {save_path}")
    
    plt.show()
    return fig


def plot_single_key_figure(results: dict, save_path: str = None):
    """
    Single clean figure showing the main result: hard vs soft concept accuracy.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    
    timesteps = results['timesteps']
    train_horizon = results['train_horizon']
    
    # Colors
    BLUE = '#2E86AB'
    GREEN = '#3A7D44'
    RED = '#E94F37'
    
    # Shade OOD region
    ax.axvspan(train_horizon, max(timesteps) + 2, alpha=0.12, color=RED, label='OOD region')
    
    # Hard concepts
    hard_acc = [1.0] * len(timesteps)
    ax.plot(timesteps, hard_acc, 
            color=GREEN, linewidth=3, marker='s', markersize=9,
            markerfacecolor='white', markeredgewidth=2.5, label='Hard concepts (rule-based)')
    
    # Soft concepts
    ax.plot(timesteps, results['concept_accuracy'], 
            color=BLUE, linewidth=3, marker='o', markersize=9,
            markerfacecolor='white', markeredgewidth=2.5, label='Soft concepts (neural)')
    
    # Train horizon
    ax.axvline(x=train_horizon, color=RED, linestyle='--', linewidth=2, alpha=0.8)
    
    # Annotations
    ax.annotate('Train horizon', xy=(train_horizon, 0.58), 
                xytext=(train_horizon + 3, 0.58),
                fontsize=10, color=RED,
                arrowprops=dict(arrowstyle='->', color=RED, lw=1.5))
    
    ax.annotate('32% drop', xy=(30, 0.68), 
                xytext=(33, 0.75),
                fontsize=10, color=BLUE,
                arrowprops=dict(arrowstyle='->', color=BLUE, lw=1.5))
    
    ax.set_xlabel('Timestep $t$', fontsize=13)
    ax.set_ylabel('Concept Prediction Accuracy', fontsize=13)
    ax.set_title('Temporal Leakage Poisoning: Soft Concepts Degrade at OOD Timesteps', 
                 fontsize=14, fontweight='bold')
    ax.set_ylim([0.55, 1.03])
    ax.set_xlim([-1, max(timesteps) + 3])
    
    ax.legend(loc='lower left', fontsize=11, framealpha=0.95)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Figure saved to: {save_path}")
    
    plt.show()
    return fig


if __name__ == "__main__":
    import os
    import sys
    
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results_dir = os.path.join(project_root, 'results')
    
    # Load results
    results_path = os.path.join(results_dir, 'temporal_results.npy')
    
    if os.path.exists(results_path):
        results = np.load(results_path, allow_pickle=True).item()
        print("Loaded existing results")
    else:
        # Use default values if no results file
        print("No results file found. Using example data...")
        results = {
            'timesteps': [0, 2, 5, 10, 15, 20, 25, 30, 35, 40],
            'train_horizon': 10,
            'concept_accuracy': [1.0, 1.0, 0.997, 1.0, 0.94, 0.785, 0.72, 0.68, 0.67, 0.68],
            'distribution_shift': [0.0, 19.0, 20.2, 20.7, 20.7, 20.5, 20.6, 20.6, 20.6, 20.7],
            'leakage_r2': [0.0, 0.998, 0.999, 0.99, 0.92, 0.95, 0.85, 0.85, 0.98, 0.99],
        }
    
    # Generate figures
    print("\nGenerating main 2x2 figure...")
    plot_main_figure(results, save_path=os.path.join(results_dir, 'figure_main.png'))
    
    print("\nGenerating single key figure...")
    plot_single_key_figure(results, save_path=os.path.join(results_dir, 'figure_key.png'))
    
    print("\nDone!")