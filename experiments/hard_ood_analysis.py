"""
Hard OOD Analysis — ICML-ready figures
No titles inside figures — captions go in LaTeX only.
Consistent style across all three figures.

Run: python3 experiments/hard_ood_analysis.py
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os, sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(project_root, 'src'))
results_dir = os.path.join(project_root, 'results')

from gridworld import WindyGridworld, collect_trajectory
from policies import EpsilonGreedyPolicy
from concepts import HardConcepts, SoftConcepts
from utils import set_seed

# ── Consistent style for all figures ─────────────────────────────────────────
STYLE = {
    'font.family':       'DejaVu Sans',
    'font.size':         11,
    'axes.labelsize':    10,
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'axes.linewidth':    1.1,
    'axes.grid':         True,
    'grid.alpha':        0.22,
    'grid.linestyle':    '--',
    'lines.linewidth':   2.0,
    'lines.markersize':  6,
    'figure.dpi':        150,
    'legend.fontsize':   9,
    'legend.framealpha': 0.92,
    'xtick.labelsize':   9,
    'ytick.labelsize':   9,
}

C_HARD  = '#27ae60'
C_SOFT  = '#e74c3c'
C_VLINE = '#7f8c8d'
C_IN    = '#e8f5e9'
C_OOD   = '#fce4e4'

CONCEPT_COLORS = ['#e74c3c', '#2980b9', '#27ae60',
                  '#f39c12', '#8e44ad']

concept_names = ['near_goal', 'high_wind', 'in_left_half',
                 'in_top_half', 'near_start']


def vec_to_bin(vec):
    return int(sum((1 if vec[i] > 0.5 else 0) * (2**i)
                   for i in range(5)))


# =============================================================================
# COLLECT AND ANALYSE
# =============================================================================

def analyse_hard_ood(n_traj=500, seed=42, train_horizon=10):
    set_seed(seed)
    env  = WindyGridworld()
    bp   = EpsilonGreedyPolicy(env, epsilon=0.4, seed=seed)
    hard = HardConcepts(env)

    print("Collecting trajectories...")
    trajs = [collect_trajectory(env, bp, max_steps=50)
             for _ in range(n_traj)]

    soft = SoftConcepts(env, use_leakage=True, seed=seed)
    train_trajs = []
    for traj in trajs[:200]:
        early = [s for i, s in enumerate(traj)
                 if i < train_horizon]
        if early:
            train_trajs.append(early)
    print(f"Training soft concepts on t < {train_horizon}...")
    soft.train_on_trajectories(train_trajs, hard, epochs=200)

    # Find which hard bins appear in training
    training_bins = set()
    for traj in trajs[:200]:
        for t, step in enumerate(traj):
            if t < train_horizon:
                training_bins.add(
                    vec_to_bin(hard.extract(step['state'])))

    print(f"\nHard bins seen in training (t < {train_horizon}): "
          f"{sorted(training_bins)}")
    print(f"Count: {len(training_bins)}/32")

    # Per-concept: which values seen in training
    concept_vals_in_training = {i: set() for i in range(5)}
    for traj in trajs[:200]:
        for t, step in enumerate(traj):
            if t < train_horizon:
                h_vec = hard.extract(step['state'])
                for ci in range(5):
                    concept_vals_in_training[ci].add(
                        int(h_vec[ci] > 0.5))

    print("\nConcept values seen in training:")
    for ci, name in enumerate(concept_names):
        vals = concept_vals_in_training[ci]
        print(f"  {name}: {sorted(vals)} "
              f"{'BOTH' if len(vals)==2 else 'ONLY ONE'}")

    # Main analysis per timestep
    results_by_t = {}
    ts = list(range(0, 43, 2))

    for t_target in ts:
        hard_indist_agree = []
        hard_ood_agree    = []
        per_concept_correct = {ci: {0: [], 1: []}
                                for ci in range(5)}
        per_concept_ood     = {ci: {True: [], False: []}
                                for ci in range(5)}

        for traj in trajs:
            for t, step in enumerate(traj):
                if t != t_target:
                    continue
                s     = step['state']
                h_vec = hard.extract(s)
                s_vec = soft(s)
                h_bin = vec_to_bin(h_vec)
                s_bin = vec_to_bin(s_vec)
                agree = (h_bin == s_bin)

                hard_is_ood = h_bin not in training_bins
                if hard_is_ood:
                    hard_ood_agree.append(agree)
                else:
                    hard_indist_agree.append(agree)

                for ci in range(5):
                    h_val   = int(h_vec[ci] > 0.5)
                    s_val   = int(s_vec[ci] > 0.5)
                    correct = (h_val == s_val)
                    per_concept_correct[ci][h_val].append(correct)
                    seen = h_val in concept_vals_in_training[ci]
                    per_concept_ood[ci][seen].append(correct)

        results_by_t[t_target] = {
            'hard_indist_agree': (
                np.mean(hard_indist_agree) * 100
                if hard_indist_agree else np.nan),
            'hard_ood_agree': (
                np.mean(hard_ood_agree) * 100
                if hard_ood_agree else np.nan),
            'n_indist': len(hard_indist_agree),
            'n_ood':    len(hard_ood_agree),
            'per_concept_correct': {
                ci: {
                    v: (np.mean(per_concept_correct[ci][v]) * 100
                        if per_concept_correct[ci][v] else np.nan)
                    for v in [0, 1]
                }
                for ci in range(5)
            },
            'per_concept_ood': {
                ci: {
                    seen: (np.mean(per_concept_ood[ci][seen]) * 100
                           if per_concept_ood[ci][seen] else np.nan)
                    for seen in [True, False]
                }
                for ci in range(5)
            },
        }

    return results_by_t, training_bins


# =============================================================================
# FIGURE 1 — HARD OOD vs IN-DIST AGREEMENT
# No title — caption goes in LaTeX
# Axis labels + legend + region labels sufficient
# =============================================================================

def plot_hard_ood_agreement(results_by_t, train_horizon=10,
                            save_path=None):
    plt.rcParams.update(STYLE)
    ts          = sorted(results_by_t.keys())
    indist_vals = [results_by_t[t]['hard_indist_agree']
                   for t in ts]
    ood_vals    = [results_by_t[t]['hard_ood_agree']
                   for t in ts]

    fig, ax = plt.subplots(figsize=(8, 4.5))

    # Shading
    ax.axvspan(-0.5, train_horizon,
               color=C_IN, alpha=0.8, zorder=0)
    ax.axvspan(train_horizon, max(ts) + 0.5,
               color=C_OOD, alpha=0.8, zorder=0)
    ax.axvline(x=train_horizon, color=C_VLINE,
               ls='--', lw=1.5,
               label=f'Train horizon ($t={train_horizon}$)',
               zorder=2)
    ax.axhline(y=50, color='#888888', ls=':',
               lw=1.0, alpha=0.5,
               label='50% random baseline')

    # Filter NaN
    ts_id  = [t for t, v in zip(ts, indist_vals)
               if not np.isnan(v)]
    val_id = [v for v in indist_vals
               if not np.isnan(v)]
    ts_od  = [t for t, v in zip(ts, ood_vals)
               if not np.isnan(v)]
    val_od = [v for v in ood_vals
               if not np.isnan(v)]

    ax.plot(ts_id, val_id, color=C_HARD, marker='s',
            linewidth=2.0, markersize=6, zorder=3,
            label='Known bin (seen in training)')
    ax.plot(ts_od, val_od, color=C_SOFT, marker='^',
            linewidth=2.0, markersize=6, zorder=3,
            label='Unknown bin (not seen in training)')

    # Annotate every other point
    for t, v in zip(ts_id, val_id):
        if t % 4 == 0:
            ax.annotate(f'{v:.0f}%',
                        xy=(t, v), xytext=(0, 8),
                        textcoords='offset points',
                        ha='center', fontsize=8,
                        color='#1a7a1a', fontweight='bold')
    for t, v in zip(ts_od, val_od):
        if t % 4 == 0:
            ax.annotate(f'{v:.0f}%',
                        xy=(t, v), xytext=(0, -14),
                        textcoords='offset points',
                        ha='center', fontsize=8,
                        color='#cc0000', fontweight='bold')

    # Region labels
    ax.text(train_horizon / 2, 107, 'In-distribution',
            ha='center', fontsize=9,
            color='#1a6b35', style='italic')
    ax.text((train_horizon + max(ts)) / 2, 107,
            'Out-of-distribution',
            ha='center', fontsize=9,
            color='#a93226', style='italic')

    ax.set_xlabel('Trajectory timestep $t$', fontsize=10)
    ax.set_ylabel('Soft-hard bin agreement (%)', fontsize=10)
    ax.set_ylim(0, 115)
    ax.set_xlim(-0.5, max(ts) + 0.5)
    ax.legend(fontsize=9, loc='lower left')

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    return fig


# =============================================================================
# FIGURE 2 — PER-CONCEPT ACCURACY: SEEN vs UNSEEN VALUE
# No figure title — caption goes in LaTeX
# Short concept name as subplot title is fine
# =============================================================================

def plot_per_concept_seen_vs_unseen(results_by_t,
                                    train_horizon=10,
                                    save_path=None):
    plt.rcParams.update(STYLE)
    fig, axes = plt.subplots(
        1, 5, figsize=(20, 5),
        gridspec_kw={'wspace': 0.32})

    ts = sorted(results_by_t.keys())

    for ci, (name, color) in enumerate(
            zip(concept_names, CONCEPT_COLORS)):
        ax = axes[ci]

        seen_vals    = [results_by_t[t]['per_concept_ood'][ci][True]
                        for t in ts]
        notseen_vals = [results_by_t[t]['per_concept_ood'][ci][False]
                        for t in ts]

        # Filter NaN
        ts_s   = [t for t, v in zip(ts, seen_vals)
                   if not np.isnan(v)]
        val_s  = [v for v in seen_vals
                   if not np.isnan(v)]
        ts_ns  = [t for t, v in zip(ts, notseen_vals)
                   if not np.isnan(v)]
        val_ns = [v for v in notseen_vals
                   if not np.isnan(v)]

        # Shading
        ax.axvspan(-0.5, train_horizon,
                   color=C_IN, alpha=0.7, zorder=0)
        ax.axvspan(train_horizon, max(ts) + 0.5,
                   color=C_OOD, alpha=0.7, zorder=0)
        ax.axvline(x=train_horizon, color=C_VLINE,
                   ls='--', lw=1.3, zorder=2)
        ax.axhline(y=50, color='#888888',
                   ls=':', lw=1.0, alpha=0.5)

        if ts_s:
            ax.plot(ts_s, val_s, color=color,
                    marker='o', linewidth=2.0,
                    markersize=5, zorder=3,
                    label='Seen in training')
        if ts_ns:
            ax.plot(ts_ns, val_ns, color=color,
                    marker='^', linewidth=2.0,
                    markersize=5, linestyle='--',
                    zorder=3, alpha=0.75,
                    label='Unseen in training')

        # Short concept name as subplot label — not a title
        ax.set_title(name.replace('_', '\n'),
                     fontsize=10, fontweight='normal',
                     color=color, pad=4)
        ax.set_xlabel('Trajectory timestep $t$',
                      fontsize=10)
        ax.set_ylabel('Soft accuracy (%)'
                      if ci == 0 else '', fontsize=10)
        ax.set_ylim(0, 115)
        ax.set_xlim(-0.5, max(ts) + 0.5)
        ax.set_xticks(range(0, max(ts) + 1, 10))
        ax.legend(fontsize=8, loc='lower left')

        # Annotate final values
        if ts_s and val_s:
            ax.annotate(f'{val_s[-1]:.0f}%',
                        xy=(ts_s[-1], val_s[-1]),
                        xytext=(0, 7),
                        textcoords='offset points',
                        ha='center', fontsize=8,
                        color=color, fontweight='bold')
        if ts_ns and val_ns:
            ax.annotate(f'{val_ns[-1]:.0f}%',
                        xy=(ts_ns[-1], val_ns[-1]),
                        xytext=(0, -13),
                        textcoords='offset points',
                        ha='center', fontsize=8,
                        color=color, fontweight='bold')

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    return fig


# =============================================================================
# FIGURE 3 — HEATMAP: CONCEPT ACCURACY vs TIMESTEP
# No title — caption goes in LaTeX
# Colorbar + axis labels + dashed train line sufficient
# =============================================================================

def plot_concept_accuracy_heatmap(results_by_t,
                                   train_horizon=10,
                                   save_path=None):
    plt.rcParams.update(STYLE)
    ts   = sorted(results_by_t.keys())
    data = np.zeros((5, len(ts)))

    for ti, t in enumerate(ts):
        for ci in range(5):
            c0 = results_by_t[t]['per_concept_correct'][ci][0]
            c1 = results_by_t[t]['per_concept_correct'][ci][1]
            all_vals = []
            if not np.isnan(c0):
                all_vals.append(c0)
            if not np.isnan(c1):
                all_vals.append(c1)
            data[ci, ti] = (np.mean(all_vals)
                            if all_vals else np.nan)

    fig, ax = plt.subplots(figsize=(13, 4.5))

    masked = np.ma.masked_invalid(data)
    im = ax.imshow(masked, aspect='auto', cmap='RdYlGn',
                   vmin=0, vmax=100,
                   interpolation='nearest')

    # Value text in each cell
    for ci in range(5):
        for ti, t in enumerate(ts):
            val = data[ci, ti]
            if not np.isnan(val):
                tc = ('white'
                      if val < 40 or val > 85
                      else 'black')
                ax.text(ti, ci, f'{val:.0f}',
                        ha='center', va='center',
                        fontsize=8.5, color=tc,
                        fontweight='bold')

    # Train horizon dashed line
    th_idx = (ts.index(train_horizon)
               if train_horizon in ts
               else next(i for i, t in enumerate(ts)
                         if t >= train_horizon))
    ax.axvline(x=th_idx - 0.5,
               color='white', linewidth=2.5)
    ax.axvline(x=th_idx - 0.5,
               color='#333333', linewidth=1.5,
               linestyle='--',
               label=f'Train horizon ($t={train_horizon}$)')
    ax.text(th_idx - 0.5, -0.8,
            f'$t={train_horizon}$',
            ha='center', fontsize=9,
            color='#333333', fontweight='bold')

    ax.set_yticks(range(5))
    ax.set_yticklabels(
        [n.replace('_', ' ') for n in concept_names],
        fontsize=10)
    ax.set_xticks(range(len(ts)))
    ax.set_xticklabels(
        [f'$t={t}$' for t in ts],
        rotation=45, ha='right', fontsize=8.5)
    ax.set_xlabel('Trajectory timestep $t$', fontsize=10)
    ax.set_ylabel('Concept', fontsize=10)

    cbar = fig.colorbar(im, ax=ax,
                        fraction=0.025, pad=0.02)
    cbar.set_label('Soft accuracy (%)', fontsize=10)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    return fig


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    os.makedirs(results_dir, exist_ok=True)

    results_by_t, training_bins = analyse_hard_ood(
        n_traj=500, seed=42, train_horizon=10)

    print("\nGenerating figures...")

    plot_hard_ood_agreement(
        results_by_t, train_horizon=10,
        save_path=os.path.join(
            results_dir, 'fig_hard_ood_agreement.png'))

    plot_per_concept_seen_vs_unseen(
        results_by_t, train_horizon=10,
        save_path=os.path.join(
            results_dir, 'fig_per_concept_seen_vs_unseen.png'))

    plot_concept_accuracy_heatmap(
        results_by_t, train_horizon=10,
        save_path=os.path.join(
            results_dir, 'fig_concept_accuracy_heatmap.png'))

    plt.show()
    print("\nDone.")