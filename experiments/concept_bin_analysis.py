"""
Concept Bin Analysis — ICML-ready figures
No titles inside figures — captions go in LaTeX only.
Consistent style across all three figures.

Run: python3 experiments/concept_bin_analysis.py
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

concept_names = ['near_goal', 'high_wind', 'in_left_half',
                 'in_top_half', 'near_start']


def vec_to_bin(vec):
    return int(sum((1 if vec[i] > 0.5 else 0) * (2**i)
                   for i in range(5)))


def bin_label(idx):
    active = [concept_names[i][:6]
              for i in range(5) if (idx >> i) & 1]
    return '+'.join(active) if active else 'none'


# =============================================================================
# COLLECT DATA
# =============================================================================

def collect_all_data(n_traj=500, seed=42, train_horizon=10):
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

    hard_early = np.zeros(32)
    hard_late  = np.zeros(32)
    soft_early = np.zeros(32)
    soft_late  = np.zeros(32)
    agree_early = []
    agree_late  = []
    agree_by_t  = {t: [] for t in range(0, 43, 2)}

    for traj in trajs:
        for t, step in enumerate(traj):
            s     = step['state']
            h_bin = vec_to_bin(hard.extract(s))
            s_bin = vec_to_bin(soft(s))
            agree = (h_bin == s_bin)

            if t < train_horizon:
                hard_early[h_bin] += 1
                soft_early[s_bin] += 1
                agree_early.append(agree)
            else:
                hard_late[h_bin] += 1
                soft_late[s_bin] += 1
                agree_late.append(agree)

            for key in agree_by_t:
                if t == key:
                    agree_by_t[key].append(agree)

    ne = hard_early.sum()
    nl = hard_late.sum()

    print(f"\nSteps: early={int(ne)}, late={int(nl)}")
    print(f"Hard bins early: {np.sum(hard_early>0)}/32  "
          f"late: {np.sum(hard_late>0)}/32")
    print(f"Soft bins early: {np.sum(soft_early>0)}/32  "
          f"late: {np.sum(soft_late>0)}/32")
    print(f"Agreement early: {np.mean(agree_early)*100:.1f}%  "
          f"late: {np.mean(agree_late)*100:.1f}%")

    return dict(hard_early=hard_early, hard_late=hard_late,
                soft_early=soft_early, soft_late=soft_late,
                agree_early=np.mean(agree_early) * 100,
                agree_late=np.mean(agree_late) * 100,
                agree_by_t=agree_by_t,
                ne=ne, nl=nl, train_horizon=train_horizon)


# =============================================================================
# FIGURE 1 — EARLY vs LATE BIN DISTRIBUTIONS (4 subplots)
# No figure title — caption goes in LaTeX
# Short subplot titles kept for readability
# =============================================================================

def plot_early_vs_late(d, save_path=None):
    plt.rcParams.update(STYLE)
    fig, axes = plt.subplots(
        2, 2, figsize=(14, 8),
        gridspec_kw={'hspace': 0.45, 'wspace': 0.28})
    th = d['train_horizon']

    panels = [
        (d['hard_early'], d['ne'], C_HARD,
         f"Hard concepts   $t < {th}$"
         f"   ({int(np.sum(d['hard_early']>0))}/32 bins active)",
         axes[0, 0]),
        (d['hard_late'],  d['nl'], C_HARD,
         f"Hard concepts   $t \\geq {th}$"
         f"   ({int(np.sum(d['hard_late']>0))}/32 bins active)",
         axes[0, 1]),
        (d['soft_early'], d['ne'], C_SOFT,
         f"Soft concepts   $t < {th}$"
         f"   (agreement: {d['agree_early']:.1f}%)",
         axes[1, 0]),
        (d['soft_late'],  d['nl'], C_SOFT,
         f"Soft concepts   $t \\geq {th}$"
         f"   (agreement: {d['agree_late']:.1f}%)",
         axes[1, 1]),
    ]

    for counts, total, color, subtitle, ax in panels:
        pcts  = counts / max(total, 1) * 100
        bcols = [color if c > 0 else '#e0e0e0' for c in counts]
        bars  = ax.bar(range(32), pcts, color=bcols,
                       alpha=0.85, edgecolor='white',
                       linewidth=0.4)

        for bar, p in zip(bars, pcts):
            if p > 1.5:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        p + 0.15, f'{p:.1f}',
                        ha='center', va='bottom',
                        fontsize=6.5, color='#333333',
                        rotation=90)

        # Short subtitle — not a figure title
        ax.set_title(subtitle, fontsize=9.5, pad=5,
                     fontweight='normal')
        ax.set_xlabel('Concept bin index', fontsize=10)
        ax.set_ylabel('Frequency (%)', fontsize=10)
        ax.set_xticks(range(0, 32, 4))
        ax.set_xticklabels(range(0, 32, 4), fontsize=9)

    # Annotate flooded bins in soft-OOD subplot only
    ax  = axes[1, 1]
    sl  = d['soft_late']
    hl  = d['hard_late']
    nl  = d['nl']
    for i in range(32):
        diff = (sl[i] - hl[i]) / max(nl, 1) * 100
        if diff > 8:
            ax.annotate('flood',
                        xy=(i, sl[i] / max(nl, 1) * 100),
                        xytext=(0, 5),
                        textcoords='offset points',
                        ha='center', fontsize=6,
                        color='#cc0000', fontweight='bold')

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    return fig


# =============================================================================
# FIGURE 2 — BIN AGREEMENT OVER TIME
# No title inside figure — caption goes in LaTeX
# Axis labels + legend + annotations are sufficient
# =============================================================================

def plot_agreement_over_time(d, save_path=None):
    plt.rcParams.update(STYLE)
    abt  = d['agree_by_t']
    ts   = [t for t in sorted(abt) if len(abt[t]) > 10]
    vals = [np.mean(abt[t]) * 100 for t in ts]
    th   = d['train_horizon']

    fig, ax = plt.subplots(figsize=(8, 4.5))

    # Shading
    ax.axvspan(-0.5, th,
               color=C_IN, alpha=0.8, zorder=0)
    ax.axvspan(th, max(ts) + 0.5,
               color=C_OOD, alpha=0.8, zorder=0)

    # Reference lines
    ax.axvline(x=th, color=C_VLINE, ls='--', lw=1.5,
               label=f'Train horizon ($t={th}$)', zorder=2)
    ax.axhline(y=d['agree_early'], color=C_HARD, ls=':',
               lw=1.4, alpha=0.7,
               label=f"In-distribution: {d['agree_early']:.1f}%")

    # Main line
    ax.plot(ts, vals, color='#2980b9', marker='o',
            linewidth=2.0, markersize=5, zorder=3,
            label='Bin agreement')
    ax.fill_between(ts, vals, alpha=0.10, color='#2980b9')

    # Annotate every other point
    for i, (t, v) in enumerate(zip(ts, vals)):
        if i % 2 == 0:
            ax.annotate(f'{v:.0f}%',
                        xy=(t, v), xytext=(0, 8),
                        textcoords='offset points',
                        ha='center', fontsize=8,
                        color='#1a5a8a', fontweight='bold')

    # Mark where agreement drops below 50%
    drop_t = next((t for t, v in zip(ts, vals)
                   if v < 50), None)
    if drop_t:
        ax.axvline(x=drop_t, color=C_SOFT, ls=':', lw=1.4,
                   label=f'Below 50% at $t={drop_t}$')

    # Region labels
    ax.text(th / 2, 107, 'In-distribution',
            ha='center', fontsize=9,
            color='#1a6b35', style='italic')
    ax.text((th + max(ts)) / 2, 107, 'Out-of-distribution',
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
# FIGURE 3 — 32-CELL BIN GRID
# No title inside figure — caption goes in LaTeX
# Colour legend added at bottom
# =============================================================================

def plot_bin_grid(d, save_path=None):
    plt.rcParams.update(STYLE)
    he = d['hard_early']
    hl = d['hard_late']
    sl = d['soft_late']
    ne = d['ne']
    nl = d['nl']

    fig = plt.figure(figsize=(18, 9))
    gs  = gridspec.GridSpec(4, 8, figure=fig,
                            hspace=0.50, wspace=0.22,
                            top=0.92, bottom=0.06,
                            left=0.02, right=0.98)

    for idx in range(32):
        row = idx // 8
        col = idx % 8
        ax  = fig.add_subplot(gs[row, col])
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')

        ep  = he[idx] / max(ne, 1) * 100
        lp  = hl[idx] / max(nl, 1) * 100
        slp = sl[idx] / max(nl, 1) * 100
        act = [concept_names[i]
               for i in range(5) if (idx >> i) & 1]

        # Background colour by frequency
        if ep == 0 and lp == 0:
            bg = '#f0f0f0'; tc = '#aaaaaa'
            status = 'impossible'; sc = '#aaaaaa'
        elif ep > 10 or lp > 10:
            bg = '#d5f5e3'; tc = '#1a7a1a'
            status = 'dominant'; sc = '#1a7a1a'
        elif ep > 2 or lp > 2:
            bg = '#d6eaf8'; tc = '#1a4a8a'
            status = 'common'; sc = '#1a4a8a'
        else:
            bg = '#fef9e7'; tc = '#7a6a00'
            status = 'rare'; sc = '#7a6a00'

        ax.add_patch(plt.Rectangle(
            (0.02, 0.02), 0.96, 0.96,
            facecolor=bg, edgecolor='#cccccc',
            linewidth=0.8))

        # Bin number
        ax.text(0.5, 0.93, f'Bin {idx}',
                ha='center', va='top',
                fontsize=7.5, fontweight='bold',
                color='#333333',
                transform=ax.transAxes)

        # Early / late frequency
        ax.text(0.5, 0.76,
                f'E: {ep:.1f}%   L: {lp:.1f}%',
                ha='center', va='center',
                fontsize=6.5, color=tc,
                fontweight='bold',
                transform=ax.transAxes)

        # Soft mismatch at OOD
        diff = slp - lp
        if abs(diff) > 3:
            mc = '#cc0000' if diff > 0 else '#0055cc'
            ax.text(0.5, 0.58,
                    f'Soft: {slp:.1f}% ({diff:+.1f})',
                    ha='center', va='center',
                    fontsize=5.8, color=mc,
                    fontweight='bold',
                    transform=ax.transAxes)

        # Active concept names
        c_str = ('\n'.join([n[:8] for n in act])
                 if act else 'none')
        ax.text(0.5, 0.36, c_str,
                ha='center', va='center',
                fontsize=5.5, color='#444444',
                transform=ax.transAxes)

        # Status label
        ax.text(0.5, 0.06, status,
                ha='center', va='bottom',
                fontsize=6, style='italic',
                color=sc, transform=ax.transAxes)

    # Colour legend at bottom — no title at top
    fig.text(
        0.5, 0.01,
        'Cell colour:   '
        'Green = dominant bin   '
        'Blue = common bin   '
        'Yellow = rare bin   '
        'Grey = impossible bin   |   '
        'E = early in-distribution   '
        'L = late OOD   '
        'Red = soft floods bin at OOD',
        ha='center', fontsize=8, color='#444444')

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    return fig


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    os.makedirs(results_dir, exist_ok=True)
    data = collect_all_data(n_traj=500, seed=42,
                            train_horizon=10)

    print("\nGenerating figures...")
    plot_early_vs_late(
        data,
        save_path=os.path.join(
            results_dir, 'fig_bin_early_vs_late.png'))
    plot_agreement_over_time(
        data,
        save_path=os.path.join(
            results_dir, 'fig_bin_agreement_over_time.png'))
    plot_bin_grid(
        data,
        save_path=os.path.join(
            results_dir, 'fig_bin_grid.png'))

    plt.show()
    print("\nDone.")