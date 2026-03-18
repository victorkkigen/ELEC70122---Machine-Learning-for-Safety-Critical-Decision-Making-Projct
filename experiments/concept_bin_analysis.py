"""
Concept Bin Analysis — Fixed Version
Answers professor's question properly:
  1. Figure 1: Early (t<10) vs Late (t>=10) bin distributions
               for BOTH hard and soft concepts
  2. Figure 2: Bin agreement over time — shows WHEN soft goes wrong
  3. Figure 3: 32-cell grid with early/late frequency + soft mismatch

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

plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 11,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.grid': True, 'grid.alpha': 0.22, 'figure.dpi': 150,
})

concept_names = ['near_goal', 'high_wind', 'in_left_half',
                 'in_top_half', 'near_start']


def vec_to_bin(vec):
    return int(sum((1 if vec[i] > 0.5 else 0) * (2**i) for i in range(5)))


def bin_label(idx):
    active = [concept_names[i][:6] for i in range(5) if (idx >> i) & 1]
    return '+'.join(active) if active else 'none'


# =============================================================================
# COLLECT DATA
# =============================================================================

def collect_all_data(n_traj=500, seed=42, train_horizon=10):
    set_seed(seed)
    env   = WindyGridworld()
    bp    = EpsilonGreedyPolicy(env, epsilon=0.4, seed=seed)
    hard  = HardConcepts(env)

    print("Collecting trajectories...")
    trajs = [collect_trajectory(env, bp, max_steps=50) for _ in range(n_traj)]

    soft = SoftConcepts(env, use_leakage=True, seed=seed)
    train_trajs = []
    for traj in trajs[:200]:
        early = [s for i, s in enumerate(traj) if i < train_horizon]
        if early:
            train_trajs.append(early)
    print(f"Training soft concepts on t < {train_horizon}...")
    soft.train_on_trajectories(train_trajs, hard, epochs=200)

    hard_early = np.zeros(32); hard_late  = np.zeros(32)
    soft_early = np.zeros(32); soft_late  = np.zeros(32)
    agree_early = []; agree_late = []
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

    ne = hard_early.sum(); nl = hard_late.sum()

    print(f"\nSteps: early={int(ne)}, late={int(nl)}")
    print(f"Hard bins early: {np.sum(hard_early>0)}/32  "
          f"late: {np.sum(hard_late>0)}/32")
    print(f"Soft bins early: {np.sum(soft_early>0)}/32  "
          f"late: {np.sum(soft_late>0)}/32")
    print(f"Agreement early: {np.mean(agree_early)*100:.1f}%  "
          f"late: {np.mean(agree_late)*100:.1f}%")

    print(f"\n{'='*75}")
    print("HARD BIN SHIFTS: Early vs Late")
    print(f"{'Bin':<5} {'Concepts':<32} {'Early%':<9} {'Late%':<9} {'Change'}")
    print("-"*65)
    for i in np.argsort(-hard_early):
        ep = hard_early[i]/ne*100 if ne>0 else 0
        lp = hard_late[i] /nl*100 if nl>0 else 0
        if ep < 0.1 and lp < 0.1:
            continue
        d = lp - ep
        print(f"{i:<5} {bin_label(i):<32} {ep:<9.1f} {lp:<9.1f} "
              f"{'↑' if d>2 else ('↓' if d<-2 else '≈')}{d:+.1f}%")

    print(f"\n{'='*75}")
    print("SOFT BIN SHIFTS vs HARD at OOD (t >= train_horizon)")
    print(f"{'Bin':<5} {'Concepts':<32} {'Hard Late%':<12} "
          f"{'Soft Late%':<12} {'Mismatch'}")
    print("-"*75)
    visited = set(np.where(hard_late>0)[0]) | set(np.where(soft_late>0)[0])
    for i in sorted(visited, key=lambda x: -soft_late[x]):
        hl = hard_late[i]/nl*100 if nl>0 else 0
        sl = soft_late[i]/nl*100 if nl>0 else 0
        d  = sl - hl
        tag = "soft FLOODS ↑↑" if d>8 else (
              "soft avoids ↓↓" if d<-8 else "≈ match")
        print(f"{i:<5} {bin_label(i):<32} {hl:<12.1f} {sl:<12.1f} {tag}")

    return dict(hard_early=hard_early, hard_late=hard_late,
                soft_early=soft_early, soft_late=soft_late,
                agree_early=np.mean(agree_early)*100,
                agree_late=np.mean(agree_late)*100,
                agree_by_t=agree_by_t,
                ne=ne, nl=nl, train_horizon=train_horizon)


# =============================================================================
# FIGURE 1 — EARLY vs LATE: 4 SUBPLOTS
# =============================================================================

def plot_early_vs_late(d, save_path=None):
    fig, axes = plt.subplots(2, 2, figsize=(16, 10),
                             gridspec_kw={'hspace':0.45, 'wspace':0.28})
    th = d['train_horizon']

    panels = [
        (d['hard_early'], d['ne'], '#27ae60',
         f"Hard Concepts  In-Distribution (t < {th})\n"
         f"{int(np.sum(d['hard_early']>0))}/32 bins used  —  ground truth",
         axes[0,0]),
        (d['hard_late'],  d['nl'], '#27ae60',
         f"Hard Concepts  OOD (t ≥ {th})\n"
         f"{int(np.sum(d['hard_late']>0))}/32 bins used  —  ground truth",
         axes[0,1]),
        (d['soft_early'], d['ne'], '#e74c3c',
         f"Soft Concepts  In-Distribution (t < {th})\n"
         f"Agreement: {d['agree_early']:.1f}%  |  "
         f"{int(np.sum(d['soft_early']>0))}/32 bins used",
         axes[1,0]),
        (d['soft_late'],  d['nl'], '#e74c3c',
         f"Soft Concepts  OOD (t ≥ {th})\n"
         f"Agreement: {d['agree_late']:.1f}%  |  "
         f"{int(np.sum(d['soft_late']>0))}/32 bins used",
         axes[1,1]),
    ]

    for counts, total, color, title, ax in panels:
        pcts  = counts / max(total,1) * 100
        bcols = [color if c>0 else '#eeeeee' for c in counts]
        bars  = ax.bar(range(32), pcts, color=bcols,
                       alpha=0.82, edgecolor='white', linewidth=0.4)
        for i, (bar, p) in enumerate(zip(bars, pcts)):
            if p > 1.5:
                ax.text(bar.get_x()+bar.get_width()/2,
                        p+0.2, f'{p:.1f}',
                        ha='center', va='bottom',
                        fontsize=6.5, fontweight='bold',
                        color='#333333', rotation=90)
        ax.set_title(title, fontsize=10, pad=5)
        ax.set_xlabel('Concept Bin (0-31)', fontsize=10)
        ax.set_ylabel('Frequency (%)', fontsize=10)
        ax.set_xticks(range(0, 32, 2))
        ax.set_xticklabels(range(0, 32, 2), fontsize=8)

    # Highlight flooded bins in soft-OOD subplot
    ax    = axes[1,1]
    hl    = d['hard_late']
    sl    = d['soft_late']
    nl    = d['nl']
    for i in range(32):
        diff = (sl[i]-hl[i])/max(nl,1)*100
        if diff > 8:
            ax.annotate('↑flood',
                        xy=(i, sl[i]/max(nl,1)*100),
                        xytext=(0, 8), textcoords='offset points',
                        ha='center', fontsize=6.5,
                        color='#cc0000', fontweight='bold')

    fig.suptitle('Concept Bin Distributions: In-Distribution vs OOD\n'
                 'Does soft assign states to the same bins as hard?',
                 fontsize=13, fontweight='bold')

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    return fig


# =============================================================================
# FIGURE 2 — AGREEMENT OVER TIME
# =============================================================================

def plot_agreement_over_time(d, save_path=None):
    abt  = d['agree_by_t']
    ts   = [t for t in sorted(abt) if len(abt[t]) > 10]
    vals = [np.mean(abt[t])*100 for t in ts]
    th   = d['train_horizon']

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.axvspan(-0.5, th,         color='#e8f5e9', alpha=0.8, zorder=0)
    ax.axvspan(th,  max(ts)+0.5, color='#fce4e4', alpha=0.8, zorder=0)
    ax.axvline(x=th, color='#7f8c8d', ls='--', lw=1.5,
               label=f'Train horizon (t={th})', zorder=2)
    ax.axhline(y=d['agree_early'], color='#27ae60', ls=':',
               lw=1.5, alpha=0.7,
               label=f"In-dist agreement: {d['agree_early']:.1f}%")

    ax.plot(ts, vals, color='#2980b9', marker='o',
            linewidth=2.2, markersize=6, zorder=3)
    ax.fill_between(ts, vals, alpha=0.12, color='#2980b9')

    for i, (t, v) in enumerate(zip(ts, vals)):
        if i % 2 == 0:
            ax.annotate(f'{v:.0f}%',
                        xy=(t, v), xytext=(0, 9),
                        textcoords='offset points',
                        ha='center', fontsize=8,
                        color='#1a5a8a', fontweight='bold')

    drop_t = next((t for t, v in zip(ts, vals) if v < 50), None)
    if drop_t:
        ax.axvline(x=drop_t, color='#e74c3c', ls=':', lw=1.5,
                   label=f'Agreement < 50% at t={drop_t}')

    ax.set_xlabel('Trajectory Timestep t', fontsize=11)
    ax.set_ylabel('Soft-Hard Bin Agreement (%)', fontsize=11)
    ax.set_title('How Often Does Soft Assign States to the\n'
                 'CORRECT (Hard) Concept Bin?', fontsize=12)
    ax.set_ylim(0, 115)
    ax.set_xlim(-0.5, max(ts)+0.5)
    ax.legend(fontsize=9, loc='lower left')
    ax.text(th/2, 108, 'In-dist', ha='center',
            fontsize=9, color='#1a6b35', style='italic')
    ax.text((th+max(ts))/2, 108, 'Out-of-distribution',
            ha='center', fontsize=9, color='#a93226', style='italic')

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    return fig


# =============================================================================
# FIGURE 3 — 32-CELL GRID
# =============================================================================

def plot_bin_grid(d, save_path=None):
    he = d['hard_early']; hl = d['hard_late']
    sl = d['soft_late'];  ne = d['ne']; nl = d['nl']

    fig = plt.figure(figsize=(20, 10))
    gs  = gridspec.GridSpec(4, 8, figure=fig,
                            hspace=0.55, wspace=0.25,
                            top=0.88, bottom=0.04)

    for idx in range(32):
        row = idx // 8; col = idx % 8
        ax  = fig.add_subplot(gs[row, col])
        ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis('off')

        ep  = he[idx]/max(ne,1)*100
        lp  = hl[idx]/max(nl,1)*100
        slp = sl[idx]/max(nl,1)*100
        act = [concept_names[i] for i in range(5) if (idx>>i)&1]

        if ep==0 and lp==0:
            bg,tc,status,sc = '#f0f0f0','#aaaaaa','impossible','#aaaaaa'
        elif ep>10 or lp>10:
            bg,tc,status,sc = '#d5f5e3','#1a7a1a','dominant','#1a7a1a'
        elif ep>2  or lp>2:
            bg,tc,status,sc = '#d6eaf8','#1a4a8a','common','#1a4a8a'
        else:
            bg,tc,status,sc = '#fef9e7','#7a6a00','rare','#7a6a00'

        ax.add_patch(plt.Rectangle((0.02,0.02),0.96,0.96,
                                   facecolor=bg,edgecolor='#cccccc',lw=1.0))
        ax.text(0.5,0.93,f'Bin {idx}',ha='center',va='top',
                fontsize=8,fontweight='bold',color='#333333',
                transform=ax.transAxes)
        ax.text(0.5,0.78,f'E:{ep:.1f}% L:{lp:.1f}%',
                ha='center',va='center',fontsize=6.5,
                color=tc,fontweight='bold',transform=ax.transAxes)

        # Soft mismatch
        diff = slp - lp
        if abs(diff) > 3:
            mc = '#cc0000' if diff>0 else '#0055cc'
            ax.text(0.5,0.60,f'Soft:{slp:.1f}%({diff:+.1f})',
                    ha='center',va='center',fontsize=5.8,
                    color=mc,fontweight='bold',transform=ax.transAxes)

        c_str = '\n'.join([n[:8] for n in act]) if act else 'none'
        ax.text(0.5,0.38,c_str,ha='center',va='center',
                fontsize=5.5,color='#444444',transform=ax.transAxes)
        ax.text(0.5,0.06,status,ha='center',va='bottom',
                fontsize=6.5,style='italic',color=sc,
                transform=ax.transAxes)

    fig.text(0.5,0.93,
             'All 32 Concept Bins — Frequency and Soft/Hard Mismatch',
             ha='center',fontsize=13,fontweight='bold')
    fig.text(0.5,0.90,
             'E=early(in-dist)  L=late(OOD)  '
             'Red soft% = soft floods wrong bin at OOD',
             ha='center',fontsize=10,color='#444444')

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    return fig


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    os.makedirs(results_dir, exist_ok=True)
    data = collect_all_data(n_traj=500, seed=42, train_horizon=10)

    print("\nGenerating figures...")
    plot_early_vs_late(
        data,
        save_path=os.path.join(results_dir,'bin_early_vs_late.png'))
    plot_agreement_over_time(
        data,
        save_path=os.path.join(results_dir,'bin_agreement_over_time.png'))
    plot_bin_grid(
        data,
        save_path=os.path.join(results_dir,'bin_meaningfulness_grid.png'))

    plt.show()
    print("\nDone.")