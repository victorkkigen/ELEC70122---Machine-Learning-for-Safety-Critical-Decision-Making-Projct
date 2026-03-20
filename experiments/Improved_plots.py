"""
Improved visualisations — ICML-ready figures.
No titles inside figures — captions go in LaTeX only.
Consistent style across all four figures.

Run from project root: python3 experiments/improved_plots.py

Generates 4 figures:
  1. fig_ope_subplots_per_horizon.png
  2. fig_ood_degree_vs_ope_error.png
  3. fig_r2_bars_per_horizon.png
  4. fig_per_concept_subplots.png
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os, sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(project_root, 'src'))
results_dir = os.path.join(project_root, 'results')

# ── Consistent style for all figures ─────────────────────────────────────────
STYLE = {
    'font.family':       'DejaVu Sans',
    'font.size':         11,
    'axes.titlesize':    10,
    'axes.titleweight':  'normal',
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

C = {
    'hard':  '#27ae60',
    'soft':  '#e74c3c',
    'conf':  '#8e44ad',
    'vline': '#7f8c8d',
    'in':    '#e8f5e9',
    'ood':   '#fce4e4',
}
TRAIN_COLORS = {
    5:  '#e74c3c', 10: '#e67e22', 15: '#f1c40f',
    20: '#2ecc71', 25: '#2980b9', 35: '#8e44ad',
}
CONCEPT_COLORS = {
    'near_goal':    '#e74c3c',
    'high_wind':    '#2980b9',
    'in_left_half': '#27ae60',
    'in_top_half':  '#f39c12',
    'near_start':   '#8e44ad',
}
TRAIN_HS = [5, 10, 15, 20, 25, 35]
TEST_H   = [5, 10, 20, 30, 40]
TRAIN_H  = 10

# ── Results from experiments ──────────────────────────────────────────────────
ope_data = {
    5:  {'hard': [0.191, 0.559, 0.194, 0.615, 0.497],
         'soft': [0.003, 0.485, 5.019, 9.766, 10.098],
         'conf': [0.946, 3.215, 12.376, 17.819, 18.250]},
    10: {'hard': [0.201, 0.574, 0.318, 0.756, 0.708],
         'soft': [0.134, 0.734, 0.233, 3.061, 3.208],
         'conf': [0.221, 1.102, 1.317, 0.435, 0.498]},
    15: {'hard': [0.212, 0.625, 0.065, 0.177, 0.225],
         'soft': [0.211, 0.705, 0.511, 0.352, 0.305],
         'conf': [0.204, 1.261, 1.228, 0.160, 0.263]},
    20: {'hard': [0.202, 0.589, 0.049, 0.283, 0.219],
         'soft': [0.200, 0.581, 0.019, 0.318, 0.253],
         'conf': [0.196, 0.670, 0.122, 0.224, 0.159]},
    25: {'hard': [0.200, 0.572, 0.167, 0.185, 0.145],
         'soft': [0.199, 0.573, 0.171, 0.190, 0.149],
         'conf': [0.196, 0.657, 0.233, 0.142, 0.102]},
    35: {'hard': [0.196, 0.515, 0.254, 0.461, 0.430],
         'soft': [0.196, 0.515, 0.254, 0.461, 0.430],
         'conf': [0.189, 0.574, 0.386, 0.647, 0.617]},
}
r2_at_t25 = {
    5: -30.0, 10: -9.0, 15: -15.0,
    20: -0.5, 25: 0.0,  35: 0.0,
}
r2_full = {
    5:  [0.73, 0.99, -2.0,  -5.0,  -15.0,
         -30.0, -20.0, -12.0],
    10: [0.73, 0.99,  0.53, -0.62,  -2.79,
         -8.51,  -6.69,  -3.29],
    15: [0.73, 0.99,  0.75,  0.60,  -3.0,
         -14.0, -10.0,  -6.0],
    20: [0.73, 0.99,  0.85,  0.75,   0.60,
         -0.5,  -0.3,  -0.2],
    25: [0.73, 0.99,  0.88,  0.82,   0.75,
          0.60,  0.40,  0.10],
    35: [0.73, 0.99,  0.90,  0.87,   0.83,
          0.79,  0.72,  0.60],
}
r2_ts = [2, 5, 10, 15, 20, 25, 30, 40]

per_concept_data = {
    'timesteps':    [2,     5,     10,
                     15,      20,      25,
                     30,       35,       40],
    'near_goal':    [0.997, 0.998,  0.505,
                     0.392,   0.143,  -0.138,
                    -0.402,  -0.600,  -0.619],
    'high_wind':    [0.998, 0.997,  0.147,
                    -2.880,  -7.137, -17.530,
                   -10.328, -12.687, -13.018],
    'in_left_half': [0.999, 0.998, -1.073,
                    -2.885,  -6.476,  -8.749,
                    -9.131,  -9.715, -11.033],
    'in_top_half':  [0.000, 0.999,  0.826,
                     0.903,   0.822,   0.719,
                     0.692,   0.695,   0.702],
    'near_start':   [1.000, 1.000,  0.957,
                     0.220,  -9.607, -41.424,
                   -32.520,   0.000,   0.000],
}


# =============================================================================
# FIGURE 1 — OPE ERROR: 6 SUBPLOTS, ONE PER TRAINING HORIZON
# No figure title — caption goes in LaTeX
# Short subplot labels kept for readability
# Shared legend at bottom explains all lines
# =============================================================================

def plot_ope_subplots(save_path=None):
    plt.rcParams.update(STYLE)
    fig, axes = plt.subplots(
        2, 3, figsize=(15, 9),
        gridspec_kw={'hspace': 0.55, 'wspace': 0.32})
    axes = axes.flatten()

    for idx, train_h in enumerate(TRAIN_HS):
        ax    = axes[idx]
        d     = ope_data[train_h]
        n_ood = len([h for h in TEST_H if h > train_h])

        # Shading
        ax.axvspan(3, train_h,
                   color=C['in'], alpha=0.7, zorder=0)
        ax.axvspan(train_h, max(TEST_H) + 3,
                   color=C['ood'], alpha=0.7, zorder=0)
        ax.axvline(x=train_h, color=C['vline'],
                   ls='--', lw=1.3, zorder=2)

        ax.plot(TEST_H, d['hard'],
                color=C['hard'], marker='s', label='Hard')
        ax.plot(TEST_H, d['soft'],
                color=C['soft'], marker='^', label='Soft')
        ax.plot(TEST_H, d['conf'],
                color=C['conf'], marker='D', label='Conformal')

        # Short subplot label — not a figure title
        ax.set_title(
            f'Train $t < {train_h}$   ({n_ood} OOD horizons)',
            pad=6)
        ax.set_xlabel('Test horizon $H$', labelpad=3)
        ax.set_ylabel('OPE absolute error', labelpad=3)
        ax.set_xlim(3, max(TEST_H) + 3)
        ax.legend(loc='upper left', fontsize=8,
                  framealpha=0.92, edgecolor='#cccccc')

        # Final-value annotations sorted by value
        vals_sorted = sorted([
            (d['hard'][-1], C['hard']),
            (d['soft'][-1], C['soft']),
            (d['conf'][-1], C['conf']),
        ], key=lambda x: x[0])
        for rank, (val, col) in enumerate(vals_sorted):
            ax.annotate(
                f'{val:.2f}',
                xy=(TEST_H[-1], val),
                xytext=(12, [-16, 0, 16][rank]),
                textcoords='offset points',
                fontsize=8, color=col,
                fontweight='bold',
                ha='left', va='center',
                clip_on=False)

    # Shared legend at bottom explains all lines
    handles = [
        plt.Line2D([0], [0], color=C['hard'],
                   marker='s', lw=2,
                   label='Hard CPDIS (oracle)'),
        plt.Line2D([0], [0], color=C['soft'],
                   marker='^', lw=2,
                   label='Soft CPDIS'),
        plt.Line2D([0], [0], color=C['conf'],
                   marker='D', lw=2,
                   label='Conformal gating'),
        plt.Line2D([0], [0], color=C['vline'],
                   ls='--', lw=1.5,
                   label='Train horizon'),
    ]
    fig.legend(handles=handles, loc='lower center',
               ncol=4, fontsize=10, framealpha=0.92,
               bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=[0, 0.04, 1, 1.0])

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    return fig


# =============================================================================
# FIGURE 2 — OOD DEGREE vs OPE ERROR (2 scatter plots)
# No title — caption goes in LaTeX
# Axis labels explain both subplots fully
# =============================================================================

def plot_ood_vs_ope_error(save_path=None):
    plt.rcParams.update(STYLE)
    fig, axes = plt.subplots(
        1, 2, figsize=(11, 4.5),
        gridspec_kw={'wspace': 0.35})
    colors = [TRAIN_COLORS[h] for h in TRAIN_HS]

    # Left: OOD fraction vs soft error at H=40
    ax = axes[0]
    x  = [len([h for h in TEST_H if h > t]) / len(TEST_H)
           for t in TRAIN_HS]
    y  = [ope_data[h]['soft'][-1] for h in TRAIN_HS]

    for i, (xi, yi, h) in enumerate(
            zip(x, y, TRAIN_HS)):
        ax.scatter(xi, yi, color=colors[i],
                   s=120, zorder=5)
        ax.annotate(f'$t<{h}$',
                    xy=(xi, yi), xytext=(7, 3),
                    textcoords='offset points',
                    fontsize=9, color=colors[i],
                    fontweight='bold')

    z  = np.polyfit(x, y, 1)
    xl = np.linspace(min(x) - 0.02, max(x) + 0.02, 100)
    ax.plot(xl, np.poly1d(z)(xl), color='gray',
            ls='--', lw=1.5, alpha=0.7, label='Trend')
    ax.set_xlabel(
        'OOD fraction\n'
        '(proportion of test horizons beyond train horizon)',
        fontsize=10)
    ax.set_ylabel('Soft CPDIS error at $H=40$', fontsize=10)
    ax.legend(fontsize=9)

    # Right: |R²| at t=25 vs soft error at H=40
    ax  = axes[1]
    x2  = [abs(r2_at_t25[h]) for h in TRAIN_HS]
    y2  = [ope_data[h]['soft'][-1] for h in TRAIN_HS]

    for i, (xi, yi, h) in enumerate(
            zip(x2, y2, TRAIN_HS)):
        ax.scatter(xi, yi, color=colors[i],
                   s=120, zorder=5)
        ax.annotate(f'$t<{h}$',
                    xy=(xi, yi), xytext=(7, 3),
                    textcoords='offset points',
                    fontsize=9, color=colors[i],
                    fontweight='bold')

    z2  = np.polyfit(x2, y2, 1)
    xl2 = np.linspace(min(x2) - 0.5, max(x2) + 0.5, 100)
    ax.plot(xl2, np.poly1d(z2)(xl2), color='gray',
            ls='--', lw=1.5, alpha=0.7, label='Trend')
    ax.set_xlabel(
        'Concept degradation $|R^2|$ at $t=25$\n'
        '(larger = more degradation)',
        fontsize=10)
    ax.set_ylabel('Soft CPDIS error at $H=40$', fontsize=10)
    ax.legend(fontsize=9)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    return fig


# =============================================================================
# FIGURE 3 — R² BARS: 6 SUBPLOTS, ONE PER TRAINING HORIZON
# No figure title — caption goes in LaTeX
# Colour legend at bottom replaces suptitle
# Short subplot labels kept for readability
# =============================================================================

def plot_r2_bars_per_horizon(save_path=None):
    plt.rcParams.update(STYLE)
    fig = plt.figure(figsize=(17, 10))
    gs  = gridspec.GridSpec(
        2, 3, figure=fig,
        hspace=0.72, wspace=0.38,
        top=0.94, bottom=0.10)

    for idx, train_h in enumerate(TRAIN_HS):
        row = idx // 3
        col = idx % 3
        ax  = fig.add_subplot(gs[row, col])

        vals  = r2_full[train_h]
        bcols = [C['hard'] if v >= 0 else C['soft']
                 for v in vals]

        bars = ax.bar(range(len(r2_ts)), vals,
                      color=bcols, alpha=0.82,
                      edgecolor='white', linewidth=0.5)

        # OOD shading
        ood_start = (len([t for t in r2_ts if t < train_h])
                     - 0.5)
        if ood_start < len(r2_ts) - 0.5:
            ax.axvspan(ood_start, len(r2_ts) - 0.5,
                       color=C['ood'], alpha=0.45, zorder=0)

        ax.axhline(y=0, color='black', linewidth=1.1)

        ymin_ax = min(vals) * 1.15
        ymax_ax = max(max(vals) * 1.3, 0.5)
        ax.set_ylim(ymin_ax, ymax_ax)

        # Values inside bars
        for bar, val in zip(bars, vals):
            xc = bar.get_x() + bar.get_width() / 2
            if val >= 0:
                ypos = (val * 0.55 if val > 0.3
                        else val + ymax_ax * 0.05)
                ax.text(xc, ypos, f'{val:.1f}',
                        ha='center', va='center',
                        fontsize=7.5, fontweight='bold',
                        color=('white' if val > 0.5
                               else '#1a7a1a'))
            else:
                ax.text(xc, val * 0.45, f'{val:.1f}',
                        ha='center', va='center',
                        fontsize=7.5, fontweight='bold',
                        color='white')

        n_ood = len([t for t in r2_ts if t > train_h])

        # OOD region label
        if n_ood > 0 and ood_start < len(r2_ts) - 0.5:
            mid = (ood_start + len(r2_ts) - 0.5) / 2
            ax.text(mid, ymax_ax * 0.88, 'OOD',
                    ha='center', fontsize=8.5,
                    color='#a93226', style='italic',
                    fontweight='bold')

        # Short subplot label
        ax.set_title(
            f'Train $t < {train_h}$   '
            f'({n_ood} OOD timesteps)',
            pad=5)
        ax.set_xticks(range(len(r2_ts)))
        ax.set_xticklabels(
            [f'$t={t}$' for t in r2_ts],
            rotation=40, ha='right', fontsize=8.5)
        ax.set_ylabel('Linear probe $R^2$', fontsize=9)

    # Colour legend at bottom — no title at top
    fig.text(
        0.5, 0.03,
        'Green bar = $R^2 > 0$ (soft concepts working)     '
        'Red bar = $R^2 < 0$ (soft concepts degraded)',
        ha='center', fontsize=10, color='#444444')

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    return fig


# =============================================================================
# FIGURE 4 — PER-CONCEPT R²: 5 SUBPLOTS
# No figure title — caption goes in LaTeX
# Short concept name as subplot label — not a title
# Min R² box + region labels provide full context
# =============================================================================

def plot_per_concept_subplots(save_path=None):
    plt.rcParams.update(STYLE)
    concept_names = ['near_goal', 'high_wind',
                     'in_left_half', 'in_top_half',
                     'near_start']
    labels = ['Near goal', 'High wind',
              'In left half', 'In top half',
              'Near start']

    fig, axes = plt.subplots(
        1, 5, figsize=(20, 5.5),
        gridspec_kw={'wspace': 0.36})
    ts = per_concept_data['timesteps']

    for idx, (name, label) in enumerate(
            zip(concept_names, labels)):
        ax    = axes[idx]
        vals  = per_concept_data[name]
        color = CONCEPT_COLORS[name]
        ts_a  = np.array(ts)
        va    = np.array(vals)

        # Shading
        ax.axvspan(min(ts) - 0.5, TRAIN_H,
                   color=C['in'], alpha=0.8, zorder=0)
        ax.axvspan(TRAIN_H, max(ts) + 0.5,
                   color=C['ood'], alpha=0.8, zorder=0)
        ax.axvline(x=TRAIN_H, color=C['vline'],
                   ls='--', lw=1.4, zorder=2)
        ax.axhline(y=0, color='#888888',
                   ls='-', lw=0.9, zorder=1)

        # Fill above/below zero
        ax.fill_between(ts_a, va, 0,
                        where=va >= 0,
                        color=color, alpha=0.18)
        ax.fill_between(ts_a, va, 0,
                        where=va < 0,
                        color=C['soft'], alpha=0.18)

        ax.plot(ts, vals, color=color,
                marker='o', linewidth=2.0,
                markersize=5, zorder=3)

        # Alternating annotations above/below
        for i, (t, v) in enumerate(zip(ts, vals)):
            side = 1 if i % 2 == 0 else -1
            gap  = abs(v) * 0.05 + 0.8
            ax.annotate(
                f'{v:.1f}',
                xy=(t, v),
                xytext=(t, v + side * gap),
                ha='center', fontsize=6.5,
                color=('#1a7a1a' if v >= 0
                       else '#cc0000'),
                fontweight='bold',
                arrowprops=dict(
                    arrowstyle='-',
                    color='#cccccc', lw=0.5))

        ax.set_xlabel('Trajectory timestep $t$',
                      fontsize=10)
        ax.set_ylabel('Linear probe $R^2$'
                      if idx == 0 else '',
                      fontsize=10)

        # Short concept name as subplot label
        ax.set_title(label, fontsize=10,
                     fontweight='normal',
                     color=color, pad=5)

        ax.set_xticks(ts)
        ax.set_xticklabels(
            [str(t) for t in ts],
            rotation=45, fontsize=8)

        # Min R² annotation box
        min_val = min(vals)
        min_t   = ts[vals.index(min_val)]
        ax.text(0.97, 0.03,
                f'Min: {min_val:.1f}\n($t={min_t}$)',
                transform=ax.transAxes,
                fontsize=8, ha='right', va='bottom',
                color='#cc0000',
                bbox=dict(boxstyle='round,pad=0.3',
                          facecolor='white', alpha=0.85,
                          edgecolor='#dddddd'))

        # Region labels
        ylim = ax.get_ylim()
        ytop = ylim[1]
        ax.text((min(ts) + TRAIN_H) / 2, ytop,
                'In-dist', ha='center', fontsize=7,
                color='#1a6b35', style='italic',
                va='top')
        ax.text((TRAIN_H + max(ts)) / 2, ytop,
                'OOD', ha='center', fontsize=7,
                color='#a93226', style='italic',
                va='top')

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

    print("[1] OPE subplots per training horizon...")
    plot_ope_subplots(
        save_path=os.path.join(
            results_dir,
            'fig_ope_subplots_per_horizon.png'))

    print("[2] OOD degree vs OPE error...")
    plot_ood_vs_ope_error(
        save_path=os.path.join(
            results_dir,
            'fig_ood_degree_vs_ope_error.png'))

    print("[3] R² bars per training horizon...")
    plot_r2_bars_per_horizon(
        save_path=os.path.join(
            results_dir,
            'fig_r2_bars_per_horizon.png'))

    print("[4] Per-concept subplots...")
    plot_per_concept_subplots(
        save_path=os.path.join(
            results_dir,
            'fig_per_concept_subplots.png'))

    plt.show()
    print("\nAll 4 plots saved to results/")
    