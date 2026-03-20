"""
Per-Concept Conformal Gating — ICML-ready figures.
No titles inside figures — captions go in LaTeX only.
Consistent style across both figures.

Run: python3 experiments/per_concept_conformal.py
"""

import numpy as np
import matplotlib.pyplot as plt
import os, sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(project_root, 'src'))
results_dir = os.path.join(project_root, 'results')

from gridworld import WindyGridworld, collect_trajectory
from policies import EpsilonGreedyPolicy, OptimalPolicy
from concepts import (HardConcepts, SoftConcepts,
                      ConformalGatedConcepts, SoftConceptEncoder)
from utils import set_seed

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

C_HARD   = '#27ae60'
C_SOFT   = '#e74c3c'
C_CONF   = '#8e44ad'
C_PCCONF = '#f39c12'
C_VLINE  = '#7f8c8d'
C_IN     = '#e8f5e9'
C_OOD    = '#fce4e4'

CONCEPT_COLORS = ['#e74c3c', '#2980b9', '#27ae60',
                  '#f39c12', '#8e44ad']

concept_names = ['near_goal', 'high_wind', 'in_left_half',
                 'in_top_half', 'near_start']


# =============================================================================
# PER-CONCEPT CONFORMAL GATING CLASS
# =============================================================================

class PerConceptConformalGating:
    """
    Conformal prediction gating applied independently per concept.

    For each concept i:
      score_i(s)  = |hard(s)[i] - soft(s)[i]|
      threshold_i = 95th percentile of calibration scores
      gate_i(s)   = 1.0             if score_i <= threshold_i
      gate_i(s)   = 1/(1+excess_i) otherwise

      c_out[i] = global_mean[i] + gate_i(s) * (soft[i] - global_mean[i])

    global_mean computed from training states only — no data leakage.
    """

    def __init__(self, env, seed=None):
        self.env            = env
        self.n_concepts     = 5
        self.seed           = seed
        sample_features     = env.state_to_features((0, 0))
        self.input_dim      = len(sample_features)
        self.encoder        = SoftConceptEncoder(
            input_dim=self.input_dim,
            hidden_dim=32,
            output_dim=self.n_concepts,
            seed=seed)
        self.thresholds     = np.ones(self.n_concepts) * 0.05
        self.global_mean    = np.ones(self.n_concepts) * 0.5
        self._hard_concepts = None

    def _soft_probs(self, state) -> np.ndarray:
        features = self.env.state_to_features(state)
        probs, _ = self.encoder.forward(features)
        return probs

    def _per_concept_scores(self, state) -> np.ndarray:
        soft = self._soft_probs(state)
        hard = self._hard_concepts.extract(state)
        return np.abs(soft - hard)

    def _per_concept_gates(self, state) -> np.ndarray:
        scores = self._per_concept_scores(state)
        gates  = np.ones(self.n_concepts)
        for i in range(self.n_concepts):
            if scores[i] > self.thresholds[i]:
                excess   = ((scores[i] - self.thresholds[i]) /
                            (self.thresholds[i] + 1e-7))
                gates[i] = max(0.0, 1.0 / (1.0 + excess))
        return gates

    def calibrate(self, states, hard_concepts):
        scores_per_concept = [[] for _ in range(self.n_concepts)]
        for s in states:
            hard = hard_concepts.extract(s)
            soft = self._soft_probs(s)
            for i in range(self.n_concepts):
                scores_per_concept[i].append(abs(hard[i] - soft[i]))
        print(f"    Per-concept calibration ({len(states)} states):")
        for i, name in enumerate(concept_names):
            self.thresholds[i] = float(
                np.percentile(scores_per_concept[i], 95))
            print(f"      {name:<16}: "
                  f"threshold={self.thresholds[i]:.4f}  "
                  f"mean={np.mean(scores_per_concept[i]):.4f}")

    def get_gate_value(self, state) -> float:
        return float(np.mean(self._per_concept_gates(state)))

    def extract(self, state) -> np.ndarray:
        gates = self._per_concept_gates(state)
        soft  = self._soft_probs(state)
        return self.global_mean + gates * (soft - self.global_mean)

    def __call__(self, state) -> np.ndarray:
        return self.extract(state)

    def train_on_trajectories(self, trajectories, hard_concepts,
                              epochs=100, lr=0.01, verbose=False):
        self._hard_concepts = hard_concepts
        X, Y, cal_states    = [], [], []

        for traj in trajectories:
            for step in traj:
                state    = step['state']
                features = step.get(
                    'features', self.env.state_to_features(state))
                X.append(features)
                Y.append(hard_concepts.extract(state))
                cal_states.append(state)

        X = np.array(X)
        Y = np.array(Y)

        for epoch in range(epochs):
            perm       = np.random.permutation(len(X))
            total_loss = 0
            n_batches  = 0
            for i in range(0, len(X), 32):
                loss = self.encoder.train_step(
                    X[perm][i:i+32], Y[perm][i:i+32], lr=lr)
                total_loss += loss
                n_batches  += 1
            if verbose and (epoch + 1) % 20 == 0:
                print(f"    Epoch {epoch+1}/{epochs} "
                      f"Loss={total_loss/n_batches:.4f}")

        all_probs = []
        for x in X:
            probs, _ = self.encoder.forward(x)
            all_probs.append(probs)
        self.global_mean = np.mean(all_probs, axis=0)

        print(f"    Global mean (training states only): "
              f"{[f'{v:.3f}' for v in self.global_mean]}")
        self.calibrate(cal_states, hard_concepts)


# =============================================================================
# POLICY CLASSES
# =============================================================================

class ConceptBasedPolicy:
    def __init__(self, concept_extractor,
                 n_concepts=32, n_actions=4):
        self.concept_extractor = concept_extractor
        self.n_concepts        = n_concepts
        self.n_actions         = n_actions
        self.policy_table      = (np.ones((n_concepts, n_actions))
                                  / n_actions)

    def state_to_concept_index(self, state) -> int:
        concept_vec = self.concept_extractor(state)
        if len(concept_vec) > 5:
            concept_vec = concept_vec[:5]
        binary = (concept_vec > 0.5).astype(int)
        return int(sum(b * (2**i)
                       for i, b in enumerate(binary)))

    def learn_from_trajectories(self, trajectories,
                                smoothing=1.0):
        counts = (np.ones((self.n_concepts, self.n_actions))
                  * smoothing)
        for traj in trajectories:
            for step in traj:
                c = self.state_to_concept_index(step['state'])
                counts[c, step['action']] += 1
        self.policy_table = (counts /
                             counts.sum(axis=1, keepdims=True))

    def prob(self, state, action) -> float:
        return float(self.policy_table[
            self.state_to_concept_index(state), action])


class GatedPolicy:
    def __init__(self, gated_extractor,
                 n_concepts=32, n_actions=4):
        self.gated_extractor = gated_extractor
        self.n_concepts      = n_concepts
        self.n_actions       = n_actions
        self.policy_table    = (np.ones((n_concepts, n_actions))
                                / n_actions)
        self.global_policy   = np.ones(n_actions) / n_actions

    def state_to_concept_index(self, state) -> int:
        concept_vec = self.gated_extractor(state)
        if len(concept_vec) > 5:
            concept_vec = concept_vec[:5]
        binary = (concept_vec > 0.5).astype(int)
        return int(sum(b * (2**i)
                       for i, b in enumerate(binary)))

    def learn_from_trajectories(self, trajectories,
                                smoothing=1.0):
        counts = (np.ones((self.n_concepts, self.n_actions))
                  * smoothing)
        for traj in trajectories:
            for step in traj:
                c = self.state_to_concept_index(step['state'])
                counts[c, step['action']] += 1
        self.policy_table = (counts /
                             counts.sum(axis=1, keepdims=True))
        total = counts.sum(axis=0)
        self.global_policy = total / total.sum()

    def prob(self, state, action) -> float:
        gate = self.gated_extractor.get_gate_value(state)
        c    = self.state_to_concept_index(state)
        return (gate * float(self.policy_table[c, action]) +
                (1 - gate) * float(self.global_policy[action]))


def cpdis(trajectories, b_pol, e_pol, H, gamma=0.99):
    returns = []
    for traj in trajectories:
        G, rho = 0.0, 1.0
        for t in range(min(len(traj), H)):
            step = traj[t]
            pe   = e_pol.prob(step['state'], step['action'])
            pb   = b_pol.prob(step['state'], step['action'])
            rho *= (pe / pb) if pb > 1e-10 else 0.0
            G   += (gamma**t) * rho * step['reward']
        returns.append(G)
    returns = np.array(returns)
    return float(np.mean(returns)), float(np.var(returns))


def mc_true(env, eval_policy, H, n=1000, gamma=0.99):
    returns = []
    for _ in range(n):
        traj = collect_trajectory(env, eval_policy,
                                  max_steps=H)
        returns.append(sum((gamma**t) * s['reward']
                           for t, s in enumerate(traj)))
    return float(np.mean(returns))


# =============================================================================
# EXPERIMENT — now collects all 4 metrics
# =============================================================================

def run_experiment(seed=42, n_traj=500, train_horizon=10,
                   test_horizons=None):
    if test_horizons is None:
        test_horizons = [5, 10, 15, 20, 25, 30, 35, 40]

    set_seed(seed)
    env  = WindyGridworld()
    bp   = EpsilonGreedyPolicy(env, epsilon=0.4, seed=seed)
    ep   = OptimalPolicy(env, epsilon=0.05, seed=seed)
    hard = HardConcepts(env)

    print(f"\nSeed {seed} — Collecting {n_traj} trajectories...")
    trajs = [collect_trajectory(env, bp, max_steps=50)
             for _ in range(n_traj)]

    train_trajs = []
    for traj in trajs[:200]:
        early = [s for i, s in enumerate(traj)
                 if i < train_horizon]
        if early:
            train_trajs.append(early)

    print("  Training soft concepts...")
    soft = SoftConcepts(env, use_leakage=True, seed=seed)
    soft.train_on_trajectories(train_trajs, hard, epochs=200)

    print("  Training standard conformal...")
    conf = ConformalGatedConcepts(env, seed=seed)
    conf.train_on_trajectories(train_trajs, hard, epochs=200)

    print("  Training per-concept conformal...")
    pcconf = PerConceptConformalGating(env, seed=seed)
    pcconf.train_on_trajectories(train_trajs, hard, epochs=200)

    hard_b   = ConceptBasedPolicy(hard,   32, 4)
    soft_b   = ConceptBasedPolicy(soft,   32, 4)
    conf_b   = GatedPolicy(conf,          32, 4)
    pcconf_b = GatedPolicy(pcconf,        32, 4)

    for pol in [hard_b, soft_b, conf_b, pcconf_b]:
        pol.learn_from_trajectories(trajs)

    eval_trajs = [collect_trajectory(env, ep, max_steps=50)
                  for _ in range(200)]

    hard_e   = ConceptBasedPolicy(hard,   32, 4)
    soft_e   = ConceptBasedPolicy(soft,   32, 4)
    conf_e   = GatedPolicy(conf,          32, 4)
    pcconf_e = GatedPolicy(pcconf,        32, 4)

    for pol in [hard_e, soft_e, conf_e, pcconf_e]:
        pol.learn_from_trajectories(eval_trajs)

    # All 4 metrics stored
    results = {k: {'estimates': [], 'errors': [],
                   'variances': [], 'mse': [],
                   'bias': []}
               for k in ['hard', 'soft', 'conf', 'pcconf']}
    results['horizons']    = test_horizons
    results['true_values'] = []

    for H in test_horizons:
        tv = mc_true(env, ep, H, n=500)
        results['true_values'].append(tv)

        for key, b_pol, e_pol in [
            ('hard',   hard_b,   hard_e),
            ('soft',   soft_b,   soft_e),
            ('conf',   conf_b,   conf_e),
            ('pcconf', pcconf_b, pcconf_e),
        ]:
            est, var = cpdis(trajs, b_pol, e_pol, H)
            err  = abs(est - tv)
            bias = est - tv
            mse  = (est - tv) ** 2

            results[key]['estimates'].append(est)
            results[key]['errors'].append(err)
            results[key]['variances'].append(var)
            results[key]['bias'].append(bias)
            results[key]['mse'].append(mse)

    return results


def run_multiseed(n_seeds=30, test_horizons=None):
    if test_horizons is None:
        test_horizons = [5, 10, 15, 20, 25, 30, 35, 40]

    keys     = ['hard', 'soft', 'conf', 'pcconf']
    all_est  = {k: [] for k in keys}
    all_true = []

    for seed in range(n_seeds):
        print(f"\n{'='*50}\nSEED {seed+1}/{n_seeds}")
        r = run_experiment(seed=seed,
                           test_horizons=test_horizons)
        all_true.append(r['true_values'])
        for k in keys:
            all_est[k].append(r[k]['estimates'])

    true_vals = np.mean(all_true, axis=0)
    labels    = {
        'hard':   'Hard CPDIS',
        'soft':   'Soft CPDIS',
        'conf':   'Conformal (1 gate)',
        'pcconf': 'Per-concept conformal (5 gates)',
    }

    # Build multiseed results dict with all 4 metrics
    ms_results = {k: {'errors': [], 'variances': [],
                      'mse': [], 'bias': []}
                  for k in keys}
    ms_results['horizons']    = test_horizons
    ms_results['true_values'] = list(true_vals)

    print(f"\n{'='*65}")
    print(f"MULTI-SEED RESULTS ({n_seeds} seeds)")
    print(f"{'H':<5} {'Estimator':<32} {'MSE':<10} "
          f"{'Bias':<10} {'Variance':<12}")
    print("-" * 70)

    for k in keys:
        ests = np.array(all_est[k])
        for i, H in enumerate(test_horizons):
            tv   = true_vals[i]
            mse  = float(np.mean((ests[:, i] - tv)**2))
            bias = float(np.mean(ests[:, i]) - tv)
            var  = float(np.var(ests[:, i]))
            err  = float(np.mean(np.abs(ests[:, i] - tv)))

            ms_results[k]['mse'].append(mse)
            ms_results[k]['bias'].append(bias)
            ms_results[k]['variances'].append(var)
            ms_results[k]['errors'].append(err)

            print(f"{H:<5} {labels[k]:<32} {mse:<10.4f} "
                  f"{bias:<10.4f} {var:<12.4f}")
        print()

    return ms_results


# =============================================================================
# PER-CONCEPT GATE BEHAVIOUR ANALYSIS
# =============================================================================

def analyse_per_concept_gates(env, hard, pcconf,
                               trajectories,
                               train_horizon=10):
    ts         = list(range(0, 43, 2))
    gates_by_t = {t: {i: [] for i in range(5)}
                  for t in ts}

    for traj in trajectories:
        for t, step in enumerate(traj):
            if t in gates_by_t:
                gates = pcconf._per_concept_gates(
                    step['state'])
                for i in range(5):
                    gates_by_t[t][i].append(gates[i])

    print(f"\n{'='*75}")
    print("PER-CONCEPT GATE VALUES OVER TIME")
    print(f"{'t':<6} {'in-dist':<10}", end="")
    for name in concept_names:
        print(f" {name[:10]:<13}", end="")
    print()
    print("-" * 80)

    for t in ts:
        if not gates_by_t[t][0]:
            continue
        in_dist = "Yes" if t < train_horizon else "No"
        print(f"{t:<6} {in_dist:<10}", end="")
        for i in range(5):
            print(f" {np.mean(gates_by_t[t][i]):<13.3f}",
                  end="")
        print()

    return gates_by_t


# =============================================================================
# FIGURE 1 — 4 METRICS: 2x2 SUBPLOTS
# No title — caption goes in LaTeX
# Axis labels + legend + shading sufficient
# =============================================================================

def plot_comparison(results, save_path=None):
    plt.rcParams.update(STYLE)
    fig, axes = plt.subplots(
        2, 2, figsize=(12, 9),
        gridspec_kw={'hspace': 0.38, 'wspace': 0.32})

    H       = results['horizons']
    cols    = {'hard':   C_HARD,
               'soft':   C_SOFT,
               'conf':   C_CONF,
               'pcconf': C_PCCONF}
    marks   = {'hard': 's', 'soft': '^',
                'conf': 'D', 'pcconf': 'o'}
    labs    = {'hard':   'Hard CPDIS (oracle)',
               'soft':   'Soft CPDIS',
               'conf':   'Conformal (1 gate)',
               'pcconf': 'Per-concept conformal (5 gates)'}
    TRAIN_H = 10

    panels = [
        ('errors',    'OPE absolute error',  axes[0, 0]),
        ('mse',       'MSE',                 axes[0, 1]),
        ('bias',      'Bias',                axes[1, 0]),
        ('variances', 'Variance',            axes[1, 1]),
    ]

    for metric, ylabel, ax in panels:
        # Shading
        ax.axvspan(min(H) - 2, TRAIN_H,
                   color=C_IN, alpha=0.7, zorder=0)
        ax.axvspan(TRAIN_H, max(H) + 2,
                   color=C_OOD, alpha=0.7, zorder=0)
        ax.axvline(x=TRAIN_H, color=C_VLINE,
                   ls='--', lw=1.4, zorder=2)

        for k in ['hard', 'soft', 'conf', 'pcconf']:
            ax.plot(H, results[k][metric],
                    color=cols[k], marker=marks[k],
                    linewidth=2.0, markersize=6,
                    label=labs[k], zorder=3)

        ax.set_xlabel('Test horizon $H$', fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_xlim(min(H) - 2, max(H) + 2)

        # Region labels
        ylim = ax.get_ylim()
        ytop = ylim[0] + (ylim[1] - ylim[0]) * 0.96
        ax.text((min(H) + TRAIN_H) / 2, ytop,
                'In-dist', ha='center', fontsize=8.5,
                color='#1a6b35', style='italic')
        ax.text((TRAIN_H + max(H)) / 2, ytop,
                'OOD', ha='center', fontsize=8.5,
                color='#a93226', style='italic')

    # One shared legend for all subplots
    handles = [
        plt.Line2D([0], [0], color=C_HARD,
                   marker='s', lw=2,
                   label='Hard CPDIS (oracle)'),
        plt.Line2D([0], [0], color=C_SOFT,
                   marker='^', lw=2,
                   label='Soft CPDIS'),
        plt.Line2D([0], [0], color=C_CONF,
                   marker='D', lw=2,
                   label='Conformal (1 gate)'),
        plt.Line2D([0], [0], color=C_PCCONF,
                   marker='o', lw=2,
                   label='Per-concept conformal (5 gates)'),
        plt.Line2D([0], [0], color=C_VLINE,
                   ls='--', lw=1.5,
                   label='Train horizon'),
    ]
    fig.legend(handles=handles, loc='lower center',
               ncol=3, fontsize=9, framealpha=0.92,
               bbox_to_anchor=(0.5, 0.01))
    fig.tight_layout(rect=[0, 0.06, 1, 1.0])

    if save_path:
        fig.savefig(save_path, dpi=150,
                    bbox_inches='tight')
        print(f"  Saved: {save_path}")
    return fig


# =============================================================================
# FIGURE 2 — PER-CONCEPT GATE VALUES OVER TIME
# No title — caption goes in LaTeX
# =============================================================================

def plot_gate_values(gates_by_t, train_horizon=10,
                     save_path=None):
    plt.rcParams.update(STYLE)
    ts = sorted(gates_by_t.keys())

    fig, ax = plt.subplots(figsize=(9, 4.5))

    # Shading
    ax.axvspan(-0.5, train_horizon,
               color=C_IN, alpha=0.7, zorder=0)
    ax.axvspan(train_horizon, max(ts) + 0.5,
               color=C_OOD, alpha=0.7, zorder=0)
    ax.axvline(x=train_horizon, color=C_VLINE,
               ls='--', lw=1.5,
               label=f'Train horizon ($t={train_horizon}$)',
               zorder=2)

    for i, (name, color) in enumerate(
            zip(concept_names, CONCEPT_COLORS)):
        vals  = [np.mean(gates_by_t[t][i])
                 if gates_by_t[t][i] else np.nan
                 for t in ts]
        ts_v  = [t for t, v in zip(ts, vals)
                  if not np.isnan(v)]
        val_v = [v for v in vals if not np.isnan(v)]

        ax.plot(ts_v, val_v, color=color,
                marker='o', linewidth=2.0,
                markersize=5, label=name, zorder=3)

        if val_v:
            ax.annotate(
                f'{val_v[-1]:.2f}',
                xy=(ts_v[-1], val_v[-1]),
                xytext=(4, 0),
                textcoords='offset points',
                fontsize=8, color=color,
                fontweight='bold', clip_on=False)

    # Region labels
    ax.text(train_horizon / 2, 1.06,
            'In-distribution', ha='center',
            fontsize=9, color='#1a6b35',
            style='italic')
    ax.text((train_horizon + max(ts)) / 2, 1.06,
            'Out-of-distribution', ha='center',
            fontsize=9, color='#a93226',
            style='italic')

    ax.set_xlabel('Trajectory timestep $t$', fontsize=10)
    ax.set_ylabel('Gate value  (1 = open,  0 = closed)',
                  fontsize=10)
    ax.set_ylim(-0.05, 1.12)
    ax.set_xlim(-0.5, max(ts) + 0.5)
    ax.legend(fontsize=9, loc='lower left')

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150,
                    bbox_inches='tight')
        print(f"  Saved: {save_path}")
    return fig


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    os.makedirs(results_dir, exist_ok=True)

    print("Running single seed experiment (seed=42)...")
    results = run_experiment(
        seed=42,
        test_horizons=[5, 10, 15, 20, 25, 30, 35, 40])

    print(f"\n{'='*65}")
    print("SINGLE SEED RESULTS (seed=42)")
    print(f"{'H':<5} {'Hard':<12} {'Soft':<12} "
          f"{'Conformal':<12} {'Per-concept':<12}")
    print("-" * 55)
    for i, H in enumerate(results['horizons']):
        print(f"{H:<5} "
              f"{results['hard']['errors'][i]:<12.4f} "
              f"{results['soft']['errors'][i]:<12.4f} "
              f"{results['conf']['errors'][i]:<12.4f} "
              f"{results['pcconf']['errors'][i]:<12.4f}")

    # Gate behaviour analysis
    set_seed(42)
    env   = WindyGridworld()
    bp    = EpsilonGreedyPolicy(env, epsilon=0.4, seed=42)
    hard  = HardConcepts(env)
    trajs = [collect_trajectory(env, bp, max_steps=50)
             for _ in range(500)]

    train_trajs = []
    for traj in trajs[:200]:
        early = [s for i, s in enumerate(traj) if i < 10]
        if early:
            train_trajs.append(early)

    pcconf = PerConceptConformalGating(env, seed=42)
    pcconf.train_on_trajectories(train_trajs, hard,
                                 epochs=200)

    gates_by_t = analyse_per_concept_gates(
        env, hard, pcconf, trajs, train_horizon=10)

    print("\nGenerating figures...")
    plot_comparison(
        results,
        save_path=os.path.join(
            results_dir,
            'fig_per_concept_conformal_comparison.png'))

    plot_gate_values(
        gates_by_t, train_horizon=10,
        save_path=os.path.join(
            results_dir,
            'fig_per_concept_gate_values.png'))

    plt.show()

    # Multi-seed validation — uncomment when ready (~30 min)
    # print("\nRunning 30-seed experiment...")
    # multi = run_multiseed(n_seeds=30)
    # np.save(os.path.join(results_dir,
    #         'per_concept_multiseed.npy'), multi)