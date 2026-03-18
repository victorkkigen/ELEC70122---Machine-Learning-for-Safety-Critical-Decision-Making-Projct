"""
Per-Concept Conformal Gating
Novel improvement: instead of one global gate for all concepts,
use a SEPARATE gate for each concept independently.

Key insight from our analysis:
  high_wind:   always accurate → gate should never close
  near_start:  always accurate → gate should never close
  near_goal:   fails immediately at OOD → gate should close fast
  in_top_half: fails quickly at OOD → gate should close fast
  in_left_half: degrades gradually → gate closes proportionally

Each concept gets its own:
  - Calibration threshold (95th pct of its own score)
  - Gate value (1/(1+excess) for that concept only)
  - Fallback = its own global_mean[i] (training average)

Formula:
  c_out[i] = global_mean[i] + gate_i(s) × (soft[i] - global_mean[i])

Compare against:
  Hard CPDIS          (oracle)
  Soft CPDIS          (leaks)
  Current conformal   (1 gate for all)
  Per-concept conformal (5 gates, one per concept)

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

plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 11,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.grid': True, 'grid.alpha': 0.22, 'figure.dpi': 150,
})

concept_names = ['near_goal', 'high_wind', 'in_left_half',
                 'in_top_half', 'near_start']


# =============================================================================
# PER-CONCEPT CONFORMAL GATING CLASS
# =============================================================================

class PerConceptConformalGating:
    """
    Conformal prediction gating applied INDEPENDENTLY per concept.

    For each concept i:
      score_i(s) = |hard(s)[i] - soft(s)[i]|
      threshold_i = 95th percentile of calibration scores for concept i
      gate_i(s)  = 1.0                      if score_i <= threshold_i
      gate_i(s)  = 1/(1+excess_i)           if score_i >  threshold_i

      c_out[i] = global_mean[i] + gate_i(s) × (soft[i] - global_mean[i])

    Advantage over single-gate conformal:
      Concepts that are always accurate (high_wind, near_start)
      get gate_i ≈ 1.0 always → no unnecessary perturbation
      Concepts that fail at OOD (near_goal, in_top_half)
      get gate_i → 0.0 quickly → fast fallback to global_mean[i]
      Concepts that degrade gradually (in_left_half)
      get gate_i closing proportionally

    Key honest constraint:
      global_mean[i] = average of concept[i] over TRAINING states only
      (t < train_horizon)
      We do NOT use OOD states to compute global_mean → no data leakage
    """

    def __init__(self, env, seed=None):
        self.env        = env
        self.n_concepts = 5
        self.seed       = seed

        sample_features = env.state_to_features((0, 0))
        self.input_dim  = len(sample_features)

        self.encoder = SoftConceptEncoder(
            input_dim=self.input_dim,
            hidden_dim=32,
            output_dim=self.n_concepts,
            seed=seed
        )

        # Per-concept calibration thresholds and global means
        self.thresholds  = np.ones(self.n_concepts) * 0.05
        self.global_mean = np.ones(self.n_concepts) * 0.5
        self._hard_concepts = None

    def _soft_probs(self, state) -> np.ndarray:
        features = self.env.state_to_features(state)
        probs, _ = self.encoder.forward(features)
        return probs

    def _per_concept_scores(self, state) -> np.ndarray:
        """
        Returns 5 scores — one per concept.
        score[i] = |hard[i] - soft[i]|
        """
        soft = self._soft_probs(state)
        hard = self._hard_concepts.extract(state)
        return np.abs(soft - hard)

    def _per_concept_gates(self, state) -> np.ndarray:
        """
        Returns 5 gate values — one per concept.
        gate[i] = 1.0 if score[i] <= threshold[i]
        gate[i] = 1/(1+excess) if score[i] > threshold[i]
        """
        scores = self._per_concept_scores(state)
        gates  = np.ones(self.n_concepts)
        for i in range(self.n_concepts):
            if scores[i] > self.thresholds[i]:
                excess   = (scores[i] - self.thresholds[i]) / \
                           (self.thresholds[i] + 1e-7)
                gates[i] = max(0.0, 1.0 / (1.0 + excess))
        return gates

    def calibrate(self, states, hard_concepts):
        """
        Calibrate each concept independently.
        threshold[i] = 95th percentile of |hard[i] - soft[i]|
        over all calibration (training) states.
        """
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
            print(f"      {name:<16}: threshold={self.thresholds[i]:.4f}  "
                  f"mean_score={np.mean(scores_per_concept[i]):.4f}")

    def get_gate_value(self, state) -> float:
        """Overall gate = mean of per-concept gates (for compatibility)."""
        return float(np.mean(self._per_concept_gates(state)))

    def extract(self, state) -> np.ndarray:
        """Per-concept gated output."""
        gates = self._per_concept_gates(state)
        soft  = self._soft_probs(state)
        return self.global_mean + gates * (soft - self.global_mean)

    def __call__(self, state) -> np.ndarray:
        return self.extract(state)

    def train_on_trajectories(self, trajectories, hard_concepts,
                              epochs=100, lr=0.01, verbose=False):
        self._hard_concepts = hard_concepts

        X, Y, cal_states = [], [], []
        for traj in trajectories:
            for step in traj:
                state    = step['state']
                features = step.get('features',
                                    self.env.state_to_features(state))
                X.append(features)
                Y.append(hard_concepts.extract(state))
                cal_states.append(state)

        X = np.array(X); Y = np.array(Y)

        for epoch in range(epochs):
            perm       = np.random.permutation(len(X))
            total_loss = 0; n_batches = 0
            for i in range(0, len(X), 32):
                loss = self.encoder.train_step(
                    X[perm][i:i+32], Y[perm][i:i+32], lr=lr)
                total_loss += loss; n_batches += 1
            if verbose and (epoch+1) % 20 == 0:
                print(f"    Epoch {epoch+1}/{epochs} "
                      f"Loss={total_loss/n_batches:.4f}")

        # Global mean from training states only (honest — no OOD data)
        all_probs = []
        for x in X:
            probs, _ = self.encoder.forward(x)
            all_probs.append(probs)
        self.global_mean = np.mean(all_probs, axis=0)

        print(f"    Per-concept global_mean: "
              f"{[f'{v:.3f}' for v in self.global_mean]}")
        print(f"    Meaning: fraction of TRAINING states where each "
              f"concept is active")
        for i, name in enumerate(concept_names):
            print(f"      {name:<16}: {self.global_mean[i]:.3f}")

        # Calibrate per concept
        self.calibrate(cal_states, hard_concepts)


# =============================================================================
# POLICY CLASSES (reused from main experiment)
# =============================================================================

class ConceptBasedPolicy:
    def __init__(self, concept_extractor, n_concepts=32, n_actions=4):
        self.concept_extractor = concept_extractor
        self.n_concepts        = n_concepts
        self.n_actions         = n_actions
        self.policy_table      = np.ones((n_concepts, n_actions)) / n_actions

    def state_to_concept_index(self, state) -> int:
        concept_vec = self.concept_extractor(state)
        if len(concept_vec) > 5:
            concept_vec = concept_vec[:5]
        binary = (concept_vec > 0.5).astype(int)
        return int(sum(b * (2**i) for i, b in enumerate(binary)))

    def learn_from_trajectories(self, trajectories, smoothing=1.0):
        counts = np.ones((self.n_concepts, self.n_actions)) * smoothing
        for traj in trajectories:
            for step in traj:
                c = self.state_to_concept_index(step['state'])
                counts[c, step['action']] += 1
        self.policy_table = counts / counts.sum(axis=1, keepdims=True)

    def prob(self, state, action) -> float:
        return float(self.policy_table[
            self.state_to_concept_index(state), action])


class GatedPolicy:
    def __init__(self, gated_extractor, n_concepts=32, n_actions=4):
        self.gated_extractor = gated_extractor
        self.n_concepts      = n_concepts
        self.n_actions       = n_actions
        self.policy_table    = np.ones((n_concepts, n_actions)) / n_actions
        self.global_policy   = np.ones(n_actions) / n_actions

    def state_to_concept_index(self, state) -> int:
        concept_vec = self.gated_extractor(state)
        if len(concept_vec) > 5:
            concept_vec = concept_vec[:5]
        binary = (concept_vec > 0.5).astype(int)
        return int(sum(b * (2**i) for i, b in enumerate(binary)))

    def learn_from_trajectories(self, trajectories, smoothing=1.0):
        counts = np.ones((self.n_concepts, self.n_actions)) * smoothing
        for traj in trajectories:
            for step in traj:
                c = self.state_to_concept_index(step['state'])
                counts[c, step['action']] += 1
        self.policy_table = counts / counts.sum(axis=1, keepdims=True)
        total = counts.sum(axis=0)
        self.global_policy = total / total.sum()

    def prob(self, state, action) -> float:
        gate  = self.gated_extractor.get_gate_value(state)
        c     = self.state_to_concept_index(state)
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
        traj = collect_trajectory(env, eval_policy, max_steps=H)
        returns.append(sum((gamma**t)*s['reward']
                           for t, s in enumerate(traj)))
    return float(np.mean(returns))


# =============================================================================
# MAIN EXPERIMENT
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
        early = [s for i, s in enumerate(traj) if i < train_horizon]
        if early:
            train_trajs.append(early)

    # Train all concept models
    print("  Training soft concepts...")
    soft = SoftConcepts(env, use_leakage=True, seed=seed)
    soft.train_on_trajectories(train_trajs, hard, epochs=200)

    print("  Training standard conformal...")
    conf = ConformalGatedConcepts(env, seed=seed)
    conf.train_on_trajectories(train_trajs, hard, epochs=200)

    print("  Training per-concept conformal...")
    pcconf = PerConceptConformalGating(env, seed=seed)
    pcconf.train_on_trajectories(train_trajs, hard, epochs=200)

    # Build policies
    hard_b  = ConceptBasedPolicy(hard,   32, 4)
    soft_b  = ConceptBasedPolicy(soft,   32, 4)
    conf_b  = GatedPolicy(conf,          32, 4)
    pcconf_b= GatedPolicy(pcconf,        32, 4)

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

    results = {k: {'estimates':[], 'errors':[], 'variances':[]}
               for k in ['hard','soft','conf','pcconf']}
    results['horizons']     = test_horizons
    results['true_values']  = []

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
            err = abs(est - tv)
            results[key]['estimates'].append(est)
            results[key]['errors'].append(err)
            results[key]['variances'].append(var)

    return results


def run_multiseed(n_seeds=30, test_horizons=None):
    if test_horizons is None:
        test_horizons = [5, 10, 15, 20, 25, 30, 35, 40]

    keys = ['hard', 'soft', 'conf', 'pcconf']
    all_est  = {k: [] for k in keys}
    all_true = []

    for seed in range(n_seeds):
        print(f"\n{'='*50}")
        print(f"SEED {seed+1}/{n_seeds}")
        r = run_experiment(seed=seed, test_horizons=test_horizons)
        all_true.append(r['true_values'])
        for k in keys:
            all_est[k].append(r[k]['estimates'])

    true_vals = np.mean(all_true, axis=0)

    print(f"\n{'='*65}")
    print(f"MULTI-SEED RESULTS ({n_seeds} seeds)")
    print(f"{'='*65}")
    labels = {'hard':'Hard CPDIS','soft':'Soft CPDIS',
              'conf':'Conformal (1 gate)',
              'pcconf':'Per-Concept Conformal (5 gates)'}

    print(f"\n{'H':<5} {'Estimator':<30} {'MSE':<10} {'Bias':<10} "
          f"{'Variance':<12}")
    print("-"*70)

    for k in keys:
        ests = np.array(all_est[k])
        for i, H in enumerate(test_horizons):
            tv   = true_vals[i]
            mse  = float(np.mean((ests[:,i]-tv)**2))
            bias = float(np.mean(ests[:,i])-tv)
            var  = float(np.var(ests[:,i]))
            print(f"{H:<5} {labels[k]:<30} {mse:<10.4f} "
                  f"{bias:<10.4f} {var:<12.4f}")
        print()

    # Key comparison at H=40
    print(f"\n{'='*65}")
    print("KEY COMPARISON AT H=40 (30 seeds)")
    print(f"{'='*65}")
    print(f"{'Estimator':<30} {'MSE':<10} {'Bias':<10} {'Variance':<12} "
          f"{'vs Soft MSE'}")
    print("-"*65)
    soft_mse_40 = None
    for k in keys:
        ests = np.array(all_est[k])
        tv   = true_vals[-1]
        mse  = float(np.mean((ests[:,-1]-tv)**2))
        bias = float(np.mean(ests[:,-1])-tv)
        var  = float(np.var(ests[:,-1]))
        if k == 'soft':
            soft_mse_40 = mse
        reduction = f"{(soft_mse_40-mse)/soft_mse_40*100:.1f}% better" \
                    if soft_mse_40 and k != 'soft' else "—"
        print(f"{labels[k]:<30} {mse:<10.4f} {bias:<10.4f} "
              f"{var:<12.4f} {reduction}")

    return {'all_est': all_est, 'true_vals': true_vals,
            'horizons': test_horizons, 'n_seeds': n_seeds}


# =============================================================================
# PER-CONCEPT GATE BEHAVIOUR ANALYSIS
# =============================================================================

def analyse_per_concept_gates(env, hard, pcconf,
                               trajectories, train_horizon=10):
    """
    Show per-concept gate values over time.
    Which concepts close early? Which stay open?
    """
    ts = list(range(0, 43, 2))
    gates_by_t = {t: {i: [] for i in range(5)} for t in ts}

    for traj in trajectories:
        for t, step in enumerate(traj):
            if t in gates_by_t:
                s     = step['state']
                gates = pcconf._per_concept_gates(s)
                for i in range(5):
                    gates_by_t[t][i].append(gates[i])

    print(f"\n{'='*75}")
    print("PER-CONCEPT GATE VALUES OVER TIME")
    print(f"{'='*75}")
    print(f"{'t':<6} {'in-dist':<10}", end="")
    for name in concept_names:
        print(f" {name[:10]:<13}", end="")
    print()
    print("-"*80)

    for t in ts:
        if not gates_by_t[t][0]:
            continue
        in_dist = "Yes" if t < train_horizon else "No"
        print(f"{t:<6} {in_dist:<10}", end="")
        for i in range(5):
            mean_g = np.mean(gates_by_t[t][i])
            print(f" {mean_g:<13.3f}", end="")
        print()

    return gates_by_t


# =============================================================================
# PLOTS
# =============================================================================

def plot_comparison(results, save_path=None):
    """Compare all 4 estimators: error vs horizon."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5),
                             gridspec_kw={'wspace':0.32})

    H     = results['horizons']
    cols  = {'hard':'#27ae60','soft':'#e74c3c',
             'conf':'#8e44ad','pcconf':'#f39c12'}
    marks = {'hard':'s','soft':'^','conf':'D','pcconf':'o'}
    labs  = {'hard':'Hard CPDIS (oracle)',
             'soft':'Soft CPDIS',
             'conf':'Conformal (1 gate)',
             'pcconf':'Per-Concept Conformal (5 gates)'}
    TRAIN_H = 10

    for ax_idx, metric in enumerate(['errors','variances']):
        ax = axes[ax_idx]
        ax.axvspan(min(H)-2, TRAIN_H,    color='#e8f5e9', alpha=0.7, zorder=0)
        ax.axvspan(TRAIN_H, max(H)+2,    color='#fce4e4', alpha=0.7, zorder=0)
        ax.axvline(x=TRAIN_H, color='#7f8c8d', ls='--', lw=1.4, zorder=2)

        for k in ['hard','soft','conf','pcconf']:
            ax.plot(H, results[k][metric],
                    color=cols[k], marker=marks[k],
                    linewidth=2.0, markersize=6,
                    label=labs[k], zorder=3)

        title = 'OPE Absolute Error' if metric=='errors' else 'Variance'
        ax.set_title(f'{title} vs Horizon', fontsize=12)
        ax.set_xlabel('Test Horizon H', fontsize=11)
        ax.set_ylabel(title, fontsize=11)
        ax.set_xlim(min(H)-2, max(H)+2)
        ax.legend(fontsize=9, loc='upper left')

    fig.suptitle('Per-Concept Conformal vs Standard Conformal\n'
                 'Does concept-level gating improve OPE?',
                 fontsize=13, fontweight='bold')

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    return fig


def plot_gate_values(gates_by_t, train_horizon=10, save_path=None):
    """Show per-concept gate values over time."""
    ts     = sorted(gates_by_t.keys())
    colors = ['#e74c3c','#2980b9','#27ae60','#f39c12','#8e44ad']

    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.axvspan(-0.5, train_horizon,  color='#e8f5e9', alpha=0.7, zorder=0)
    ax.axvspan(train_horizon, max(ts)+0.5, color='#fce4e4', alpha=0.7, zorder=0)
    ax.axvline(x=train_horizon, color='#7f8c8d', ls='--', lw=1.5,
               label=f'Train horizon (t={train_horizon})', zorder=2)

    for i, (name, color) in enumerate(zip(concept_names, colors)):
        vals = [np.mean(gates_by_t[t][i])
                if gates_by_t[t][i] else np.nan for t in ts]
        ts_v = [t for t, v in zip(ts, vals) if not np.isnan(v)]
        val_v = [v for v in vals if not np.isnan(v)]
        ax.plot(ts_v, val_v, color=color, marker='o',
                linewidth=2.0, markersize=5, label=name, zorder=3)

        # Annotate final value
        if val_v:
            ax.annotate(f'{val_v[-1]:.2f}',
                        xy=(ts_v[-1], val_v[-1]),
                        xytext=(4, 0), textcoords='offset points',
                        fontsize=8, color=color, fontweight='bold',
                        clip_on=False)

    ax.set_xlabel('Trajectory Timestep t', fontsize=11)
    ax.set_ylabel('Gate Value (1=open, 0=closed)', fontsize=11)
    ax.set_title('Per-Concept Gate Values Over Time\n'
                 'Does each concept gate correctly detect OOD?',
                 fontsize=12)
    ax.set_ylim(-0.05, 1.1)
    ax.set_xlim(-0.5, max(ts)+0.5)
    ax.legend(fontsize=10, loc='lower left')
    ax.text(train_horizon/2, 1.07, 'In-dist',
            ha='center', fontsize=9, color='#1a6b35', style='italic')
    ax.text((train_horizon+max(ts))/2, 1.07, 'Out-of-distribution',
            ha='center', fontsize=9, color='#a93226', style='italic')

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

    # ── Single seed — quick look first ───────────────────────────────────────
    print("Running single seed experiment (seed=42)...")
    results = run_experiment(seed=42, test_horizons=[5,10,15,20,25,30,35,40])

    print(f"\n{'='*65}")
    print("SINGLE SEED RESULTS (seed=42)")
    print(f"{'='*65}")
    print(f"{'H':<5} {'Hard err':<12} {'Soft err':<12} "
          f"{'Conf err':<12} {'PCConf err':<12}")
    print("-"*55)
    for i, H in enumerate(results['horizons']):
        print(f"{H:<5} "
              f"{results['hard']['errors'][i]:<12.4f} "
              f"{results['soft']['errors'][i]:<12.4f} "
              f"{results['conf']['errors'][i]:<12.4f} "
              f"{results['pcconf']['errors'][i]:<12.4f}")

    # ── Gate behaviour analysis ───────────────────────────────────────────────
    set_seed(42)
    env  = WindyGridworld()
    bp   = EpsilonGreedyPolicy(env, epsilon=0.4, seed=42)
    hard = HardConcepts(env)
    trajs = [collect_trajectory(env, bp, max_steps=50) for _ in range(500)]

    train_trajs = []
    for traj in trajs[:200]:
        early = [s for i, s in enumerate(traj) if i < 10]
        if early:
            train_trajs.append(early)

    pcconf = PerConceptConformalGating(env, seed=42)
    pcconf.train_on_trajectories(train_trajs, hard, epochs=200)

    gates_by_t = analyse_per_concept_gates(
        env, hard, pcconf, trajs, train_horizon=10)

    # ── Plots ─────────────────────────────────────────────────────────────────
    print("\nGenerating plots...")
    plot_comparison(
        results,
        save_path=os.path.join(results_dir,
                               'per_concept_conformal_comparison.png'))

    plot_gate_values(
        gates_by_t, train_horizon=10,
        save_path=os.path.join(results_dir,
                               'per_concept_gate_values.png'))

    plt.show()

    # ── Multi-seed (uncomment for full validation ~30 min) ───────────────────
    # print("\nRunning 30-seed experiment...")
    # multi = run_multiseed(n_seeds=30)
    # np.save(os.path.join(results_dir,'per_concept_multiseed.npy'), multi)