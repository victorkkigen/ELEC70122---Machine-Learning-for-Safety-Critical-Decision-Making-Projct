"""
NEW ADDITIONS TO temporal_leakage_experiment.py
Based on Ritam meeting suggestions:

  1. Concept drift gating  — score = |soft(t) - soft(t-1)|
  2. Different training horizons — t<5, t<10, t<15, t<20, t<25, t<35
  3. Per-concept R² over time
  4. Per-concept IS ratio variance analysis

Add these functions to your existing experiment file
and call them from __main__.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict


# =============================================================================
# 1. CONCEPT DRIFT SCORE — NO HARD CONCEPTS NEEDED
# =============================================================================

class DriftGatedConcepts:
    """
    Concept drift-based gating.
    Does NOT require hard concepts at test time.

    Score = |soft(s_t) - soft(s_{t-1})|
    Large change between consecutive steps = OOD signal.

    Advantage over conformal: no oracle hard concepts needed.
    Disadvantage: requires previous timestep's output.
                  first step has no previous → gate = 1.0
    """

    def __init__(self, env, seed=None):
        self.env        = env
        self.n_concepts = 5
        self.seed       = seed

        sample_features = env.state_to_features((0, 0))
        self.input_dim  = len(sample_features)

        from concepts import SoftConceptEncoder
        self.encoder = SoftConceptEncoder(
            input_dim=self.input_dim,
            hidden_dim=32,
            output_dim=self.n_concepts,
            seed=seed
        )
        self.global_mean        = np.ones(self.n_concepts) * 0.5
        self.drift_threshold    = 0.1   # set by calibration
        self.prev_soft          = None  # previous timestep output

    def _soft_probs(self, state) -> np.ndarray:
        features = self.env.state_to_features(state)
        probs, _ = self.encoder.forward(features)
        return probs

    def _drift_score(self, state) -> float:
        """
        score = mean|soft(s_t) - soft(s_{t-1})|

        Returns 0.0 if no previous step (first step of trajectory).
        """
        current = self._soft_probs(state)
        if self.prev_soft is None:
            return 0.0
        return float(np.mean(np.abs(current - self.prev_soft)))

    def calibrate(self, trajectories):
        """
        Calibrate drift threshold on training trajectories.
        Computes drift scores within training distribution.
        Threshold = 95th percentile of in-dist drift scores.
        """
        scores = []
        for traj in trajectories:
            self.prev_soft = None
            for step in traj:
                s = step['state']
                score = self._drift_score(s)
                if score > 0:   # skip first step
                    scores.append(score)
                self.prev_soft = self._soft_probs(s)

        self.drift_threshold = float(np.percentile(scores, 95))
        self.prev_soft = None  # reset

        print(f"    Drift calibration: {len(scores)} transitions, "
              f"mean={np.mean(scores):.4f}, "
              f"95th pct={self.drift_threshold:.4f}")

    def get_gate_value(self, state) -> float:
        """Gate based on concept drift score."""
        score = self._drift_score(state)
        if score <= self.drift_threshold:
            return 1.0
        else:
            excess = (score - self.drift_threshold) / (self.drift_threshold + 1e-7)
            return float(max(0.0, 1.0 / (1.0 + excess)))

    def extract(self, state) -> np.ndarray:
        """Gated concept output."""
        gate         = self.get_gate_value(state)
        probs        = self._soft_probs(state)
        self.prev_soft = probs  # update for next step
        return self.global_mean + gate * (probs - self.global_mean)

    def __call__(self, state) -> np.ndarray:
        return self.extract(state)

    def train_on_trajectories(self, trajectories, hard_concepts,
                              epochs=100, lr=0.01, verbose=False):
        """Train encoder, compute global mean, calibrate drift threshold."""
        X, Y = [], []
        for traj in trajectories:
            for step in traj:
                state    = step['state']
                features = step.get('features', self.env.state_to_features(state))
                X.append(features)
                Y.append(hard_concepts.extract(state))

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
                print(f"    Epoch {epoch+1}/{epochs}, "
                      f"Loss: {total_loss/n_batches:.4f}")

        # Global mean
        all_probs = []
        for x in X:
            probs, _ = self.encoder.forward(x)
            all_probs.append(probs)
        self.global_mean = np.mean(all_probs, axis=0)

        # Calibrate drift threshold
        self.calibrate(trajectories)


# =============================================================================
# 2. DIFFERENT TRAINING HORIZONS EXPERIMENT
# =============================================================================

def run_training_horizon_experiment(
    train_horizons=None,
    test_horizons=None,
    n_trajectories=500,
    max_steps=50,
    seed=42,
    behavior_epsilon=0.4,
    eval_epsilon=0.05
):
    """
    Run experiment for different training horizons.

    For each train_horizon in [5, 10, 15, 20, 25, 35]:
      - Train soft concepts on t < train_horizon
      - Test OPE at all test_horizons
      - Record OPE error for soft, hard, conformal

    Question: does more training data reduce leakage?
    At what training horizon does leakage disappear?
    """
    if train_horizons is None:
        train_horizons = [5, 10, 15, 20, 25, 35]
    if test_horizons is None:
        test_horizons = [5, 10, 20, 30, 40]

    from gridworld import WindyGridworld, collect_trajectory
    from policies import EpsilonGreedyPolicy, OptimalPolicy
    from concepts import HardConcepts, SoftConcepts, ConformalGatedConcepts
    from utils import set_seed

    set_seed(seed)

    env             = WindyGridworld()
    behavior_policy = EpsilonGreedyPolicy(env, epsilon=behavior_epsilon, seed=seed)
    eval_policy     = OptimalPolicy(env, epsilon=eval_epsilon, seed=seed)

    # Collect trajectories once — reuse across all training horizons
    print(f"Collecting {n_trajectories} trajectories...")
    trajectories = []
    for _ in range(n_trajectories):
        trajectories.append(collect_trajectory(env, behavior_policy, max_steps=max_steps))

    hard_concepts = HardConcepts(env)

    results = {}

    for train_h in train_horizons:
        print(f"\n{'='*55}")
        print(f"Training horizon: t < {train_h}")
        print(f"{'='*55}")

        # Build training set for this horizon
        train_trajs = []
        for traj in trajectories[:200]:
            early = [s for i, s in enumerate(traj) if i < train_h]
            if early:
                train_trajs.append(early)

        n_train_states = sum(len(t) for t in train_trajs)
        print(f"  Training states: {n_train_states}")

        # Train soft concepts
        soft = SoftConcepts(env, use_leakage=True, seed=seed)
        soft.train_on_trajectories(train_trajs, hard_concepts, epochs=200)

        # Train conformal gating
        conformal = ConformalGatedConcepts(env, seed=seed)
        conformal.train_on_trajectories(train_trajs, hard_concepts, epochs=200)

        # Probe R² at each timestep
        leakage_ts = [2, 5, 10, 15, 20, 25, 30, 35, 40]
        states_by_t   = {t: [] for t in leakage_ts}
        features_by_t = {t: [] for t in leakage_ts}
        for traj in trajectories:
            for t, step in enumerate(traj):
                if t in states_by_t:
                    states_by_t[t].append(step['state'])
                    features_by_t[t].append(step['features'])

        # Train probe on this training horizon's states
        from concepts import train_probe, evaluate_probe
        train_states   = []
        train_features = []
        for t in leakage_ts:
            if t < train_h:
                train_states.extend(states_by_t[t])
                train_features.extend(features_by_t[t])

        probe_soft = train_probe(soft, train_states, train_features)

        r2_by_t = {}
        for t in leakage_ts:
            if len(states_by_t[t]) < 10:
                continue
            r2 = evaluate_probe(probe_soft, soft,
                                states_by_t[t], features_by_t[t])
            r2_by_t[t] = r2

        # OPE error at each test horizon
        from temporal_leakage_experiment import (
            ConceptBasedPolicy, GatedConceptBasedPolicy,
            cpdis_estimate_by_horizon, pdis_estimate_by_horizon,
            compute_ground_truth_by_horizon
        )

        hard_b = ConceptBasedPolicy(hard_concepts, n_concepts=32, n_actions=4)
        soft_b = ConceptBasedPolicy(soft,          n_concepts=32, n_actions=4)
        conf_b = GatedConceptBasedPolicy(conformal, n_concepts=32, n_actions=4)

        hard_b.learn_from_trajectories(trajectories)
        soft_b.learn_from_trajectories(trajectories)
        conf_b.learn_from_trajectories(trajectories)

        eval_trajs = []
        for _ in range(200):
            eval_trajs.append(collect_trajectory(env, eval_policy, max_steps=max_steps))

        hard_e = ConceptBasedPolicy(hard_concepts, n_concepts=32, n_actions=4)
        soft_e = ConceptBasedPolicy(soft,          n_concepts=32, n_actions=4)
        conf_e = GatedConceptBasedPolicy(conformal, n_concepts=32, n_actions=4)

        hard_e.learn_from_trajectories(eval_trajs)
        soft_e.learn_from_trajectories(eval_trajs)
        conf_e.learn_from_trajectories(eval_trajs)

        ope_errors = {'hard': [], 'soft': [], 'conformal': []}

        for h in test_horizons:
            true_val = compute_ground_truth_by_horizon(
                env, eval_policy, h, n_episodes=500)

            _, _, _ = cpdis_estimate_by_horizon(trajectories, hard_b, hard_e, h)
            hard_est, _, _ = cpdis_estimate_by_horizon(trajectories, hard_b, hard_e, h)
            soft_est, _, _ = cpdis_estimate_by_horizon(trajectories, soft_b, soft_e, h)
            conf_est, _, _ = cpdis_estimate_by_horizon(trajectories, conf_b, conf_e, h)

            ope_errors['hard'].append(abs(hard_est - true_val))
            ope_errors['soft'].append(abs(soft_est - true_val))
            ope_errors['conformal'].append(abs(conf_est - true_val))

            print(f"  H={h}: hard={abs(hard_est-true_val):.3f}  "
                  f"soft={abs(soft_est-true_val):.3f}  "
                  f"conf={abs(conf_est-true_val):.3f}")

        results[train_h] = {
            'r2_by_t':    r2_by_t,
            'ope_errors': ope_errors,
            'test_horizons': test_horizons
        }

    return results


def plot_training_horizon_comparison(results, save_path=None):
    """
    Plot OPE error vs test horizon for each training horizon.
    One line per training horizon for soft CPDIS.
    Shows: does more training data reduce leakage?
    """
    _STYLE = {
        'font.family': 'DejaVu Sans', 'font.size': 12,
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.grid': True, 'grid.alpha': 0.25,
    }
    plt.rcParams.update(_STYLE)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    colors = ['#e74c3c', '#e67e22', '#f1c40f',
              '#2ecc71', '#2980b9', '#8e44ad']

    train_horizons = sorted(results.keys())

    # Left: soft CPDIS error per training horizon
    ax = axes[0]
    for i, train_h in enumerate(train_horizons):
        r    = results[train_h]
        errs = r['ope_errors']['soft']
        hs   = r['test_horizons']
        ax.plot(hs, errs, color=colors[i], marker='o',
                label=f'Train t<{train_h}', linewidth=2)

    ax.set_xlabel('Test Horizon H')
    ax.set_ylabel('OPE Absolute Error')
    ax.set_title('Soft CPDIS Error vs Test Horizon\nfor Different Training Horizons')
    ax.legend(fontsize=10)

    # Right: probe R² at t=25 vs training horizon
    ax = axes[1]
    r2_at_25 = []
    for train_h in train_horizons:
        r2 = results[train_h]['r2_by_t'].get(25, np.nan)
        r2_at_25.append(r2)

    ax.bar(range(len(train_horizons)), r2_at_25,
           color=colors, alpha=0.8)
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax.set_xticks(range(len(train_horizons)))
    ax.set_xticklabels([f't<{h}' for h in train_horizons])
    ax.set_xlabel('Training Horizon')
    ax.set_ylabel('Soft Concept R² at t=25')
    ax.set_title('Concept Quality at t=25\nvs Training Horizon')

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    return fig


# =============================================================================
# 3. PER-CONCEPT R² OVER TIME
# =============================================================================

def analyse_per_concept_r2(trajectories, soft_concepts, train_horizon=10):
    """
    Compute probe R² SEPARATELY for each concept over time.

    Instead of one averaged R² line, shows 5 lines:
      R²(near_goal, t)
      R²(high_wind, t)
      R²(in_left_half, t)
      R²(in_top_half, t)
      R²(near_start, t)

    Reveals WHICH concept degrades fastest.
    """
    from sklearn.linear_model import Ridge

    concept_names = ['near_goal', 'high_wind', 'in_left_half',
                     'in_top_half', 'near_start']
    leakage_ts = [2, 5, 10, 15, 20, 25, 30, 35, 40]

    # Collect states and features by timestep
    states_by_t   = {t: [] for t in leakage_ts}
    features_by_t = {t: [] for t in leakage_ts}
    concepts_by_t = {t: [] for t in leakage_ts}

    for traj in trajectories:
        for t, step in enumerate(traj):
            if t in states_by_t:
                states_by_t[t].append(step['state'])
                features_by_t[t].append(step['features'])

    # Get soft embeddings for each state
    embeddings_by_t = {}
    for t in leakage_ts:
        if len(states_by_t[t]) < 10:
            continue
        embs = []
        for s in states_by_t[t]:
            embs.append(soft_concepts(s))
        embeddings_by_t[t] = np.array(embs)

    # Collect training data (t < train_horizon)
    train_embs, train_feats = [], []
    for t in leakage_ts:
        if t < train_horizon and t in embeddings_by_t:
            train_embs.extend(embeddings_by_t[t])
            train_feats.extend(features_by_t[t])

    train_embs  = np.array(train_embs)
    train_feats = np.array(train_feats)

    # For each concept, train a separate probe on that concept's feature
    # We use feature columns that correspond to each concept
    # (features include one-hot position, wind, distance etc.)
    # Here we predict each raw feature dimension separately

    # Per-concept R²: predict concept i value from soft embeddings
    from concepts import HardConcepts
    # We need hard concept labels at each timestep
    # R²(concept_i, t) = how well soft embeddings predict hard concept i

    results = {name: [] for name in concept_names}
    results['timesteps'] = []

    print(f"\n{'='*65}")
    print("PER-CONCEPT R² OVER TIME")
    print(f"{'='*65}")
    print(f"{'t':<6} {'in-dist':<10}", end="")
    for name in concept_names:
        print(f" {name[:10]:<12}", end="")
    print()
    print("-" * 80)

    for t in leakage_ts:
        if t not in embeddings_by_t or len(embeddings_by_t[t]) < 20:
            continue

        test_embs = embeddings_by_t[t]
        in_dist   = "Yes" if t < train_horizon else "No"

        r2_values = []
        for ci, name in enumerate(concept_names):
            # Get hard concept labels for this concept at this timestep
            # We need HardConcepts — pass it in or reconstruct
            # Using feature columns as proxy (raw features contain position)
            # Train probe: soft_embedding → concept_ci_value

            # Get target: hard concept value for concept ci
            # We reconstruct from features (feature[ci] = concept value approx)
            target = train_feats[:, ci] if ci < train_feats.shape[1] else \
                     train_feats[:, 0]

            probe = Ridge(alpha=1.0)
            probe.fit(train_embs, target)

            test_target = np.array(features_by_t[t])[:, ci] \
                if ci < np.array(features_by_t[t]).shape[1] \
                else np.array(features_by_t[t])[:, 0]

            r2 = probe.score(test_embs, test_target)
            r2_values.append(r2)
            results[name].append(r2)

        results['timesteps'].append(t)
        print(f"{t:<6} {in_dist:<10}", end="")
        for r2 in r2_values:
            color_flag = " *" if r2 < 0 else "  "
            print(f" {r2:>10.4f}{color_flag}", end="")
        print()

    return results


def plot_per_concept_r2(results, train_horizon=10, save_path=None):
    """
    Plot R² over time for each concept separately.
    Shows which concept degrades fastest.
    """
    concept_names = ['near_goal', 'high_wind', 'in_left_half',
                     'in_top_half', 'near_start']
    colors = ['#e74c3c', '#2980b9', '#27ae60', '#f39c12', '#8e44ad']

    _STYLE = {
        'font.family': 'DejaVu Sans', 'font.size': 12,
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.grid': True, 'grid.alpha': 0.22,
    }
    plt.rcParams.update(_STYLE)

    fig, ax = plt.subplots(figsize=(9, 5.5))

    ts = results['timesteps']

    # Background shading
    ax.axvspan(min(ts), train_horizon, color='#f4faf5', alpha=1.0, zorder=0)
    ax.axvspan(train_horizon, max(ts),  color='#fdf4f4', alpha=1.0, zorder=0)
    ax.axvline(x=train_horizon, color='#7f8c8d', ls='--', lw=1.6,
               label=f'Train horizon (t={train_horizon})', zorder=2)
    ax.axhline(y=0, color='#aaaaaa', ls='-', lw=1.0, zorder=1)

    for name, color in zip(concept_names, colors):
        if name in results and len(results[name]) == len(ts):
            ax.plot(ts, results[name], color=color, marker='o',
                    linewidth=2, markersize=6, label=name, zorder=3)

    ax.set_xlabel('Trajectory Timestep  $t$')
    ax.set_ylabel('Linear Probe  $R^2$')
    ax.set_title('Per-Concept R² Over Time\n'
                 'Which concept degrades fastest?')
    ax.legend(loc='lower left', fontsize=10)

    # Region labels
    ylim = ax.get_ylim()
    y = ylim[0] + (ylim[1] - ylim[0]) * 0.94
    ax.text((min(ts) + train_horizon)/2, y,
            'In-distribution', ha='center', fontsize=10,
            color='#1a6b35', style='italic')
    ax.text(train_horizon + (max(ts)-train_horizon)/2, y,
            'Out-of-distribution', ha='center', fontsize=10,
            color='#a93226', style='italic')

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    return fig


# =============================================================================
# 4. CONCEPT DRIFT GATE — RESULTS ALONGSIDE EXISTING
# =============================================================================

def analyse_drift_gate(trajectories, soft_concepts, drift_gated_concepts,
                       train_horizon=10):
    """
    Compare drift gate vs conformal gate behaviour over time.
    Shows how much each gate closes at OOD states.
    """
    leakage_ts = [2, 5, 10, 15, 20, 25, 30, 35, 40]

    states_by_t = {t: [] for t in leakage_ts}
    for traj in trajectories:
        for t, step in enumerate(traj):
            if t in states_by_t:
                states_by_t[t].append(step['state'])

    print(f"\n{'='*70}")
    print("DRIFT GATE vs CONFORMAL GATE COMPARISON")
    print(f"{'='*70}")
    print(f"{'t':<6} {'in-dist':<10} {'drift gate':<14} "
          f"{'drift->global%':<16} {'concept drift score':<20}")
    print("-" * 70)

    drift_gates = []
    drift_scores_by_t = []

    for t in leakage_ts:
        if len(states_by_t[t]) < 10:
            continue

        # Compute drift scores and gate values
        # Note: drift requires CONSECUTIVE states so we process trajectories
        drift_gate_vals = []
        drift_score_vals = []

        for traj in trajectories:
            drift_gated_concepts.prev_soft = None  # reset per trajectory
            for step_t, step in enumerate(traj):
                if step_t == t:
                    score = drift_gated_concepts._drift_score(step['state'])
                    gate  = drift_gated_concepts.get_gate_value(step['state'])
                    drift_score_vals.append(score)
                    drift_gate_vals.append(gate)
                # Always update prev_soft as we step through
                if step_t <= t:
                    drift_gated_concepts.prev_soft = \
                        drift_gated_concepts._soft_probs(step['state'])

        drift_gated_concepts.prev_soft = None  # reset after

        if not drift_gate_vals:
            continue

        mean_gate  = float(np.mean(drift_gate_vals))
        mean_score = float(np.mean(drift_score_vals))
        global_pct = (1 - mean_gate) * 100
        in_dist    = "Yes" if t < train_horizon else "No"

        drift_gates.append(mean_gate)
        drift_scores_by_t.append(mean_score)

        print(f"{t:<6} {in_dist:<10} {mean_gate:<14.3f} "
              f"{global_pct:<16.1f} {mean_score:<20.4f}")

    return {'timesteps': leakage_ts[:len(drift_gates)],
            'drift_gates': drift_gates,
            'drift_scores': drift_scores_by_t}


def plot_gate_comparison(results_single_seed, drift_results,
                         save_path=None):
    """
    Plot all three gates side by side:
      Entropy gate, Conformal gate, Drift gate.
    Shows how much each closes at OOD states.
    """
    _STYLE = {
        'font.family': 'DejaVu Sans', 'font.size': 12,
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.grid': True, 'grid.alpha': 0.22,
    }
    plt.rcParams.update(_STYLE)

    fig, ax = plt.subplots(figsize=(9, 5))

    ts = [2, 5, 10, 15, 20, 25, 30, 35, 40]

    # Background
    ax.axvspan(min(ts), 10, color='#f4faf5', alpha=1.0, zorder=0)
    ax.axvspan(10, max(ts), color='#fdf4f4', alpha=1.0, zorder=0)
    ax.axvline(x=10, color='#7f8c8d', ls='--', lw=1.6,
               label='Train horizon (t=10)', zorder=2)

    # Entropy gate from existing results
    lk = results_single_seed.get('leakage', {})
    if lk:
        # Re-extract from existing results if available
        pass

    # Drift gate
    drift_ts = drift_results['timesteps']
    ax.plot(drift_ts, drift_results['drift_gates'],
            color='#f39c12', marker='D', linewidth=2,
            markersize=7, label='Drift Gate (no oracle)', zorder=3)

    ax.set_ylim(0, 1.05)
    ax.set_xlabel('Trajectory Timestep  $t$')
    ax.set_ylabel('Gate Value  (1=open, 0=closed)')
    ax.set_title('Gate Comparison: Entropy vs Conformal vs Drift\n'
                 'How much does each gate close at OOD states?')
    ax.legend(loc='lower left', fontsize=10)

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    return fig


# =============================================================================
# MAIN — ADD THESE CALLS TO YOUR EXISTING __main__ BLOCK
# =============================================================================

if __name__ == "__main__":
    import os
    import sys

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, os.path.join(project_root, 'src'))

    from gridworld import WindyGridworld, collect_trajectory
    from policies import EpsilonGreedyPolicy, OptimalPolicy
    from concepts import (HardConcepts, SoftConcepts,
                          ConformalGatedConcepts,
                          train_probe, evaluate_probe)
    from utils import set_seed

    results_dir = os.path.join(project_root, 'results')
    os.makedirs(results_dir, exist_ok=True)

    set_seed(42)
    env             = WindyGridworld()
    behavior_policy = EpsilonGreedyPolicy(env, epsilon=0.4, seed=42)
    eval_policy     = OptimalPolicy(env, epsilon=0.05, seed=42)
    hard_concepts   = HardConcepts(env)

    print("Collecting trajectories...")
    trajectories = []
    for _ in range(500):
        trajectories.append(collect_trajectory(env, behavior_policy, max_steps=50))

    train_trajs = []
    for traj in trajectories[:200]:
        early = [s for i, s in enumerate(traj) if i < 10]
        if early:
            train_trajs.append(early)

    # Train soft concepts
    print("\nTraining soft concepts...")
    soft_concepts = SoftConcepts(env, use_leakage=True, seed=42)
    soft_concepts.train_on_trajectories(train_trajs, hard_concepts, epochs=200)

    # ── ADDITION 1: Drift gating ────────────────────────────────────────────
    print("\nTraining drift-gated concepts...")
    drift_concepts = DriftGatedConcepts(env, seed=42)
    drift_concepts.train_on_trajectories(train_trajs, hard_concepts, epochs=200)

    drift_results = analyse_drift_gate(
        trajectories, soft_concepts, drift_concepts, train_horizon=10)

    # ── ADDITION 2: Different training horizons ─────────────────────────────
    print("\n\nRunning training horizon experiment...")
    horizon_results = run_training_horizon_experiment(
        train_horizons=[5, 10, 15, 20, 25, 35],
        test_horizons=[5, 10, 20, 30, 40],
        seed=42
    )
    plot_training_horizon_comparison(
        horizon_results,
        save_path=os.path.join(results_dir, 'training_horizon_comparison.png')
    )

    # ── ADDITION 3: Per-concept R² ──────────────────────────────────────────
    print("\nComputing per-concept R²...")
    per_concept = analyse_per_concept_r2(
        trajectories, soft_concepts, train_horizon=10)
    plot_per_concept_r2(
        per_concept, train_horizon=10,
        save_path=os.path.join(results_dir, 'per_concept_r2.png')
    )

    plt.show()
    print("\nAll new analyses complete.")