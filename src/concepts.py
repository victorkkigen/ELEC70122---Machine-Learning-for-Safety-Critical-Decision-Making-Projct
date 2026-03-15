"""
Concept Extractors for Windy Gridworld

Includes:
- HardConcepts: Rule-based binary concepts (no leakage)
- SoftConcepts: Neural network concept encoder (can leak)
- GatedSoftConcepts: Entropy-gated soft concepts (MixCEM adaptation)
- ConformalGatedConcepts: Conformal prediction-based gating (novel)
- measure_leakage: Probe to measure information leakage in embeddings
"""

import numpy as np
from typing import Tuple, List, Dict, Optional


class HardConcepts:
    """
    Rule-based binary concepts with no information leakage.

    Concepts:
    - near_goal:    Manhattan distance to goal <= 2
    - high_wind:    Current column has wind >= 2
    - in_left_half: Column < 5
    - in_top_half:  Row < 3 (closer to top)
    - near_start:   Manhattan distance to start <= 2

    Satisfies all four concept desiderata:
    explainability, conciseness, diversity, and coverage.
    Under known concepts, CPDIS is unbiased (Theorem 4.3).
    """

    def __init__(self, env):
        self.env = env
        self.n_concepts = 5
        self.concept_names = [
            'near_goal', 'high_wind', 'in_left_half',
            'in_top_half', 'near_start'
        ]

    def extract(self, state: Tuple[int, int]) -> np.ndarray:
        """Extract binary concept vector from state."""
        row, col = state

        goal_dist  = abs(row - self.env.goal[0])  + abs(col - self.env.goal[1])
        near_goal  = float(goal_dist <= 2)
        high_wind  = float(self.env.wind[col] >= 2)
        in_left    = float(col < 5)
        in_top     = float(row < 3)
        start_dist = abs(row - self.env.start[0]) + abs(col - self.env.start[1])
        near_start = float(start_dist <= 2)

        return np.array([near_goal, high_wind, in_left, in_top, near_start])

    def __call__(self, state: Tuple[int, int]) -> np.ndarray:
        return self.extract(state)

    def extract_from_index(self, idx: int) -> list:
        """
        Decode concept bin index (0-31) back to concept names.
        Used in concept-variance interpretability analysis.

        Args:
            idx: Integer index 0-31

        Returns:
            List of active concept names for this bin
        """
        names = []
        for i in range(5):
            if (idx >> i) & 1:
                names.append(self.concept_names[i])
        return names if names else ['none']


class SoftConceptEncoder:
    """
    Simple MLP encoder from state features to concept probabilities.

    Trained with binary cross-entropy (output loss only).
    Does NOT implement interpretability (L1) or diversity (cosine) losses
    from Algorithm 1 — those require PyTorch autograd (out of scope).

    Limitation: binary cross-entropy pushes outputs to 0/1 making
    the network overconfident. This causes entropy gating to fail
    (gate barely closes at OOD states). See ConformalGatedConcepts.
    """

    def __init__(self, input_dim: int = 8, hidden_dim: int = 32,
                 output_dim: int = 5, seed: int = None):
        self.input_dim  = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        rng = np.random.RandomState(seed)

        self.W1 = rng.randn(input_dim, hidden_dim)  * np.sqrt(2.0 / input_dim)
        self.b1 = np.zeros(hidden_dim)
        self.W2 = rng.randn(hidden_dim, hidden_dim) * np.sqrt(2.0 / hidden_dim)
        self.b2 = np.zeros(hidden_dim)
        self.W3 = rng.randn(hidden_dim, output_dim) * np.sqrt(2.0 / hidden_dim)
        self.b3 = np.zeros(output_dim)

    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Forward pass.
        Returns: (probs, hidden) where probs are sigmoid concept probabilities
        and hidden are the final hidden layer activations (for leakage analysis).
        """
        h1     = np.maximum(0, x @ self.W1 + self.b1)
        h2     = np.maximum(0, h1 @ self.W2 + self.b2)
        logits = h2 @ self.W3 + self.b3
        probs  = 1 / (1 + np.exp(-logits))
        return probs, h2

    def train_step(self, x: np.ndarray, y: np.ndarray,
                   lr: float = 0.01) -> float:
        """Single gradient descent step. Loss: binary cross-entropy."""
        batch_size = x.shape[0]

        h1     = np.maximum(0, x @ self.W1 + self.b1)
        h2     = np.maximum(0, h1 @ self.W2 + self.b2)
        logits = h2 @ self.W3 + self.b3
        probs  = 1 / (1 + np.exp(-np.clip(logits, -20, 20)))

        eps  = 1e-7
        loss = -np.mean(y * np.log(probs + eps) +
                        (1 - y) * np.log(1 - probs + eps))

        d_logits = (probs - y) / batch_size
        d_W3 = h2.T @ d_logits;   d_b3 = np.sum(d_logits, axis=0)
        d_h2 = (d_logits @ self.W3.T) * (h2 > 0)
        d_W2 = h1.T @ d_h2;       d_b2 = np.sum(d_h2, axis=0)
        d_h1 = (d_h2 @ self.W2.T) * (h1 > 0)
        d_W1 = x.T @ d_h1;        d_b1 = np.sum(d_h1, axis=0)

        self.W3 -= lr * d_W3;  self.b3 -= lr * d_b3
        self.W2 -= lr * d_W2;  self.b2 -= lr * d_b2
        self.W1 -= lr * d_W1;  self.b1 -= lr * d_b1

        return loss


class SoftConcepts:
    """
    Neural network-based soft concept extractor.

    Trained to predict hard concept labels (output loss only).
    Satisfies explainability and coverage desiderata but fails
    conciseness (no L1 loss) and diversity (no cosine loss).

    Failing conciseness → encoder leaks raw state info →
    Assumption 4.2 violated at OOD states →
    Theorem 5.1 bias grows → OPE error compounds.

    Returns:
    - use_leakage=True:  [probs, hidden] (37 dims) — with leakage
    - use_leakage=False: probs           (5 dims)  — probs only
    """

    def __init__(self, env, use_leakage: bool = True,
                 hidden_dim: int = 32, seed: int = None):
        self.env         = env
        self.use_leakage = use_leakage
        self.n_concepts  = 5

        sample_features = env.state_to_features((0, 0))
        self.input_dim  = len(sample_features)

        self.encoder = SoftConceptEncoder(
            input_dim=self.input_dim,
            hidden_dim=hidden_dim,
            output_dim=self.n_concepts,
            seed=seed
        )

    def extract(self, state: Tuple[int, int]) -> np.ndarray:
        features      = self.env.state_to_features(state)
        probs, hidden = self.encoder.forward(features)
        if self.use_leakage:
            return np.concatenate([probs, hidden])
        else:
            return probs

    def __call__(self, state: Tuple[int, int]) -> np.ndarray:
        return self.extract(state)

    def train_on_trajectories(self, trajectories: List[List[Dict]],
                              hard_concepts: HardConcepts,
                              epochs: int = 100, lr: float = 0.01,
                              verbose: bool = False):
        """Train encoder to predict hard concept labels."""
        X, Y = [], []
        for traj in trajectories:
            for step in traj:
                state    = step['state']
                features = step.get('features',
                                    self.env.state_to_features(state))
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


def measure_leakage(concept_extractor, states: np.ndarray,
                    features: np.ndarray) -> float:
    """
    Measure information leakage using a linear probe.
    R² > 0: embeddings encode raw state info (leakage)
    R² < 0: OOD degradation (worse than mean prediction)

    Operationalises Assumption 4.2 violation:
    R²(t) collapse ↔ pi^c_b != pi_b ↔ Theorem 5.1 bias grows
    """
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import cross_val_score

    embeddings = []
    for state in states:
        if isinstance(state, np.ndarray):
            state = tuple(state)
        embeddings.append(concept_extractor(state))
    embeddings = np.array(embeddings)

    probe = Ridge(alpha=1.0)
    try:
        scores = cross_val_score(
            probe, embeddings, features,
            cv=min(5, len(embeddings) // 10 + 1),
            scoring='r2')
        return max(0, scores.mean())
    except:
        return 0.0


def train_probe(concept_extractor, states, features):
    """
    Train a linear probe on in-distribution states.
    Used in Experiment 1 to measure temporal leakage degradation.
    Returns fitted Ridge regression probe.
    """
    from sklearn.linear_model import Ridge

    embeddings = []
    for state in states:
        if isinstance(state, np.ndarray):
            state = tuple(state)
        embeddings.append(concept_extractor(state))

    embeddings = np.array(embeddings)
    features   = np.array(features)
    probe      = Ridge(alpha=1.0)
    probe.fit(embeddings, features)
    return probe


def evaluate_probe(probe, concept_extractor, states, features):
    """
    Evaluate trained probe on new data (can be OOD).
    Returns R² — can be negative for OOD states.
    Negative R² = concepts encode garbage for these states.
    """
    embeddings = []
    for state in states:
        if isinstance(state, np.ndarray):
            state = tuple(state)
        embeddings.append(concept_extractor(state))

    embeddings = np.array(embeddings)
    features   = np.array(features)
    return probe.score(embeddings, features)


class GatedSoftConcepts:
    """
    Entropy-gated soft concept extractor.
    Adapts MixCEM (Zarlenga et al., ICML 2025) to sequential OPE.

    Formula (MixCEM equation 23 equivalent):
        gate  = 1 - H(probs) / log(2)
        c_out = global_mean + gate * (probs - global_mean)

    MixCEM mapping:
        gate(s)      ↔  κ(h_t, c_t)   automatic vs manual
        probs         ↔  c_t
        global_mean   ↔  c^int_t       training mean vs oracle

    Limitation: binary cross-entropy training → overconfident network
    → entropy stays low even at OOD → gate barely closes (Δ=0.062).
    MixCEM uses Platt scaling to fix this — we use conformal instead.
    """

    def __init__(self, env, hidden_dim: int = 32, seed: int = None):
        self.env        = env
        self.n_concepts = 5

        sample_features = env.state_to_features((0, 0))
        self.input_dim  = len(sample_features)

        self.encoder = SoftConceptEncoder(
            input_dim=self.input_dim,
            hidden_dim=hidden_dim,
            output_dim=self.n_concepts,
            seed=seed
        )
        self.global_mean = np.ones(self.n_concepts) * 0.5

    def _probs(self, state: Tuple[int, int]) -> np.ndarray:
        features = self.env.state_to_features(state)
        probs, _ = self.encoder.forward(features)
        return probs

    def _gate(self, probs: np.ndarray) -> float:
        """
        gate = 1 - normalised_binary_entropy(probs)
        In-dist: low H → gate≈1. OOD: high H → gate≈0.
        """
        eps = 1e-7
        p   = np.clip(probs, eps, 1 - eps)
        h   = -(p * np.log(p) + (1 - p) * np.log(1 - p))
        return float(1.0 - np.mean(h) / np.log(2))

    def extract(self, state: Tuple[int, int]) -> np.ndarray:
        probs = self._probs(state)
        gate  = self._gate(probs)
        return self.global_mean + gate * (probs - self.global_mean)

    def get_gate_value(self, state: Tuple[int, int]) -> float:
        """Gate value in [0,1]. 1=open (in-dist), 0=closed (OOD)."""
        return self._gate(self._probs(state))

    def __call__(self, state: Tuple[int, int]) -> np.ndarray:
        return self.extract(state)

    def train_on_trajectories(self, trajectories: List[List[Dict]],
                              hard_concepts: HardConcepts,
                              epochs: int = 100, lr: float = 0.01,
                              verbose: bool = False):
        """Train encoder, compute global mean over training states."""
        X, Y = [], []
        for traj in trajectories:
            for step in traj:
                state    = step['state']
                features = step.get('features',
                                    self.env.state_to_features(state))
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

        all_probs = []
        for x in X:
            probs, _ = self.encoder.forward(x)
            all_probs.append(probs)
        self.global_mean = np.mean(all_probs, axis=0)

        if verbose:
            print(f"    Global mean: {self.global_mean.round(3)}")


class ConformalGatedConcepts:
    """
    Conformal prediction-based gating for soft concepts.

    Novel contribution: distribution-free automatic temporal intervention.
    Equivalent to paper intervention formula (eq. 23) but fully automatic:
        gate(s)      ↔  κ(h_t, c_t)   OOD score replaces manual weight
        global_mean  ↔  c^int_t        training mean replaces oracle

    Algorithm:
    1. Calibrate on in-dist states (t < train_horizon):
       score = |hard(s) - soft(s)|
       threshold = 95th percentile of calibration scores
    2. At inference:
       score <= threshold → gate = 1.0   (in-distribution)
       score >  threshold → gate closes  (OOD)

    Why better than entropy gating (Section 4 proof):
    - Gate spread: Δ=0.658 vs Δ=0.062 for entropy
    - At t=25: 54.8% weight on global_mean vs 11.4% for entropy
    - global_mean IS ratios identical for all OOD states
      → zero covariance contribution → Theorem 4.4 partially satisfied
    - Result: 80% OPE error reduction vs 6% for entropy
    """

    def __init__(self, env, seed: int = None):
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
        self.global_mean        = np.ones(self.n_concepts) * 0.5
        self.calibration_scores = np.array([])
        self._hard_concepts     = None

    def _soft_probs(self, state: Tuple[int, int]) -> np.ndarray:
        features = self.env.state_to_features(state)
        probs, _ = self.encoder.forward(features)
        return probs

    def _conformal_score(self, state: Tuple[int, int],
                         hard_concepts) -> float:
        """
        Nonconformity score = mean |hard(s) - soft(s)|.
        Small = in-distribution. Large = OOD.
        """
        soft = self._soft_probs(state)
        hard = hard_concepts.extract(state)
        return float(np.mean(np.abs(soft - hard)))

    def calibrate(self, states, hard_concepts):
        """Calibrate on in-distribution training states."""
        scores = [self._conformal_score(s, hard_concepts) for s in states]
        self.calibration_scores = np.array(scores)
        print(f"    Conformal calibration: {len(scores)} states, "
              f"score mean={np.mean(scores):.3f}, "
              f"95th pct={np.percentile(scores, 95):.3f}")

    def get_gate_value(self, state: Tuple[int, int],
                       hard_concepts=None) -> float:
        """
        Gate value in [0,1].
        1.0  = in-distribution (score <= 95th percentile)
        →0.0 = OOD (score >> threshold)
        """
        if len(self.calibration_scores) == 0:
            return 1.0

        score     = self._conformal_score(state, self._hard_concepts)
        threshold = np.percentile(self.calibration_scores, 95)

        if score <= threshold:
            return 1.0
        else:
            excess = (score - threshold) / (threshold + 1e-7)
            return float(max(0.0, 1.0 / (1.0 + excess)))

    def extract(self, state: Tuple[int, int]) -> np.ndarray:
        gate  = self.get_gate_value(state)
        probs = self._soft_probs(state)
        return self.global_mean + gate * (probs - self.global_mean)

    def __call__(self, state: Tuple[int, int]) -> np.ndarray:
        return self.extract(state)

    def train_on_trajectories(self, trajectories: List[List[Dict]],
                              hard_concepts: HardConcepts,
                              epochs: int = 100, lr: float = 0.01,
                              verbose: bool = False):
        """Train encoder, compute global mean, then calibrate."""
        self._hard_concepts = hard_concepts

        X, Y       = [], []
        cal_states = []

        for traj in trajectories:
            for step in traj:
                state    = step['state']
                features = step.get('features',
                                    self.env.state_to_features(state))
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
                print(f"    Epoch {epoch+1}/{epochs}, "
                      f"Loss: {total_loss/n_batches:.4f}")

        all_probs = []
        for x in X:
            probs, _ = self.encoder.forward(x)
            all_probs.append(probs)
        self.global_mean = np.mean(all_probs, axis=0)

        self.calibrate(cal_states, hard_concepts)