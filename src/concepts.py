"""
Concept Extractors for Windy Gridworld

Includes:
- HardConcepts: Rule-based binary concepts (no leakage)
- SoftConcepts: Neural network concept encoder (can leak)
- measure_leakage: Probe to measure information leakage in embeddings
"""

import numpy as np
from typing import Tuple, List, Dict, Optional


class HardConcepts:
    """
    Rule-based binary concepts with no information leakage.
    
    Concepts:
    - near_goal: Manhattan distance to goal <= 2
    - high_wind: Current column has wind >= 2
    - in_left_half: Column < 5
    - in_top_half: Row < 3 (closer to top)
    - near_start: Manhattan distance to start <= 2
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
        
        # Concept 1: Near goal
        goal_dist = abs(row - self.env.goal[0]) + abs(col - self.env.goal[1])
        near_goal = float(goal_dist <= 2)
        
        # Concept 2: High wind
        high_wind = float(self.env.wind[col] >= 2)
        
        # Concept 3: In left half
        in_left = float(col < 5)
        
        # Concept 4: In top half
        in_top = float(row < 3)
        
        # Concept 5: Near start
        start_dist = abs(row - self.env.start[0]) + abs(col - self.env.start[1])
        near_start = float(start_dist <= 2)
        
        return np.array([near_goal, high_wind, in_left, in_top, near_start])
    
    def __call__(self, state: Tuple[int, int]) -> np.ndarray:
        return self.extract(state)
    
    def to_index(self, state: Tuple[int, int]) -> int:
        """Convert concepts to single index (0-31)."""
        concepts = self.extract(state)
        idx = 0
        for i, c in enumerate(concepts):
            idx += int(c) * (2 ** i)
        return idx


class SoftConceptEncoder:
    """
    Simple MLP encoder from state features to concept probabilities.
    
    This can leak information beyond the concepts if the hidden layers
    encode more than just the concept labels.
    """
    
    def __init__(self, input_dim: int = 8, hidden_dim: int = 32, 
                 output_dim: int = 5, seed: int = None):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        rng = np.random.RandomState(seed)
        
        # Initialize weights with Xavier initialization
        self.W1 = rng.randn(input_dim, hidden_dim) * np.sqrt(2.0 / input_dim)
        self.b1 = np.zeros(hidden_dim)
        
        self.W2 = rng.randn(hidden_dim, hidden_dim) * np.sqrt(2.0 / hidden_dim)
        self.b2 = np.zeros(hidden_dim)
        
        self.W3 = rng.randn(hidden_dim, output_dim) * np.sqrt(2.0 / hidden_dim)
        self.b3 = np.zeros(output_dim)
    
    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Forward pass.
        
        Returns:
            probs: Sigmoid probabilities for each concept
            hidden: Hidden layer activations (for leakage analysis)
        """
        # Layer 1
        h1 = np.maximum(0, x @ self.W1 + self.b1)  # ReLU
        
        # Layer 2
        h2 = np.maximum(0, h1 @ self.W2 + self.b2)  # ReLU
        
        # Output layer
        logits = h2 @ self.W3 + self.b3
        probs = 1 / (1 + np.exp(-logits))  # Sigmoid
        
        return probs, h2
    
    def train_step(self, x: np.ndarray, y: np.ndarray, lr: float = 0.01) -> float:
        """
        Single training step with gradient descent.
        
        Args:
            x: Input features (batch_size, input_dim)
            y: Target concept labels (batch_size, output_dim)
            lr: Learning rate
        
        Returns:
            Binary cross-entropy loss
        """
        batch_size = x.shape[0]
        
        # Forward pass
        h1 = np.maximum(0, x @ self.W1 + self.b1)
        h2 = np.maximum(0, h1 @ self.W2 + self.b2)
        logits = h2 @ self.W3 + self.b3
        probs = 1 / (1 + np.exp(-np.clip(logits, -20, 20)))
        
        # Loss
        eps = 1e-7
        loss = -np.mean(y * np.log(probs + eps) + (1 - y) * np.log(1 - probs + eps))
        
        # Backward pass
        d_logits = (probs - y) / batch_size
        
        d_W3 = h2.T @ d_logits
        d_b3 = np.sum(d_logits, axis=0)
        
        d_h2 = d_logits @ self.W3.T
        d_h2 = d_h2 * (h2 > 0)  # ReLU gradient
        
        d_W2 = h1.T @ d_h2
        d_b2 = np.sum(d_h2, axis=0)
        
        d_h1 = d_h2 @ self.W2.T
        d_h1 = d_h1 * (h1 > 0)
        
        d_W1 = x.T @ d_h1
        d_b1 = np.sum(d_h1, axis=0)
        
        # Update weights
        self.W3 -= lr * d_W3
        self.b3 -= lr * d_b3
        self.W2 -= lr * d_W2
        self.b2 -= lr * d_b2
        self.W1 -= lr * d_W1
        self.b1 -= lr * d_b1
        
        return loss


class SoftConcepts:
    """
    Neural network-based soft concept extractor.
    
    Can return:
    - Just concept probabilities (use_leakage=False)
    - Probabilities + hidden embeddings (use_leakage=True)
    
    The hidden embeddings can "leak" information beyond the concept labels.
    """
    
    def __init__(self, env, use_leakage: bool = True, 
                 hidden_dim: int = 32, seed: int = None):
        self.env = env
        self.use_leakage = use_leakage
        self.n_concepts = 5
        
        # Feature dimension from environment
        sample_state = (0, 0)
        sample_features = env.state_to_features(sample_state)
        self.input_dim = len(sample_features)
        
        self.encoder = SoftConceptEncoder(
            input_dim=self.input_dim,
            hidden_dim=hidden_dim,
            output_dim=self.n_concepts,
            seed=seed
        )
    
    def extract(self, state: Tuple[int, int]) -> np.ndarray:
        """
        Extract concept representation from state.
        
        If use_leakage=True: returns concatenation of [probs, hidden]
        If use_leakage=False: returns just probs
        """
        features = self.env.state_to_features(state)
        probs, hidden = self.encoder.forward(features)
        
        if self.use_leakage:
            return np.concatenate([probs, hidden])
        else:
            return probs
    
    def extract_probs_only(self, state: Tuple[int, int]) -> np.ndarray:
        """Extract only concept probabilities (no hidden layer)."""
        features = self.env.state_to_features(state)
        probs, _ = self.encoder.forward(features)
        return probs
    
    def __call__(self, state: Tuple[int, int]) -> np.ndarray:
        return self.extract(state)
    
    def train_on_trajectories(self, trajectories: List[List[Dict]], 
                              hard_concepts: HardConcepts,
                              epochs: int = 100, lr: float = 0.01,
                              verbose: bool = False):
        """
        Train the encoder to predict hard concept labels from features.
        """
        # Collect all (features, labels) pairs
        X = []
        Y = []
        
        for traj in trajectories:
            for step in traj:
                state = step['state']
                features = step.get('features', self.env.state_to_features(state))
                labels = hard_concepts.extract(state)
                X.append(features)
                Y.append(labels)
        
        X = np.array(X)
        Y = np.array(Y)
        
        # Training loop
        for epoch in range(epochs):
            # Shuffle
            perm = np.random.permutation(len(X))
            X_shuffled = X[perm]
            Y_shuffled = Y[perm]
            
            # Train on mini-batches
            batch_size = 32
            total_loss = 0
            n_batches = 0
            
            for i in range(0, len(X), batch_size):
                X_batch = X_shuffled[i:i+batch_size]
                Y_batch = Y_shuffled[i:i+batch_size]
                
                loss = self.encoder.train_step(X_batch, Y_batch, lr=lr)
                total_loss += loss
                n_batches += 1
            
            if verbose and (epoch + 1) % 20 == 0:
                avg_loss = total_loss / n_batches
                print(f"    Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")


def measure_leakage(concept_extractor, states: np.ndarray, 
                    features: np.ndarray) -> float:
    """
    Measure information leakage in concept embeddings.
    
    Uses a linear probe: Can we predict raw features from concept embeddings?
    High R² = high leakage (embeddings encode extra info beyond concepts)
    
    Args:
        concept_extractor: HardConcepts or SoftConcepts instance
        states: Array of states to evaluate
        features: Corresponding raw feature vectors
    
    Returns:
        R² score of linear regression from embeddings to features
    """
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import cross_val_score
    
    # Extract concept embeddings
    embeddings = []
    for state in states:
        if isinstance(state, np.ndarray):
            state = tuple(state)
        emb = concept_extractor(state)
        embeddings.append(emb)
    
    embeddings = np.array(embeddings)
    
    # Fit linear probe
    probe = Ridge(alpha=1.0)
    
    try:
        scores = cross_val_score(probe, embeddings, features, 
                                 cv=min(5, len(embeddings) // 10 + 1), 
                                 scoring='r2')
        return max(0, scores.mean())  # Clamp negative R² to 0
    except:
        return 0.0


def train_probe(concept_extractor, states, features):
    """
    Train a linear probe to predict raw features from concept embeddings.
    
    Args:
        concept_extractor: HardConcepts or SoftConcepts instance
        states: List of states to train on
        features: Corresponding raw feature vectors (n_samples, n_features)
    
    Returns:
        Fitted Ridge regression probe
    """
    from sklearn.linear_model import Ridge
    
    # Extract concept embeddings
    embeddings = []
    for state in states:
        if isinstance(state, np.ndarray):
            state = tuple(state)
        emb = concept_extractor(state)
        embeddings.append(emb)
    
    embeddings = np.array(embeddings)
    features = np.array(features)
    
    # Fit linear probe
    probe = Ridge(alpha=1.0)
    probe.fit(embeddings, features)
    
    return probe


def evaluate_probe(probe, concept_extractor, states, features):
    """
    Evaluate a trained probe on new data.
    
    Args:
        probe: Fitted Ridge regression probe from train_probe()
        concept_extractor: HardConcepts or SoftConcepts instance
        states: List of states to evaluate on
        features: Corresponding raw feature vectors
    
    Returns:
        R² score (higher = more leakage, negative = worse than baseline)
    """
    # Extract concept embeddings
    embeddings = []
    for state in states:
        if isinstance(state, np.ndarray):
            state = tuple(state)
        emb = concept_extractor(state)
        embeddings.append(emb)
    
    embeddings = np.array(embeddings)
    features = np.array(features)
    
    # Compute R² (can be negative if probe fails badly on OOD data)
    r2 = probe.score(embeddings, features)
    return r2  # Don't clamp - negative R² is informative


class GatedSoftConcepts:
    """
    Entropy-gated soft concept extractor.
    Adapts MixCEM (Zarlenga et al., ICML 2025) to sequential OPE.

        gate  = 1 - H(probs)
        c_out = global_mean + gate * (probs - global_mean)

    When in-distribution:  gate ~ 1  -> use full concept probs
    When OOD:              gate ~ 0  -> fall back to global mean
    """

    def __init__(self, env, hidden_dim: int = 32, seed: int = None):
        self.env = env
        self.n_concepts = 5

        sample_features = env.state_to_features((0, 0))
        self.input_dim = len(sample_features)

        self.encoder = SoftConceptEncoder(
            input_dim=self.input_dim,
            hidden_dim=hidden_dim,
            output_dim=self.n_concepts,
            seed=seed
        )

        # Global mean computed after training
        self.global_mean = np.ones(self.n_concepts) * 0.5

    def _probs(self, state: Tuple[int, int]) -> np.ndarray:
        """Raw sigmoid probabilities from encoder."""
        features = self.env.state_to_features(state)
        probs, _ = self.encoder.forward(features)
        return probs

    def _gate(self, probs: np.ndarray) -> float:
        """
        gate = 1 - normalised_binary_entropy(probs)

        In-distribution (confident) -> low entropy -> gate near 1
        OOD (uncertain)             -> high entropy -> gate near 0
        """
        eps = 1e-7
        p = np.clip(probs, eps, 1 - eps)
        h = -(p * np.log(p) + (1 - p) * np.log(1 - p))
        return float(1.0 - np.mean(h) / np.log(2))

    def extract(self, state: Tuple[int, int]) -> np.ndarray:
        """
        Gated concept vector (5 numbers).

        gate ~ 1 -> close to raw probs
        gate ~ 0 -> close to global_mean (safe fallback)
        """
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
        """
        Train encoder on trajectory data.
        Then compute global mean of probs over all training states.
        """
        X, Y = [], []
        for traj in trajectories:
            for step in traj:
                state    = step['state']
                features = step.get('features', self.env.state_to_features(state))
                labels   = hard_concepts.extract(state)
                X.append(features)
                Y.append(labels)

        X = np.array(X)
        Y = np.array(Y)

        for epoch in range(epochs):
            perm = np.random.permutation(len(X))
            X_shuffled = X[perm]
            Y_shuffled = Y[perm]

            total_loss = 0
            n_batches  = 0

            for i in range(0, len(X), 32):
                X_batch = X_shuffled[i:i+32]
                Y_batch = Y_shuffled[i:i+32]
                loss = self.encoder.train_step(X_batch, Y_batch, lr=lr)
                total_loss += loss
                n_batches  += 1

            if verbose and (epoch + 1) % 20 == 0:
                print(f"    Epoch {epoch+1}/{epochs}, "
                      f"Loss: {total_loss/n_batches:.4f}")

        # Compute global mean over ALL training states
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
    
    Instead of entropy (which fails because network is overconfident),
    uses conformal prediction to detect OOD states:
    
    1. On calibration states: compute residual = |hard(s) - soft(s)|
       (how different is soft concept from hard concept)
    2. At inference: gate = 1 - conformal_score(s)
       where conformal_score = quantile rank of residual
       
    This is distribution-free and requires NO knowledge of OOD states.
    In-distribution: soft ≈ hard → small residual → gate OPEN
    OOD:             soft ≠ hard → large residual → gate CLOSED
    """

    def __init__(self, env, seed: int = None):
        self.env        = env
        self.n_concepts = 5
        self.seed       = seed

        sample_features    = env.state_to_features((0, 0))
        self.input_dim     = len(sample_features)
        self.encoder       = SoftConceptEncoder(
            input_dim  = self.input_dim,
            hidden_dim = 32,
            output_dim = self.n_concepts,
            seed       = seed
        )
        self.global_mean        = np.ones(self.n_concepts) * 0.5
        self.calibration_scores = np.array([])  # conformal residuals

    def _soft_probs(self, state: Tuple[int, int]) -> np.ndarray:
        features = self.env.state_to_features(state)
        probs, _ = self.encoder.forward(features)
        return probs

    def _conformal_score(self, state: Tuple[int, int],
                         hard_concepts) -> float:
        """
        Residual = mean absolute difference between hard and soft concepts.
        Small = in-distribution, Large = OOD.
        """
        soft = self._soft_probs(state)
        hard = hard_concepts.extract(state)
        return float(np.mean(np.abs(soft - hard)))

    def calibrate(self, states, hard_concepts):
        """
        Compute conformal scores on calibration states.
        These are in-distribution states (t < train_horizon).
        """
        scores = [self._conformal_score(s, hard_concepts)
                  for s in states]
        self.calibration_scores = np.array(scores)
        print(f"    Conformal calibration: {len(scores)} states, "
              f"score mean={np.mean(scores):.3f}, "
              f"95th pct={np.percentile(scores, 95):.3f}")


    def get_gate_value(self, state: Tuple[int, int],
                       hard_concepts=None) -> float:
        if len(self.calibration_scores) == 0:
            return 1.0

        score     = self._conformal_score(state, self._hard_concepts)
        threshold = np.percentile(self.calibration_scores, 95)

        if score <= threshold:
            return 1.0  # clearly in-distribution
        else:
            # Gate closes proportionally to how far OOD
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
        """Train encoder, compute global mean, calibrate."""
        # Store hard_concepts reference for gate computation
        self._hard_concepts = hard_concepts

        X, Y = [], []
        cal_states = []

        for traj in trajectories:
            for step in traj:
                state    = step['state']
                features = step.get('features',
                                    self.env.state_to_features(state))
                labels   = hard_concepts.extract(state)
                X.append(features)
                Y.append(labels)
                cal_states.append(state)

        X = np.array(X)
        Y = np.array(Y)

        for epoch in range(epochs):
            perm       = np.random.permutation(len(X))
            X_shuffled = X[perm]
            Y_shuffled = Y[perm]
            total_loss = 0
            n_batches  = 0

            for i in range(0, len(X), 32):
                loss = self.encoder.train_step(
                    X_shuffled[i:i+32], Y_shuffled[i:i+32], lr=lr)
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

        # Calibrate on training states
        self.calibrate(cal_states, hard_concepts)


if __name__ == "__main__":
    from gridworld import WindyGridworld, collect_trajectory
    from policies import EpsilonGreedyPolicy
    
    env = WindyGridworld()
    
    # Test hard concepts
    print("Testing HardConcepts...")
    hard = HardConcepts(env)
    
    test_states = [(3, 0), (3, 7), (0, 6), (6, 3)]
    for state in test_states:
        concepts = hard.extract(state)
        print(f"  State {state}: {concepts} (names: {hard.concept_names})")
    
    # Test soft concepts
    print("\nTesting SoftConcepts...")
    soft = SoftConcepts(env, use_leakage=True)
    soft_no_leak = SoftConcepts(env, use_leakage=False)
    
    # Generate training data
    policy = EpsilonGreedyPolicy(env, epsilon=0.3, seed=42)
    trajectories = []
    for _ in range(50):
        traj = collect_trajectory(env, policy, max_steps=50)
        trajectories.append(traj)
    
    # Train soft concepts
    print("  Training soft concept encoder...")
    soft.train_on_trajectories(trajectories, hard, epochs=100, verbose=True)
    soft_no_leak.encoder = soft.encoder  # Share weights
    
    # Test output shapes
    print(f"\n  Soft (with leakage) output shape: {soft.extract((3, 0)).shape}")
    print(f"  Soft (no leakage) output shape: {soft_no_leak.extract((3, 0)).shape}")
    
    # Measure leakage
    print("\nMeasuring leakage...")
    all_states = []
    all_features = []
    for traj in trajectories[:20]:
        for step in traj:
            all_states.append(step['state'])
            all_features.append(step['features'])
    all_states = np.array(all_states)
    all_features = np.array(all_features)
    
    leakage_hard = measure_leakage(hard, all_states, all_features)
    leakage_soft = measure_leakage(soft, all_states, all_features)
    leakage_soft_no_leak = measure_leakage(soft_no_leak, all_states, all_features)
    
    print(f"  Hard concepts leakage R²: {leakage_hard:.4f}")
    print(f"  Soft concepts (with embeddings) leakage R²: {leakage_soft:.4f}")
    print(f"  Soft concepts (probs only) leakage R²: {leakage_soft_no_leak:.4f}")
    """
        Concept Extractors for Windy Gridworld

        Includes:
        - HardConcepts: Rule-based binary concepts (no leakage)
        - SoftConcepts: Neural network concept encoder (can leak)
        - measure_leakage: Probe to measure information leakage in embeddings
"""

import numpy as np
from typing import Tuple, List, Dict, Optional


class HardConcepts:
    """
    Rule-based binary concepts with no information leakage.
    
    Concepts:
    - near_goal: Manhattan distance to goal <= 2
    - high_wind: Current column has wind >= 2
    - in_left_half: Column < 5
    - in_top_half: Row < 3 (closer to top)
    - near_start: Manhattan distance to start <= 2
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
        
        # Concept 1: Near goal
        goal_dist = abs(row - self.env.goal[0]) + abs(col - self.env.goal[1])
        near_goal = float(goal_dist <= 2)
        
        # Concept 2: High wind
        high_wind = float(self.env.wind[col] >= 2)
        
        # Concept 3: In left half
        in_left = float(col < 5)
        
        # Concept 4: In top half
        in_top = float(row < 3)
        
        # Concept 5: Near start
        start_dist = abs(row - self.env.start[0]) + abs(col - self.env.start[1])
        near_start = float(start_dist <= 2)
        
        return np.array([near_goal, high_wind, in_left, in_top, near_start])
    
    def __call__(self, state: Tuple[int, int]) -> np.ndarray:
        return self.extract(state)
    
    def to_index(self, state: Tuple[int, int]) -> int:
        """Convert concepts to single index (0-31)."""
        concepts = self.extract(state)
        idx = 0
        for i, c in enumerate(concepts):
            idx += int(c) * (2 ** i)
        return idx


class SoftConceptEncoder:
    """
    Simple MLP encoder from state features to concept probabilities.
    
    This can leak information beyond the concepts if the hidden layers
    encode more than just the concept labels.
    """
    
    def __init__(self, input_dim: int = 8, hidden_dim: int = 32, 
                 output_dim: int = 5, seed: int = None):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        rng = np.random.RandomState(seed)
        
        # Initialize weights with Xavier initialization
        self.W1 = rng.randn(input_dim, hidden_dim) * np.sqrt(2.0 / input_dim)
        self.b1 = np.zeros(hidden_dim)
        
        self.W2 = rng.randn(hidden_dim, hidden_dim) * np.sqrt(2.0 / hidden_dim)
        self.b2 = np.zeros(hidden_dim)
        
        self.W3 = rng.randn(hidden_dim, output_dim) * np.sqrt(2.0 / hidden_dim)
        self.b3 = np.zeros(output_dim)
    
    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Forward pass.
        
        Returns:
            probs: Sigmoid probabilities for each concept
            hidden: Hidden layer activations (for leakage analysis)
        """
        # Layer 1
        h1 = np.maximum(0, x @ self.W1 + self.b1)  # ReLU
        
        # Layer 2
        h2 = np.maximum(0, h1 @ self.W2 + self.b2)  # ReLU
        
        # Output layer
        logits = h2 @ self.W3 + self.b3
        probs = 1 / (1 + np.exp(-logits))  # Sigmoid
        
        return probs, h2
    
    def train_step(self, x: np.ndarray, y: np.ndarray, lr: float = 0.01) -> float:
        """
        Single training step with gradient descent.
        
        Args:
            x: Input features (batch_size, input_dim)
            y: Target concept labels (batch_size, output_dim)
            lr: Learning rate
        
        Returns:
            Binary cross-entropy loss
        """
        batch_size = x.shape[0]
        
        # Forward pass
        h1 = np.maximum(0, x @ self.W1 + self.b1)
        h2 = np.maximum(0, h1 @ self.W2 + self.b2)
        logits = h2 @ self.W3 + self.b3
        probs = 1 / (1 + np.exp(-np.clip(logits, -20, 20)))
        
        # Loss
        eps = 1e-7
        loss = -np.mean(y * np.log(probs + eps) + (1 - y) * np.log(1 - probs + eps))
        
        # Backward pass
        d_logits = (probs - y) / batch_size
        
        d_W3 = h2.T @ d_logits
        d_b3 = np.sum(d_logits, axis=0)
        
        d_h2 = d_logits @ self.W3.T
        d_h2 = d_h2 * (h2 > 0)  # ReLU gradient
        
        d_W2 = h1.T @ d_h2
        d_b2 = np.sum(d_h2, axis=0)
        
        d_h1 = d_h2 @ self.W2.T
        d_h1 = d_h1 * (h1 > 0)
        
        d_W1 = x.T @ d_h1
        d_b1 = np.sum(d_h1, axis=0)
        
        # Update weights
        self.W3 -= lr * d_W3
        self.b3 -= lr * d_b3
        self.W2 -= lr * d_W2
        self.b2 -= lr * d_b2
        self.W1 -= lr * d_W1
        self.b1 -= lr * d_b1
        
        return loss


class SoftConcepts:
    """
    Neural network-based soft concept extractor.
    
    Can return:
    - Just concept probabilities (use_leakage=False)
    - Probabilities + hidden embeddings (use_leakage=True)
    
    The hidden embeddings can "leak" information beyond the concept labels.
    """
    
    def __init__(self, env, use_leakage: bool = True, 
                 hidden_dim: int = 32, seed: int = None):
        self.env = env
        self.use_leakage = use_leakage
        self.n_concepts = 5
        
        # Feature dimension from environment
        sample_state = (0, 0)
        sample_features = env.state_to_features(sample_state)
        self.input_dim = len(sample_features)
        
        self.encoder = SoftConceptEncoder(
            input_dim=self.input_dim,
            hidden_dim=hidden_dim,
            output_dim=self.n_concepts,
            seed=seed
        )
    
    def extract(self, state: Tuple[int, int]) -> np.ndarray:
        """
        Extract concept representation from state.
        
        If use_leakage=True: returns concatenation of [probs, hidden]
        If use_leakage=False: returns just probs
        """
        features = self.env.state_to_features(state)
        probs, hidden = self.encoder.forward(features)
        
        if self.use_leakage:
            return np.concatenate([probs, hidden])
        else:
            return probs
    
    def extract_probs_only(self, state: Tuple[int, int]) -> np.ndarray:
        """Extract only concept probabilities (no hidden layer)."""
        features = self.env.state_to_features(state)
        probs, _ = self.encoder.forward(features)
        return probs
    
    def __call__(self, state: Tuple[int, int]) -> np.ndarray:
        return self.extract(state)
    
    def train_on_trajectories(self, trajectories: List[List[Dict]], 
                              hard_concepts: HardConcepts,
                              epochs: int = 100, lr: float = 0.01,
                              verbose: bool = False):
        """
        Train the encoder to predict hard concept labels from features.
        """
        # Collect all (features, labels) pairs
        X = []
        Y = []
        
        for traj in trajectories:
            for step in traj:
                state = step['state']
                features = step.get('features', self.env.state_to_features(state))
                labels = hard_concepts.extract(state)
                X.append(features)
                Y.append(labels)
        
        X = np.array(X)
        Y = np.array(Y)
        
        # Training loop
        for epoch in range(epochs):
            # Shuffle
            perm = np.random.permutation(len(X))
            X_shuffled = X[perm]
            Y_shuffled = Y[perm]
            
            # Train on mini-batches
            batch_size = 32
            total_loss = 0
            n_batches = 0
            
            for i in range(0, len(X), batch_size):
                X_batch = X_shuffled[i:i+batch_size]
                Y_batch = Y_shuffled[i:i+batch_size]
                
                loss = self.encoder.train_step(X_batch, Y_batch, lr=lr)
                total_loss += loss
                n_batches += 1
            
            if verbose and (epoch + 1) % 20 == 0:
                avg_loss = total_loss / n_batches
                print(f"    Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")


def measure_leakage(concept_extractor, states: np.ndarray, 
                    features: np.ndarray) -> float:
    """
    Measure information leakage in concept embeddings.
    
    Uses a linear probe: Can we predict raw features from concept embeddings?
    High R² = high leakage (embeddings encode extra info beyond concepts)
    
    Args:
        concept_extractor: HardConcepts or SoftConcepts instance
        states: Array of states to evaluate
        features: Corresponding raw feature vectors
    
    Returns:
        R² score of linear regression from embeddings to features
    """
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import cross_val_score
    
    # Extract concept embeddings
    embeddings = []
    for state in states:
        if isinstance(state, np.ndarray):
            state = tuple(state)
        emb = concept_extractor(state)
        embeddings.append(emb)
    
    embeddings = np.array(embeddings)
    
    # Fit linear probe
    probe = Ridge(alpha=1.0)
    
    try:
        scores = cross_val_score(probe, embeddings, features, 
                                 cv=min(5, len(embeddings) // 10 + 1), 
                                 scoring='r2')
        return max(0, scores.mean())  # Clamp negative R² to 0
    except:
        return 0.0


def train_probe(concept_extractor, states, features):
    """
    Train a linear probe to predict raw features from concept embeddings.
    
    Args:
        concept_extractor: HardConcepts or SoftConcepts instance
        states: List of states to train on
        features: Corresponding raw feature vectors (n_samples, n_features)
    
    Returns:
        Fitted Ridge regression probe
    """
    from sklearn.linear_model import Ridge
    
    # Extract concept embeddings
    embeddings = []
    for state in states:
        if isinstance(state, np.ndarray):
            state = tuple(state)
        emb = concept_extractor(state)
        embeddings.append(emb)
    
    embeddings = np.array(embeddings)
    features = np.array(features)
    
    # Fit linear probe
    probe = Ridge(alpha=1.0)
    probe.fit(embeddings, features)
    
    return probe


def evaluate_probe(probe, concept_extractor, states, features):
    """
    Evaluate a trained probe on new data.
    
    Args:
        probe: Fitted Ridge regression probe from train_probe()
        concept_extractor: HardConcepts or SoftConcepts instance
        states: List of states to evaluate on
        features: Corresponding raw feature vectors
    
    Returns:
        R² score (higher = more leakage, negative = worse than baseline)
    """
    # Extract concept embeddings
    embeddings = []
    for state in states:
        if isinstance(state, np.ndarray):
            state = tuple(state)
        emb = concept_extractor(state)
        embeddings.append(emb)
    
    embeddings = np.array(embeddings)
    features = np.array(features)
    
    # Compute R² (can be negative if probe fails badly on OOD data)
    r2 = probe.score(embeddings, features)
    return r2  # Don't clamp - negative R² is informative


class GatedSoftConcepts:
    """
    Entropy-gated soft concept extractor.
    Adapts MixCEM (Zarlenga et al., ICML 2025) to sequential OPE.

        gate  = 1 - H(probs)
        c_out = global_mean + gate * (probs - global_mean)

    When in-distribution:  gate ~ 1  -> use full concept probs
    When OOD:              gate ~ 0  -> fall back to global mean
    """

    def __init__(self, env, hidden_dim: int = 32, seed: int = None):
        self.env = env
        self.n_concepts = 5

        sample_features = env.state_to_features((0, 0))
        self.input_dim = len(sample_features)

        self.encoder = SoftConceptEncoder(
            input_dim=self.input_dim,
            hidden_dim=hidden_dim,
            output_dim=self.n_concepts,
            seed=seed
        )

        # Global mean computed after training
        self.global_mean = np.ones(self.n_concepts) * 0.5

    def _probs(self, state: Tuple[int, int]) -> np.ndarray:
        """Raw sigmoid probabilities from encoder."""
        features = self.env.state_to_features(state)
        probs, _ = self.encoder.forward(features)
        return probs

    def _gate(self, probs: np.ndarray) -> float:
        """
        gate = 1 - normalised_binary_entropy(probs)

        In-distribution (confident) -> low entropy -> gate near 1
        OOD (uncertain)             -> high entropy -> gate near 0
        """
        eps = 1e-7
        p = np.clip(probs, eps, 1 - eps)
        h = -(p * np.log(p) + (1 - p) * np.log(1 - p))
        return float(1.0 - np.mean(h) / np.log(2))

    def extract(self, state: Tuple[int, int]) -> np.ndarray:
        """
        Gated concept vector (5 numbers).

        gate ~ 1 -> close to raw probs
        gate ~ 0 -> close to global_mean (safe fallback)
        """
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
        """
        Train encoder on trajectory data.
        Then compute global mean of probs over all training states.
        """
        X, Y = [], []
        for traj in trajectories:
            for step in traj:
                state    = step['state']
                features = step.get('features', self.env.state_to_features(state))
                labels   = hard_concepts.extract(state)
                X.append(features)
                Y.append(labels)

        X = np.array(X)
        Y = np.array(Y)

        for epoch in range(epochs):
            perm = np.random.permutation(len(X))
            X_shuffled = X[perm]
            Y_shuffled = Y[perm]

            total_loss = 0
            n_batches  = 0

            for i in range(0, len(X), 32):
                X_batch = X_shuffled[i:i+32]
                Y_batch = Y_shuffled[i:i+32]
                loss = self.encoder.train_step(X_batch, Y_batch, lr=lr)
                total_loss += loss
                n_batches  += 1

            if verbose and (epoch + 1) % 20 == 0:
                print(f"    Epoch {epoch+1}/{epochs}, "
                      f"Loss: {total_loss/n_batches:.4f}")

        # Compute global mean over ALL training states
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
    
    Instead of entropy (which fails because network is overconfident),
    uses conformal prediction to detect OOD states:
    
    1. On calibration states: compute residual = |hard(s) - soft(s)|
       (how different is soft concept from hard concept)
    2. At inference: gate = 1 - conformal_score(s)
       where conformal_score = quantile rank of residual
       
    This is distribution-free and requires NO knowledge of OOD states.
    In-distribution: soft ≈ hard → small residual → gate OPEN
    OOD:             soft ≠ hard → large residual → gate CLOSED
    """

    def __init__(self, env, seed: int = None):
        self.env        = env
        self.n_concepts = 5
        self.seed       = seed

        sample_features    = env.state_to_features((0, 0))
        self.input_dim     = len(sample_features)
        self.encoder       = SoftConceptEncoder(
            input_dim  = self.input_dim,
            hidden_dim = 32,
            output_dim = self.n_concepts,
            seed       = seed
        )
        self.global_mean        = np.ones(self.n_concepts) * 0.5
        self.calibration_scores = np.array([])  # conformal residuals

    def _soft_probs(self, state: Tuple[int, int]) -> np.ndarray:
        features = self.env.state_to_features(state)
        probs, _ = self.encoder.forward(features)
        return probs

    def _conformal_score(self, state: Tuple[int, int],
                         hard_concepts) -> float:
        """
        Residual = mean absolute difference between hard and soft concepts.
        Small = in-distribution, Large = OOD.
        """
        soft = self._soft_probs(state)
        hard = hard_concepts.extract(state)
        return float(np.mean(np.abs(soft - hard)))

    def calibrate(self, states, hard_concepts):
        """
        Compute conformal scores on calibration states.
        These are in-distribution states (t < train_horizon).
        """
        scores = [self._conformal_score(s, hard_concepts)
                  for s in states]
        self.calibration_scores = np.array(scores)
        print(f"    Conformal calibration: {len(scores)} states, "
              f"score mean={np.mean(scores):.3f}, "
              f"95th pct={np.percentile(scores, 95):.3f}")


    def get_gate_value(self, state: Tuple[int, int],
                       hard_concepts=None) -> float:
        if len(self.calibration_scores) == 0:
            return 1.0

        score     = self._conformal_score(state, self._hard_concepts)
        threshold = np.percentile(self.calibration_scores, 95)

        if score <= threshold:
            return 1.0  # clearly in-distribution
        else:
            # Gate closes proportionally to how far OOD
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
        """Train encoder, compute global mean, calibrate."""
        # Store hard_concepts reference for gate computation
        self._hard_concepts = hard_concepts

        X, Y = [], []
        cal_states = []

        for traj in trajectories:
            for step in traj:
                state    = step['state']
                features = step.get('features',
                                    self.env.state_to_features(state))
                labels   = hard_concepts.extract(state)
                X.append(features)
                Y.append(labels)
                cal_states.append(state)

        X = np.array(X)
        Y = np.array(Y)

        for epoch in range(epochs):
            perm       = np.random.permutation(len(X))
            X_shuffled = X[perm]
            Y_shuffled = Y[perm]
            total_loss = 0
            n_batches  = 0

            for i in range(0, len(X), 32):
                loss = self.encoder.train_step(
                    X_shuffled[i:i+32], Y_shuffled[i:i+32], lr=lr)
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

        # Calibrate on training states
        self.calibrate(cal_states, hard_concepts)


if __name__ == "__main__":
    from gridworld import WindyGridworld, collect_trajectory
    from policies import EpsilonGreedyPolicy
    
    env = WindyGridworld()
    
    # Test hard concepts
    print("Testing HardConcepts...")
    hard = HardConcepts(env)
    
    test_states = [(3, 0), (3, 7), (0, 6), (6, 3)]
    for state in test_states:
        concepts = hard.extract(state)
        print(f"  State {state}: {concepts} (names: {hard.concept_names})")
    
    # Test soft concepts
    print("\nTesting SoftConcepts...")
    soft = SoftConcepts(env, use_leakage=True)
    soft_no_leak = SoftConcepts(env, use_leakage=False)
    
    # Generate training data
    policy = EpsilonGreedyPolicy(env, epsilon=0.3, seed=42)
    trajectories = []
    for _ in range(50):
        traj = collect_trajectory(env, policy, max_steps=50)
        trajectories.append(traj)
    
    # Train soft concepts
    print("  Training soft concept encoder...")
    soft.train_on_trajectories(trajectories, hard, epochs=100, verbose=True)
    soft_no_leak.encoder = soft.encoder  # Share weights
    
    # Test output shapes
    print(f"\n  Soft (with leakage) output shape: {soft.extract((3, 0)).shape}")
    print(f"  Soft (no leakage) output shape: {soft_no_leak.extract((3, 0)).shape}")
    
    # Measure leakage
    print("\nMeasuring leakage...")
    all_states = []
    all_features = []
    for traj in trajectories[:20]:
        for step in traj:
            all_states.append(step['state'])
            all_features.append(step['features'])
    all_states = np.array(all_states)
    all_features = np.array(all_features)
    
    leakage_hard = measure_leakage(hard, all_states, all_features)
    leakage_soft = measure_leakage(soft, all_states, all_features)
    leakage_soft_no_leak = measure_leakage(soft_no_leak, all_states, all_features)
    
    print(f"  Hard concepts leakage R²: {leakage_hard:.4f}")
    print(f"  Soft concepts (with embeddings) leakage R²: {leakage_soft:.4f}")
    print(f"  Soft concepts (probs only) leakage R²: {leakage_soft_no_leak:.4f}")