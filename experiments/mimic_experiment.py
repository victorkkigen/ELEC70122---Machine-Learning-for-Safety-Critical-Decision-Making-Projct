"""
MIMIC-III Temporal Leakage Experiment
=====================================

This experiment applies the temporal leakage poisoning analysis to real
clinical data from MIMIC-III ICU stays.

Key differences from GridWorld:
- States are 15-dimensional clinical features (vitals, labs)
- Actions are 25 treatment combinations (5 fluid bins × 5 vasopressor bins)
- Trajectories are variable length ICU stays (4-20 time bins)
- Reward is based on mortality outcome

Run: python mimic_experiment.py
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score


# =============================================================================
# Data Loading
# =============================================================================

@dataclass
class MIMICData:
    """Container for MIMIC trajectory data."""
    states: np.ndarray        # (n_patients, max_len, n_features)
    actions: np.ndarray       # (n_patients, max_len)
    rewards: np.ndarray       # (n_patients, max_len)
    lengths: np.ndarray       # (n_patients,)
    died: np.ndarray          # (n_patients,)
    feature_names: List[str]
    n_actions: int
    time_bin_hours: float = 4.0
    
    @classmethod
    def load(cls, path: str) -> 'MIMICData':
        """Load from NPZ file."""
        data = np.load(path, allow_pickle=True)
        return cls(
            states=data['states'],
            actions=data['actions'],
            rewards=data['rewards'],
            lengths=data['lengths'],
            died=data['died'],
            feature_names=list(data['feature_names']),
            n_actions=int(data['n_actions']),
            time_bin_hours=float(data.get('time_bin_hours', 4.0))
        )
    
    def to_trajectories(self) -> List[List[Dict]]:
        """Convert to list of trajectory dicts (like GridWorld format)."""
        trajectories = []
        
        for i in range(len(self.lengths)):
            traj = []
            length = self.lengths[i]
            
            for t in range(length):
                step = {
                    'state': self.states[i, t],        # 15-dim feature vector
                    'action': int(self.actions[i, t]),
                    'reward': float(self.rewards[i, t]),
                    'features': self.states[i, t],     # Same as state for MIMIC
                    't': t,
                    'done': (t == length - 1),
                    'patient_idx': i,
                    'died': bool(self.died[i])
                }
                traj.append(step)
            
            trajectories.append(traj)
        
        return trajectories


# =============================================================================
# MIMIC Concept Extractors
# =============================================================================

class MIMICHardConcepts:
    """
    Rule-based clinical concepts for MIMIC-IV data.
    
    Features (indices based on MIMIC-IV preprocessing):
        0: heart_rate, 1: sbp, 2: dbp, 3: mbp, 4: resp_rate,
        5: spo2, 6: temperature, 7: gcs_total, 8: lactate, 9: creatinine,
        10: bilirubin, 11: platelet, 12: wbc, 13: hemoglobin,
        14: potassium, 15: sodium, 16: glucose, 17: bun
    
    Hard concepts (clinical thresholds on normalized [0,1] values):
        - high_creatinine: creatinine > 0.5 (normalized) → AKI risk
        - low_spo2: spo2 < 0.7 (normalized) → hypoxemia  
        - high_lactate: lactate > 0.5 (normalized) → sepsis marker
        - low_gcs: gcs < 0.6 (normalized) → altered consciousness
        - tachycardia: heart_rate > 0.6 (normalized) → stress response
    """
    
    def __init__(self, n_features: int = 18):
        self.n_features = n_features
        self.n_concepts = 5
        self.concept_names = [
            'high_creatinine', 'low_spo2', 'high_lactate', 
            'low_gcs', 'tachycardia'
        ]
        
        # Feature indices for MIMIC-IV
        self.idx_heart_rate = 0
        self.idx_spo2 = 5
        self.idx_gcs = 7
        self.idx_lactate = 8
        self.idx_creatinine = 9
    
    def extract(self, state: np.ndarray) -> np.ndarray:
        """Extract binary clinical concepts from state."""
        concepts = np.zeros(self.n_concepts)
        
        # High creatinine (AKI risk)
        concepts[0] = float(state[self.idx_creatinine] > 0.5)
        
        # Low SpO2 (hypoxemia)
        concepts[1] = float(state[self.idx_spo2] < 0.7)
        
        # High lactate (sepsis marker)
        concepts[2] = float(state[self.idx_lactate] > 0.5)
        
        # Low GCS (altered consciousness)
        concepts[3] = float(state[self.idx_gcs] < 0.6)
        
        # Tachycardia (stress response)
        concepts[4] = float(state[self.idx_heart_rate] > 0.6)
        
        return concepts
    
    def __call__(self, state: np.ndarray) -> np.ndarray:
        return self.extract(state)
    
    def to_index(self, state: np.ndarray) -> int:
        """Convert concepts to single index (0-31)."""
        concepts = self.extract(state)
        idx = 0
        for i, c in enumerate(concepts):
            idx += int(c) * (2 ** i)
        return idx


class MIMICSoftConceptEncoder:
    """
    Neural network encoder for MIMIC clinical states.
    Maps 18-dim clinical features to 5-dim concept probabilities.
    """
    
    def __init__(self, input_dim: int = 18, hidden_dim: int = 32, 
                 output_dim: int = 5, seed: int = None):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        rng = np.random.RandomState(seed)
        
        # Xavier initialization
        self.W1 = rng.randn(input_dim, hidden_dim) * np.sqrt(2.0 / input_dim)
        self.b1 = np.zeros(hidden_dim)
        
        self.W2 = rng.randn(hidden_dim, hidden_dim) * np.sqrt(2.0 / hidden_dim)
        self.b2 = np.zeros(hidden_dim)
        
        self.W3 = rng.randn(hidden_dim, output_dim) * np.sqrt(2.0 / hidden_dim)
        self.b3 = np.zeros(output_dim)
    
    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Forward pass. Returns (probs, hidden)."""
        h1 = np.maximum(0, x @ self.W1 + self.b1)
        h2 = np.maximum(0, h1 @ self.W2 + self.b2)
        logits = h2 @ self.W3 + self.b3
        probs = 1 / (1 + np.exp(-np.clip(logits, -20, 20)))
        return probs, h2
    
    def train_step(self, x: np.ndarray, y: np.ndarray, lr: float = 0.01) -> float:
        """Single training step with gradient descent."""
        batch_size = x.shape[0]
        
        # Forward
        h1 = np.maximum(0, x @ self.W1 + self.b1)
        h2 = np.maximum(0, h1 @ self.W2 + self.b2)
        logits = h2 @ self.W3 + self.b3
        probs = 1 / (1 + np.exp(-np.clip(logits, -20, 20)))
        
        # Loss
        eps = 1e-7
        loss = -np.mean(y * np.log(probs + eps) + (1 - y) * np.log(1 - probs + eps))
        
        # Backward
        d_logits = (probs - y) / batch_size
        
        d_W3 = h2.T @ d_logits
        d_b3 = np.sum(d_logits, axis=0)
        
        d_h2 = d_logits @ self.W3.T
        d_h2 = d_h2 * (h2 > 0)
        
        d_W2 = h1.T @ d_h2
        d_b2 = np.sum(d_h2, axis=0)
        
        d_h1 = d_h2 @ self.W2.T
        d_h1 = d_h1 * (h1 > 0)
        
        d_W1 = x.T @ d_h1
        d_b1 = np.sum(d_h1, axis=0)
        
        # Update
        self.W3 -= lr * d_W3
        self.b3 -= lr * d_b3
        self.W2 -= lr * d_W2
        self.b2 -= lr * d_b2
        self.W1 -= lr * d_W1
        self.b1 -= lr * d_b1
        
        return loss


class MIMICSoftConcepts:
    """
    Neural network-based soft concept extractor for MIMIC.
    Can leak information via hidden embeddings.
    """
    
    def __init__(self, n_features: int = 18, hidden_dim: int = 32, 
                 use_leakage: bool = True, seed: int = None):
        self.n_features = n_features
        self.n_concepts = 5
        self.use_leakage = use_leakage
        
        self.encoder = MIMICSoftConceptEncoder(
            input_dim=n_features,
            hidden_dim=hidden_dim,
            output_dim=self.n_concepts,
            seed=seed
        )
    
    def extract(self, state: np.ndarray) -> np.ndarray:
        """Extract concept representation."""
        probs, hidden = self.encoder.forward(state)
        
        if self.use_leakage:
            return np.concatenate([probs, hidden])
        else:
            return probs
    
    def __call__(self, state: np.ndarray) -> np.ndarray:
        return self.extract(state)
    
    def train_on_trajectories(self, trajectories: List[List[Dict]],
                              hard_concepts: MIMICHardConcepts,
                              epochs: int = 100, lr: float = 0.01,
                              verbose: bool = False):
        """Train encoder to predict hard concept labels."""
        X = []
        Y = []
        
        for traj in trajectories:
            for step in traj:
                state = step['state']
                labels = hard_concepts.extract(state)
                X.append(state)
                Y.append(labels)
        
        X = np.array(X)
        Y = np.array(Y)
        
        for epoch in range(epochs):
            perm = np.random.permutation(len(X))
            X_shuffled = X[perm]
            Y_shuffled = Y[perm]
            
            total_loss = 0
            n_batches = 0
            
            for i in range(0, len(X), 32):
                X_batch = X_shuffled[i:i+32]
                Y_batch = Y_shuffled[i:i+32]
                loss = self.encoder.train_step(X_batch, Y_batch, lr=lr)
                total_loss += loss
                n_batches += 1
            
            if verbose and (epoch + 1) % 50 == 0:
                print(f"    Epoch {epoch+1}/{epochs}, Loss: {total_loss/n_batches:.4f}")


class MIMICGatedSoftConcepts:
    """
    Entropy-gated soft concepts for MIMIC.
    Falls back to global mean when uncertainty is high (OOD).
    """
    
    def __init__(self, n_features: int = 18, hidden_dim: int = 32, seed: int = None):
        self.n_features = n_features
        self.n_concepts = 5
        
        self.encoder = MIMICSoftConceptEncoder(
            input_dim=n_features,
            hidden_dim=hidden_dim,
            output_dim=self.n_concepts,
            seed=seed
        )
        
        self.global_mean = np.ones(self.n_concepts) * 0.5
    
    def _probs(self, state: np.ndarray) -> np.ndarray:
        probs, _ = self.encoder.forward(state)
        return probs
    
    def _gate(self, probs: np.ndarray) -> float:
        """Gate = 1 - normalized_entropy. High confidence -> gate near 1."""
        eps = 1e-7
        p = np.clip(probs, eps, 1 - eps)
        h = -(p * np.log(p) + (1 - p) * np.log(1 - p))
        return float(1.0 - np.mean(h) / np.log(2))
    
    def extract(self, state: np.ndarray) -> np.ndarray:
        """Gated concept vector."""
        probs = self._probs(state)
        gate = self._gate(probs)
        return self.global_mean + gate * (probs - self.global_mean)
    
    def get_gate_value(self, state: np.ndarray) -> float:
        return self._gate(self._probs(state))
    
    def extract_with_gate(self, state: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Return (gated_concepts, gate_values_per_concept)."""
        probs = self._probs(state)
        gate = self._gate(probs)
        gated_probs = self.global_mean + gate * (probs - self.global_mean)
        # Return per-concept gate values (all same for now)
        gate_values = np.full(self.n_concepts, gate)
        return gated_probs, gate_values
    
    def __call__(self, state: np.ndarray) -> np.ndarray:
        return self.extract(state)
    
    def train_on_trajectories(self, trajectories: List[List[Dict]],
                              hard_concepts: MIMICHardConcepts,
                              epochs: int = 100, lr: float = 0.01,
                              verbose: bool = False):
        """Train encoder and compute global mean."""
        X, Y = [], []
        for traj in trajectories:
            for step in traj:
                state = step['state']
                labels = hard_concepts.extract(state)
                X.append(state)
                Y.append(labels)
        
        X = np.array(X)
        Y = np.array(Y)
        
        for epoch in range(epochs):
            perm = np.random.permutation(len(X))
            X_shuffled = X[perm]
            Y_shuffled = Y[perm]
            
            total_loss = 0
            n_batches = 0
            
            for i in range(0, len(X), 32):
                X_batch = X_shuffled[i:i+32]
                Y_batch = Y_shuffled[i:i+32]
                loss = self.encoder.train_step(X_batch, Y_batch, lr=lr)
                total_loss += loss
                n_batches += 1
            
            if verbose and (epoch + 1) % 50 == 0:
                print(f"    Epoch {epoch+1}/{epochs}, Loss: {total_loss/n_batches:.4f}")
        
        # Compute global mean
        all_probs = []
        for x in X:
            probs, _ = self.encoder.forward(x)
            all_probs.append(probs)
        self.global_mean = np.mean(all_probs, axis=0)
        
        if verbose:
            print(f"    Global mean: {self.global_mean.round(3)}")


class MIMICConformalGatedConcepts:
    """
    Conformal prediction-based gating for MIMIC soft concepts.
    
    Uses FEATURE-SPACE distance (not concept residuals) to detect OOD:
    
    1. On calibration states (survivors, early timesteps):
       - Compute mean and covariance of features
       - Store Mahalanobis distances as calibration scores
    2. At inference: 
       - Compute Mahalanobis distance to training distribution
       - Gate closes when distance exceeds calibration threshold
    
    This works for MIMIC because dying patients have different feature
    trajectories than survivors - the feature-space distance captures this
    even when concept residuals are similar.
    """
    
    def __init__(self, n_features: int = 18, hidden_dim: int = 32, seed: int = None):
        self.n_features = n_features
        self.n_concepts = 5
        self.seed = seed
        
        self.encoder = MIMICSoftConceptEncoder(
            input_dim=n_features,
            hidden_dim=hidden_dim,
            output_dim=self.n_concepts,
            seed=seed
        )
        
        self.global_mean = np.ones(self.n_concepts) * 0.5
        self.calibration_scores = np.array([])
        self._hard_concepts = None
        
        # Feature-space statistics for Mahalanobis distance
        self._feature_mean = None
        self._feature_cov_inv = None
    
    def _soft_probs(self, state: np.ndarray) -> np.ndarray:
        """Raw sigmoid probabilities from encoder."""
        probs, _ = self.encoder.forward(state)
        return probs
    
    def _mahalanobis_distance(self, state: np.ndarray) -> float:
        """
        Mahalanobis distance from training feature distribution.
        Large distance = OOD state.
        """
        if self._feature_mean is None:
            return 0.0
        
        diff = state - self._feature_mean
        dist = np.sqrt(np.dot(np.dot(diff, self._feature_cov_inv), diff))
        return float(dist)
    
    def _conformal_score(self, state: np.ndarray) -> float:
        """
        Conformal score = Mahalanobis distance in feature space.
        Small = in-distribution, Large = OOD.
        """
        return self._mahalanobis_distance(state)
    
    def calibrate(self, states: List[np.ndarray], verbose: bool = False):
        """
        Compute feature-space statistics and conformal scores on calibration states.
        """
        states_array = np.array(states)
        
        # Compute mean and covariance of training features
        self._feature_mean = np.mean(states_array, axis=0)
        cov = np.cov(states_array, rowvar=False)
        
        # Regularize covariance for numerical stability
        cov += np.eye(cov.shape[0]) * 1e-4
        self._feature_cov_inv = np.linalg.inv(cov)
        
        # Compute calibration scores (Mahalanobis distances)
        scores = [self._mahalanobis_distance(s) for s in states]
        self.calibration_scores = np.array(scores)
        
        if verbose:
            print(f"    Conformal calibration (Mahalanobis): {len(scores)} states, "
                  f"dist mean={np.mean(scores):.3f}, "
                  f"95th pct={np.percentile(scores, 95):.3f}")
    
    def get_gate_value(self, state: np.ndarray) -> float:
        """
        Gate value based on Mahalanobis distance.
        Returns 1.0 if in-distribution, decreases for OOD.
        """
        if len(self.calibration_scores) == 0:
            return 1.0
        
        score = self._conformal_score(state)
        threshold = np.percentile(self.calibration_scores, 95)
        
        if score <= threshold:
            return 1.0  # Clearly in-distribution
        else:
            # Gate closes proportionally to how far OOD
            excess = (score - threshold) / (threshold + 1e-7)
            return float(max(0.0, 1.0 / (1.0 + excess)))
    
    def extract(self, state: np.ndarray) -> np.ndarray:
        """Gated concept vector using conformal gating."""
        gate = self.get_gate_value(state)
        probs = self._soft_probs(state)
        return self.global_mean + gate * (probs - self.global_mean)
    
    def extract_with_gate(self, state: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Return (gated_concepts, gate_values_per_concept)."""
        probs = self._soft_probs(state)
        gate = self.get_gate_value(state)
        gated_probs = self.global_mean + gate * (probs - self.global_mean)
        gate_values = np.full(self.n_concepts, gate)
        return gated_probs, gate_values
    
    def __call__(self, state: np.ndarray) -> np.ndarray:
        return self.extract(state)
    
    def train_on_trajectories(self, trajectories: List[List[Dict]],
                              hard_concepts: MIMICHardConcepts,
                              epochs: int = 100, lr: float = 0.01,
                              verbose: bool = False):
        """Train encoder, compute global mean, and calibrate."""
        # Store hard_concepts reference
        self._hard_concepts = hard_concepts
        
        X, Y = [], []
        cal_states = []
        
        for traj in trajectories:
            for step in traj:
                state = step['state']
                labels = hard_concepts.extract(state)
                X.append(state)
                Y.append(labels)
                cal_states.append(state)
        
        X = np.array(X)
        Y = np.array(Y)
        
        # Train encoder
        for epoch in range(epochs):
            perm = np.random.permutation(len(X))
            X_shuffled = X[perm]
            Y_shuffled = Y[perm]
            
            total_loss = 0
            n_batches = 0
            
            for i in range(0, len(X), 32):
                X_batch = X_shuffled[i:i+32]
                Y_batch = Y_shuffled[i:i+32]
                loss = self.encoder.train_step(X_batch, Y_batch, lr=lr)
                total_loss += loss
                n_batches += 1
            
            if verbose and (epoch + 1) % 50 == 0:
                print(f"    Epoch {epoch+1}/{epochs}, Loss: {total_loss/n_batches:.4f}")
        
        # Compute global mean
        all_probs = []
        for x in X:
            probs, _ = self.encoder.forward(x)
            all_probs.append(probs)
        self.global_mean = np.mean(all_probs, axis=0)
        
        # Calibrate on training states (feature-space Mahalanobis)
        self.calibrate(cal_states, verbose=verbose)
        
        if verbose:
            print(f"    Conformal global mean: {self.global_mean.round(3)}")


# =============================================================================
# MIMIC Policies
# =============================================================================

class MIMICBehaviorPolicy:
    """
    Behavior policy estimated from MIMIC data.
    Learns P(action | state) from observed clinician behavior.
    """
    
    def __init__(self, n_actions: int = 25, n_state_bins: int = 10):
        self.n_actions = n_actions
        self.n_state_bins = n_state_bins
        
        # Simple: bin first feature and learn action distribution per bin
        self.action_counts = np.ones((n_state_bins, n_actions))  # Laplace smoothing
        self.action_probs = None
    
    def fit(self, trajectories: List[List[Dict]]):
        """Learn action distribution from trajectories."""
        for traj in trajectories:
            for step in traj:
                state = step['state']
                action = step['action']
                
                # Bin by heart rate (index 0) - a commonly measured vital
                bin_idx = min(int(state[0] * self.n_state_bins), self.n_state_bins - 1)
                self.action_counts[bin_idx, action] += 1
        
        # Normalize
        self.action_probs = self.action_counts / self.action_counts.sum(axis=1, keepdims=True)
    
    def prob(self, state: np.ndarray, action: int) -> float:
        """Return probability of action given state."""
        if self.action_probs is None:
            return 1.0 / self.n_actions
        
        bin_idx = min(int(state[0] * self.n_state_bins), self.n_state_bins - 1)
        return self.action_probs[bin_idx, action]
    
    def action_probs_for_state(self, state: np.ndarray) -> np.ndarray:
        if self.action_probs is None:
            return np.ones(self.n_actions) / self.n_actions
        
        bin_idx = min(int(state[0] * self.n_state_bins), self.n_state_bins - 1)
        return self.action_probs[bin_idx]


class MIMICEvaluationPolicy:
    """
    Evaluation policy for MIMIC.
    Uses a simple rule-based policy that favors certain actions for high-risk states.
    """
    
    def __init__(self, n_actions: int = 25, aggressiveness: float = 0.3):
        self.n_actions = n_actions
        self.aggressiveness = aggressiveness
        
        # Prefer higher fluid/vaso actions for high-risk patients
        # Action = fluid_bin * 5 + vaso_bin (5x5 grid)
        self.high_risk_prefs = np.zeros(n_actions)
        for fluid in range(5):
            for vaso in range(5):
                action = fluid * 5 + vaso
                # Prefer moderate-high treatment
                self.high_risk_prefs[action] = (fluid + vaso) / 8.0
        
        self.high_risk_prefs = self.high_risk_prefs / self.high_risk_prefs.sum()
        
        # Low risk: uniform
        self.low_risk_prefs = np.ones(n_actions) / n_actions
    
    def is_high_risk(self, state: np.ndarray) -> bool:
        """Determine if patient is high risk."""
        # High creatinine (idx 9) OR high lactate (idx 8) OR low SpO2 (idx 5)
        return state[9] > 0.5 or state[8] > 0.5 or state[5] < 0.7
    
    def prob(self, state: np.ndarray, action: int) -> float:
        """Return probability of action given state."""
        if self.is_high_risk(state):
            base_probs = self.high_risk_prefs
        else:
            base_probs = self.low_risk_prefs
        
        # Mix with uniform for exploration
        mixed = (1 - self.aggressiveness) * (np.ones(self.n_actions) / self.n_actions) + \
                self.aggressiveness * base_probs
        
        return mixed[action]


class MIMICConceptBasedPolicy:
    """
    Policy conditioned on concepts for MIMIC.
    """
    
    def __init__(self, concept_extractor, n_concepts: int = 32, n_actions: int = 25):
        self.concept_extractor = concept_extractor
        self.n_concepts = n_concepts
        self.n_actions = n_actions
        self.policy_table = np.ones((n_concepts, n_actions)) / n_actions
    
    def state_to_concept_index(self, state: np.ndarray) -> int:
        """Map state to concept index."""
        concept_vec = self.concept_extractor(state)
        if len(concept_vec) > 5:
            concept_vec = concept_vec[:5]
        binary = (concept_vec > 0.5).astype(int)
        idx = sum(b * (2 ** i) for i, b in enumerate(binary))
        return min(idx, self.n_concepts - 1)
    
    def learn_from_trajectories(self, trajectories: List[List[Dict]], smoothing: float = 1.0):
        """Learn concept-based policy from trajectories."""
        counts = np.ones((self.n_concepts, self.n_actions)) * smoothing
        
        for traj in trajectories:
            for step in traj:
                state = step['state']
                action = step['action']
                c_idx = self.state_to_concept_index(state)
                counts[c_idx, action] += 1
        
        self.policy_table = counts / counts.sum(axis=1, keepdims=True)
    
    def prob(self, state: np.ndarray, action: int) -> float:
        c_idx = self.state_to_concept_index(state)
        return self.policy_table[c_idx, action]


class MIMICGatedConceptBasedPolicy:
    """
    Gated concept-based policy for MIMIC.
    """
    
    def __init__(self, gated_extractor, n_concepts: int = 32, n_actions: int = 25):
        self.gated_extractor = gated_extractor
        self.n_concepts = n_concepts
        self.n_actions = n_actions
        self.policy_table = np.ones((n_concepts, n_actions)) / n_actions
        self.global_policy = np.ones(n_actions) / n_actions
    
    def state_to_concept_index(self, state: np.ndarray) -> int:
        concept_vec = self.gated_extractor(state)
        if len(concept_vec) > 5:
            concept_vec = concept_vec[:5]
        binary = (concept_vec > 0.5).astype(int)
        return int(sum(b * (2 ** i) for i, b in enumerate(binary)))
    
    def learn_from_trajectories(self, trajectories: List[List[Dict]], smoothing: float = 1.0):
        counts = np.ones((self.n_concepts, self.n_actions)) * smoothing
        
        for traj in trajectories:
            for step in traj:
                c = self.state_to_concept_index(step['state'])
                counts[c, step['action']] += 1
        
        self.policy_table = counts / counts.sum(axis=1, keepdims=True)
        total = counts.sum(axis=0)
        self.global_policy = total / total.sum()
    
    def prob(self, state: np.ndarray, action: int) -> float:
        gate = self.gated_extractor.get_gate_value(state)
        c = self.state_to_concept_index(state)
        concept_prob = float(self.policy_table[c, action])
        global_prob = float(self.global_policy[action])
        return gate * concept_prob + (1 - gate) * global_prob


class MIMICConceptConditionedEvalPolicy:
    """
    Evaluation policy that uses concept representations to modify action selection.
    The key insight: soft concepts leak state information, causing the policy
    to behave differently when concepts are extracted from OOD states.
    """
    
    def __init__(self, concept_extractor, base_eval_policy, n_actions: int = 25):
        self.concept_extractor = concept_extractor
        self.base_policy = base_eval_policy
        self.n_actions = n_actions
        self.n_concept_bins = 32  # 2^5 for 5 binary concepts
    
    def state_to_concept_index(self, state: np.ndarray) -> int:
        """Map state to concept index."""
        concept_vec = self.concept_extractor(state)
        if len(concept_vec) > 5:
            concept_vec = concept_vec[:5]
        binary = (concept_vec > 0.5).astype(int)
        idx = sum(b * (2 ** i) for i, b in enumerate(binary))
        return min(idx, self.n_concept_bins - 1)
    
    def prob(self, state: np.ndarray, action: int) -> float:
        """
        Return probability under concept-conditioned evaluation policy.
        
        Uses the FULL concept representation (including hidden dims for soft).
        For soft concepts with leakage, the hidden dimensions encode raw state
        information that changes the effective policy behavior when OOD.
        """
        # Get full concept representation
        concept_vec = self.concept_extractor(state)
        
        # Compute risk score from concepts
        # For soft concepts, the hidden dims (5:37) also contribute
        if len(concept_vec) >= 5:
            # Primary concepts: high_creatinine, low_spo2, high_lactate, low_gcs, tachycardia
            primary_risk = (concept_vec[0] + concept_vec[2] + concept_vec[4]) / 3
            
            # For soft concepts, hidden embedding also affects behavior
            if len(concept_vec) > 5:
                hidden = concept_vec[5:]
                hidden_signal = np.mean(np.abs(hidden - 0.5))  # Deviation from neutral
                risk_score = 0.6 * primary_risk + 0.4 * hidden_signal
            else:
                risk_score = primary_risk
        else:
            risk_score = 0.5
        
        risk_score = np.clip(risk_score, 0, 1)
        
        # Higher risk -> prefer more aggressive treatment (higher action indices)
        action_aggressiveness = action / (self.n_actions - 1)
        
        # Construct probability favoring aggressive treatment for high-risk
        if risk_score > 0.5:
            # High risk: favor high actions
            prob = 0.02 + 0.08 * (action_aggressiveness ** (1 / (risk_score + 0.5)))
        else:
            # Low risk: favor low actions
            prob = 0.02 + 0.08 * ((1 - action_aggressiveness) ** (1 / (1 - risk_score + 0.5)))
        
        return np.clip(prob, 0.01, 0.99)


class MIMICGatedConceptConditionedEvalPolicy:
    """
    Evaluation policy using gated concepts.
    The gating mechanism reduces OOD leakage effects by falling back
    to global behavior when concept uncertainty is high.
    """
    
    def __init__(self, gated_extractor, base_eval_policy, n_actions: int = 25):
        self.gated_extractor = gated_extractor
        self.base_policy = base_eval_policy
        self.n_actions = n_actions
        self.n_concept_bins = 32
        self.global_mean_risk = 0.5  # Learned global fallback
    
    def prob(self, state: np.ndarray, action: int) -> float:
        """
        Return probability under gated concept-conditioned policy.
        
        The gate controls how much we trust concept-specific decisions.
        When OOD (low gate), fall back to global behavior.
        """
        concept_vec, gate_values = self.gated_extractor.extract_with_gate(state)
        mean_gate = np.mean(gate_values)
        
        # Compute risk score from gated concepts (only primary 5)
        if len(concept_vec) >= 5:
            risk_score = (concept_vec[0] + concept_vec[2] + concept_vec[4]) / 3
        else:
            risk_score = 0.5
        
        risk_score = np.clip(risk_score, 0, 1)
        
        # Gated risk: blend concept-specific with global mean
        gated_risk = mean_gate * risk_score + (1 - mean_gate) * self.global_mean_risk
        
        action_aggressiveness = action / (self.n_actions - 1)
        
        if gated_risk > 0.5:
            prob = 0.02 + 0.08 * (action_aggressiveness ** (1 / (gated_risk + 0.5)))
        else:
            prob = 0.02 + 0.08 * ((1 - action_aggressiveness) ** (1 / (1 - gated_risk + 0.5)))
        
        return np.clip(prob, 0.01, 0.99)


# =============================================================================
# OPE Estimators
# =============================================================================

def cpdis_estimate_by_horizon(
    trajectories: List[List[Dict]],
    behavior_policy,
    eval_policy,
    max_horizon: int,
    gamma: float = 0.99
) -> Tuple[float, float, float]:
    """
    Concept-based PDIS up to a specific horizon.
    Returns: (estimate, variance, effective_sample_size)
    """
    returns = []
    final_rhos = []
    
    for traj in trajectories:
        G = 0.0
        rho_cumulative = 1.0
        T = min(len(traj), max_horizon)
        
        for t in range(T):
            step = traj[t]
            state = step['state']
            action = step['action']
            reward = step['reward']
            
            pi_e = eval_policy.prob(state, action)
            pi_b = behavior_policy.prob(state, action)
            
            if pi_b < 1e-10:
                rho_t = 0.0
            else:
                rho_t = pi_e / pi_b
            
            rho_cumulative *= rho_t
            G += (gamma ** t) * rho_cumulative * reward
        
        returns.append(G)
        final_rhos.append(rho_cumulative)
    
    returns = np.array(returns)
    final_rhos = np.array(final_rhos)
    estimate = np.mean(returns)
    variance = np.var(returns)
    
    if np.sum(final_rhos ** 2) > 0:
        ess = (np.sum(final_rhos) ** 2) / np.sum(final_rhos ** 2)
    else:
        ess = 0
    
    return estimate, variance, ess


# =============================================================================
# Leakage Measurement
# =============================================================================

def train_probe(concept_extractor, states: List[np.ndarray], 
                features: List[np.ndarray]) -> Ridge:
    """Train linear probe to predict features from concept embeddings."""
    embeddings = np.array([concept_extractor(s) for s in states])
    features = np.array(features)
    
    probe = Ridge(alpha=1.0)
    probe.fit(embeddings, features)
    return probe


def evaluate_probe(probe, concept_extractor, states: List[np.ndarray],
                   features: List[np.ndarray]) -> float:
    """Evaluate probe R² on new data."""
    embeddings = np.array([concept_extractor(s) for s in states])
    features = np.array(features)
    return probe.score(embeddings, features)


def measure_leakage(concept_extractor, states: List[np.ndarray],
                    features: List[np.ndarray]) -> float:
    """Measure leakage via cross-validated R²."""
    embeddings = np.array([concept_extractor(s) for s in states])
    features = np.array(features)
    
    probe = Ridge(alpha=1.0)
    try:
        scores = cross_val_score(probe, embeddings, features,
                                cv=min(5, len(embeddings) // 10 + 1),
                                scoring='r2')
        return max(0, scores.mean())
    except:
        return 0.0


# =============================================================================
# Main Experiment
# =============================================================================

def run_mimic_experiment(
    data_path: str = './mimic_iv_trajectories.npz',
    train_horizon: int = 8,
    test_horizons: List[int] = None,
    seed: int = 42
) -> Dict:
    """
    Run the full temporal leakage experiment on MIMIC data.
    """
    if test_horizons is None:
        test_horizons = [4, 6, 8, 10, 12, 14, 16, 18]
    
    np.random.seed(seed)
    
    print("=" * 70)
    print("MIMIC-III TEMPORAL LEAKAGE EXPERIMENT")
    print("=" * 70)
    
    # =========================================================================
    # Load Data
    # =========================================================================
    print("\n[1] Loading MIMIC data...")
    mimic_data = MIMICData.load(data_path)
    trajectories = mimic_data.to_trajectories()
    
    print(f"    Loaded {len(trajectories)} trajectories")
    print(f"    Features: {mimic_data.feature_names}")
    print(f"    Actions: {mimic_data.n_actions}")
    print(f"    Mortality rate: {mimic_data.died.mean()*100:.1f}%")
    print(f"    Mean trajectory length: {mimic_data.lengths.mean():.1f}")
    
    # =========================================================================
    # Setup Concepts
    # =========================================================================
    print("\n[2] Setting up concept extractors...")
    hard_concepts = MIMICHardConcepts(n_features=18)
    soft_concepts = MIMICSoftConcepts(n_features=18, use_leakage=True, seed=seed)
    gated_concepts = MIMICGatedSoftConcepts(n_features=18, seed=seed)
    conformal_concepts = MIMICConformalGatedConcepts(n_features=18, seed=seed)
    
    # Train on SURVIVORS ONLY (early timesteps) to induce distribution shift
    # This creates OOD when evaluating on patients who die
    train_trajs = []
    n_survivors = 0
    n_deaths = 0
    
    for traj in trajectories:
        # Check if patient survived (died flag is in first step)
        patient_died = traj[0].get('died', False)
        
        if not patient_died:  # Survivor - use for training
            early_steps = [s for i, s in enumerate(traj) if i < train_horizon]
            if len(early_steps) > 0:
                train_trajs.append(early_steps)
            n_survivors += 1
        else:
            n_deaths += 1
    
    print(f"    Training on SURVIVORS ONLY: {n_survivors} survivors, {n_deaths} deaths excluded")
    print(f"    This induces distribution shift when evaluating on dying patients")
    
    print(f"\n    Training soft concepts on t < {train_horizon} (survivors)...")
    soft_concepts.train_on_trajectories(train_trajs, hard_concepts, epochs=200, verbose=True)
    
    print(f"    Training gated concepts on t < {train_horizon} (survivors)...")
    gated_concepts.train_on_trajectories(train_trajs, hard_concepts, epochs=200, verbose=True)
    
    print(f"    Training conformal concepts on t < {train_horizon} (survivors)...")
    conformal_concepts.train_on_trajectories(train_trajs, hard_concepts, epochs=200, verbose=True)
    
    # =========================================================================
    # Setup Policies
    # =========================================================================
    print("\n[3] Setting up policies...")
    
    behavior_policy = MIMICBehaviorPolicy(n_actions=mimic_data.n_actions)
    behavior_policy.fit(trajectories)
    
    eval_policy = MIMICEvaluationPolicy(n_actions=mimic_data.n_actions)
    
    # Concept-based policies
    # Behavior policy: learned from data (same for all, using hard concepts)
    behavior_policy = MIMICBehaviorPolicy(n_actions=mimic_data.n_actions)
    behavior_policy.fit(trajectories)
    
    # Evaluation policy: a different policy we want to evaluate
    # Use a more aggressive treatment policy for high-risk patients
    eval_policy = MIMICEvaluationPolicy(n_actions=mimic_data.n_actions)
    
    # Concept-based behavior policies (for CPDIS - these approximate the behavior)
    hard_behavior = MIMICConceptBasedPolicy(hard_concepts, n_actions=mimic_data.n_actions)
    soft_behavior = MIMICConceptBasedPolicy(soft_concepts, n_actions=mimic_data.n_actions)
    gated_behavior = MIMICGatedConceptBasedPolicy(gated_concepts, n_actions=mimic_data.n_actions)
    conformal_behavior = MIMICGatedConceptBasedPolicy(conformal_concepts, n_actions=mimic_data.n_actions)
    
    hard_behavior.learn_from_trajectories(trajectories)
    soft_behavior.learn_from_trajectories(trajectories)
    gated_behavior.learn_from_trajectories(trajectories)
    conformal_behavior.learn_from_trajectories(trajectories)
    
    # Concept-based evaluation policies (wrap the eval policy with concept conditioning)
    hard_eval = MIMICConceptConditionedEvalPolicy(hard_concepts, eval_policy, n_actions=mimic_data.n_actions)
    soft_eval = MIMICConceptConditionedEvalPolicy(soft_concepts, eval_policy, n_actions=mimic_data.n_actions)
    gated_eval = MIMICGatedConceptConditionedEvalPolicy(gated_concepts, eval_policy, n_actions=mimic_data.n_actions)
    conformal_eval = MIMICGatedConceptConditionedEvalPolicy(conformal_concepts, eval_policy, n_actions=mimic_data.n_actions)
    
    # =========================================================================
    # EXPERIMENT 1: Temporal Leakage Degradation
    # =========================================================================
    print("\n[4] EXPERIMENT 1: Measuring temporal leakage degradation...")
    print("    (Probes trained on survivors, evaluated on ALL patients including deaths)")
    
    leakage_timesteps = [2, 4, 6, 8, 10, 12, 14, 16]
    
    # Collect states separately for survivors and deaths
    states_by_t_survivors = {t: [] for t in leakage_timesteps}
    features_by_t_survivors = {t: [] for t in leakage_timesteps}
    states_by_t_all = {t: [] for t in leakage_timesteps}
    features_by_t_all = {t: [] for t in leakage_timesteps}
    
    for traj in trajectories:
        patient_died = traj[0].get('died', False)
        for t, step in enumerate(traj):
            if t in states_by_t_all:
                states_by_t_all[t].append(step['state'])
                features_by_t_all[t].append(step['features'])
                
                if not patient_died:  # Survivor
                    states_by_t_survivors[t].append(step['state'])
                    features_by_t_survivors[t].append(step['features'])
    
    # Train probes on SURVIVORS ONLY (early timesteps)
    train_states = []
    train_features = []
    for t in leakage_timesteps:
        if t < train_horizon:
            train_states.extend(states_by_t_survivors[t])
            train_features.extend(features_by_t_survivors[t])
    
    print(f"    Training probes on {len(train_states)} survivor samples from t < {train_horizon}...")
    probe_soft = train_probe(soft_concepts, train_states, train_features)
    probe_gated = train_probe(gated_concepts, train_states, train_features)
    
    leakage_results = {
        'timesteps': [],
        'hard_r2': [],
        'soft_r2': [],
        'gated_r2': [],
        'n_samples': []
    }
    
    print(f"\n    {'Timestep':<10} {'In-Dist?':<10} {'Hard R²':<12} {'Soft R²':<12} {'Gated R²':<12} {'N':<8}")
    print("    " + "-" * 64)
    
    # Evaluate on ALL patients (including deaths = OOD)
    for t in leakage_timesteps:
        if len(states_by_t_all[t]) < 10:
            print(f"    t={t}: Skipping (only {len(states_by_t_all[t])} samples)")
            continue
        
        r2_hard = measure_leakage(hard_concepts, states_by_t_all[t], features_by_t_all[t])
        r2_soft = evaluate_probe(probe_soft, soft_concepts, states_by_t_all[t], features_by_t_all[t])
        r2_gated = evaluate_probe(probe_gated, gated_concepts, states_by_t_all[t], features_by_t_all[t])
        
        in_dist = "Yes" if t < train_horizon else "No"
        n_samples = len(states_by_t_all[t])
        
        leakage_results['timesteps'].append(t)
        leakage_results['hard_r2'].append(r2_hard)
        leakage_results['soft_r2'].append(r2_soft)
        leakage_results['gated_r2'].append(r2_gated)
        leakage_results['n_samples'].append(n_samples)
        
        print(f"    t={t:<8} {in_dist:<10} {r2_hard:<12.4f} {r2_soft:<12.4f} {r2_gated:<12.4f} {n_samples:<8}")
    
    # =========================================================================
    # EXPERIMENT 2 & 3: OPE Error vs Horizon
    # =========================================================================
    print(f"\n[5] EXPERIMENTS 2 & 3: OPE error vs horizon...")
    
    ope_results = {
        'horizons': test_horizons,
        'hard_cpdis': {'estimates': [], 'variances': [], 'ess': []},
        'soft_cpdis': {'estimates': [], 'variances': [], 'ess': []},
        'gated_cpdis': {'estimates': [], 'variances': [], 'ess': []},
        'conformal_cpdis': {'estimates': [], 'variances': [], 'ess': []},
    }
    
    print(f"\n    {'Horizon':<10} {'Hard Est':<12} {'Soft Est':<12} {'Entropy Est':<12} {'Conformal Est':<12}")
    print("    " + "-" * 60)
    
    for h in test_horizons:
        # Hard concept CPDIS
        est_h, var_h, ess_h = cpdis_estimate_by_horizon(
            trajectories, hard_behavior, hard_eval, h)
        ope_results['hard_cpdis']['estimates'].append(est_h)
        ope_results['hard_cpdis']['variances'].append(var_h)
        ope_results['hard_cpdis']['ess'].append(ess_h)
        
        # Soft concept CPDIS
        est_s, var_s, ess_s = cpdis_estimate_by_horizon(
            trajectories, soft_behavior, soft_eval, h)
        ope_results['soft_cpdis']['estimates'].append(est_s)
        ope_results['soft_cpdis']['variances'].append(var_s)
        ope_results['soft_cpdis']['ess'].append(ess_s)
        
        # Entropy-Gated concept CPDIS
        est_g, var_g, ess_g = cpdis_estimate_by_horizon(
            trajectories, gated_behavior, gated_eval, h)
        ope_results['gated_cpdis']['estimates'].append(est_g)
        ope_results['gated_cpdis']['variances'].append(var_g)
        ope_results['gated_cpdis']['ess'].append(ess_g)
        
        # Conformal-Gated concept CPDIS
        est_c, var_c, ess_c = cpdis_estimate_by_horizon(
            trajectories, conformal_behavior, conformal_eval, h)
        ope_results['conformal_cpdis']['estimates'].append(est_c)
        ope_results['conformal_cpdis']['variances'].append(var_c)
        ope_results['conformal_cpdis']['ess'].append(ess_c)
        
        print(f"    T={h:<8} {est_h:<12.4f} {est_s:<12.4f} {est_g:<12.4f} {est_c:<12.4f}")
    
    # =========================================================================
    # Gate Values Analysis
    # =========================================================================
    print("\n[6] Gate values over time...")
    print(f"\n    {'t':<6} {'in-dist':<10} {'entropy gate':<14} {'conformal gate':<14}")
    print("    " + "-" * 48)
    
    gate_results = {
        'timesteps': [], 
        'entropy_mean': [], 'entropy_std': [],
        'conformal_mean': [], 'conformal_std': []
    }
    
    for t in leakage_timesteps:
        if len(states_by_t_all[t]) < 10:
            continue
        entropy_gates = [gated_concepts.get_gate_value(s) for s in states_by_t_all[t]]
        conformal_gates = [conformal_concepts.get_gate_value(s) for s in states_by_t_all[t]]
        in_dist = "Yes" if t < train_horizon else "No"
        
        gate_results['timesteps'].append(t)
        gate_results['entropy_mean'].append(np.mean(entropy_gates))
        gate_results['entropy_std'].append(np.std(entropy_gates))
        gate_results['conformal_mean'].append(np.mean(conformal_gates))
        gate_results['conformal_std'].append(np.std(conformal_gates))
        
        print(f"    {t:<6} {in_dist:<10} {np.mean(entropy_gates):<14.3f} {np.mean(conformal_gates):<14.3f}")
    
    # Combine all results
    results = {
        'leakage': leakage_results,
        'ope': ope_results,
        'gates': gate_results,
        'train_horizon': train_horizon,
        'n_trajectories': len(trajectories),
        'mortality_rate': mimic_data.died.mean()
    }
    
    return results


def plot_mimic_results(results: Dict, save_path: str = None):
    """Plot MIMIC experiment results."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Colors (matching GridWorld experiment)
    HARD_COLOR = '#27ae60'      # green
    SOFT_COLOR = '#e74c3c'      # red
    ENTROPY_COLOR = '#2980b9'   # blue
    CONFORMAL_COLOR = '#8e44ad' # purple
    
    train_horizon = results['train_horizon']
    
    # =========================================================================
    # Panel 1: Leakage R² over time
    # =========================================================================
    ax1 = axes[0, 0]
    leakage = results['leakage']
    timesteps = leakage['timesteps']
    
    ax1.axvspan(min(timesteps), train_horizon, alpha=0.15, color='green')
    ax1.axvspan(train_horizon, max(timesteps) + 1, alpha=0.15, color='red')
    ax1.axvline(x=train_horizon, color='gray', linestyle='--', alpha=0.7, 
                label=f'Train horizon (t={train_horizon})')
    
    ax1.plot(timesteps, leakage['hard_r2'], 'o-', color=HARD_COLOR, 
             linewidth=2, markersize=8, label='Hard concepts')
    ax1.plot(timesteps, leakage['soft_r2'], 's-', color=SOFT_COLOR, 
             linewidth=2, markersize=8, label='Soft concepts')
    ax1.plot(timesteps, leakage['gated_r2'], '^-', color=ENTROPY_COLOR, 
             linewidth=2, markersize=8, label='Gated concepts')
    
    ax1.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    
    ax1.set_xlabel('Timestep t (4-hour bins)', fontsize=12)
    ax1.set_ylabel('Probe R² (Leakage)', fontsize=12)
    ax1.set_title('(a) Exp 1: Information Leakage Over Time', fontsize=13, fontweight='bold')
    ax1.legend(loc='lower left', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # =========================================================================
    # Panel 2: OPE Estimates over horizon (all 4 methods)
    # =========================================================================
    ax2 = axes[0, 1]
    ope = results['ope']
    horizons = ope['horizons']
    
    ax2.axvspan(min(horizons), train_horizon, alpha=0.15, color='green')
    ax2.axvspan(train_horizon, max(horizons) + 1, alpha=0.15, color='red')
    ax2.axvline(x=train_horizon, color='gray', linestyle='--', alpha=0.7)
    
    ax2.plot(horizons, ope['hard_cpdis']['estimates'], 'o-', color=HARD_COLOR,
             linewidth=2, markersize=8, label='Hard CPDIS')
    ax2.plot(horizons, ope['soft_cpdis']['estimates'], 's-', color=SOFT_COLOR,
             linewidth=2, markersize=8, label='Soft CPDIS')
    ax2.plot(horizons, ope['gated_cpdis']['estimates'], '^-', color=ENTROPY_COLOR,
             linewidth=2, markersize=8, label='Gated CPDIS (Entropy)')
    ax2.plot(horizons, ope['conformal_cpdis']['estimates'], 'D-', color=CONFORMAL_COLOR,
             linewidth=2, markersize=8, label='Gated CPDIS (Conformal)')
    
    ax2.set_xlabel('Trajectory Horizon T', fontsize=12)
    ax2.set_ylabel('OPE Estimate', fontsize=12)
    ax2.set_title('(b) Exp 2+3: OPE Estimates vs Horizon', fontsize=13, fontweight='bold')
    ax2.legend(loc='best', fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # =========================================================================
    # Panel 3: OPE Variance over horizon
    # =========================================================================
    ax3 = axes[1, 0]
    
    ax3.axvspan(min(horizons), train_horizon, alpha=0.15, color='green')
    ax3.axvspan(train_horizon, max(horizons) + 1, alpha=0.15, color='red')
    ax3.axvline(x=train_horizon, color='gray', linestyle='--', alpha=0.7)
    
    ax3.plot(horizons, ope['hard_cpdis']['variances'], 'o-', color=HARD_COLOR,
             linewidth=2, markersize=8, label='Hard CPDIS')
    ax3.plot(horizons, ope['soft_cpdis']['variances'], 's-', color=SOFT_COLOR,
             linewidth=2, markersize=8, label='Soft CPDIS')
    ax3.plot(horizons, ope['gated_cpdis']['variances'], '^-', color=ENTROPY_COLOR,
             linewidth=2, markersize=8, label='Gated CPDIS (Entropy)')
    ax3.plot(horizons, ope['conformal_cpdis']['variances'], 'D-', color=CONFORMAL_COLOR,
             linewidth=2, markersize=8, label='Gated CPDIS (Conformal)')
    
    ax3.set_xlabel('Trajectory Horizon T', fontsize=12)
    ax3.set_ylabel('OPE Variance', fontsize=12)
    ax3.set_title('(c) OPE Variance vs Horizon', fontsize=13, fontweight='bold')
    ax3.legend(loc='best', fontsize=10)
    ax3.set_yscale('log')
    ax3.grid(True, alpha=0.3)
    
    # =========================================================================
    # Panel 4: Gate values over time (Entropy vs Conformal)
    # =========================================================================
    ax4 = axes[1, 1]
    gates = results['gates']
    
    ax4.axvspan(min(gates['timesteps']), train_horizon, alpha=0.15, color='green')
    ax4.axvspan(train_horizon, max(gates['timesteps']) + 1, alpha=0.15, color='red')
    ax4.axvline(x=train_horizon, color='gray', linestyle='--', alpha=0.7,
                label=f'Train horizon')
    
    ax4.errorbar(gates['timesteps'], gates['entropy_mean'], 
                 yerr=gates['entropy_std'], fmt='o-', color=ENTROPY_COLOR,
                 linewidth=2, markersize=8, capsize=5, label='Entropy gate')
    ax4.errorbar(gates['timesteps'], gates['conformal_mean'], 
                 yerr=gates['conformal_std'], fmt='D-', color=CONFORMAL_COLOR,
                 linewidth=2, markersize=8, capsize=5, label='Conformal gate')
    
    ax4.axhline(y=1.0, color='green', linestyle=':', alpha=0.5, label='Fully open')
    ax4.axhline(y=0.0, color='red', linestyle=':', alpha=0.5, label='Fully closed')
    
    ax4.set_xlabel('Timestep t (4-hour bins)', fontsize=12)
    ax4.set_ylabel('Gate Value', fontsize=12)
    ax4.set_title('(d) Entropy Gate Closes for OOD States', fontsize=13, fontweight='bold')
    ax4.legend(loc='best', fontsize=10)
    ax4.set_ylim([-0.1, 1.1])
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nFigure saved to: {save_path}")
    
    plt.show()
    return fig


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    import os
    
    # Run experiment
    results = run_mimic_experiment(
        data_path='./mimic_iv_trajectories.npz',
        train_horizon=8,
        test_horizons=[4, 6, 8, 10, 12, 14, 16, 18],
        seed=42
    )
    
    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    print(f"\nDataset: {results['n_trajectories']} ICU stays, "
          f"{results['mortality_rate']*100:.1f}% mortality")
    print(f"Train horizon: t < {results['train_horizon']} (32 hours)")
    
    print("\nLeakage degradation (R² drops after train horizon):")
    leakage = results['leakage']
    for i, t in enumerate(leakage['timesteps']):
        in_dist = "ID" if t < results['train_horizon'] else "OOD"
        print(f"  t={t} ({in_dist}): Hard={leakage['hard_r2'][i]:.3f}, "
              f"Soft={leakage['soft_r2'][i]:.3f}, Gated={leakage['gated_r2'][i]:.3f}")
    
    # Plot results
    plot_mimic_results(results, save_path='mimic_results.png')
    
    # Save results
    np.save('mimic_experiment_results.npy', results)
    print(f"\nResults saved to mimic_experiment_results.npy")