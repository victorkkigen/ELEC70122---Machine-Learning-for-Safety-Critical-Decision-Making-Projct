"""
Concept Bottleneck Layers for OPE

Two types of concepts:
1. Hard Concepts: Binary/discrete values based on interpretable rules
2. Soft Concepts: Learned embeddings that can leak extra information

The key insight from Espinosa Zarlenga et al. (2025):
- Soft concepts "leak" information beyond the concept meaning
- This leakage becomes corrupted under distribution shift (OOD inputs)
- We hypothesize this corruption compounds over time in sequential OPE
"""

import numpy as np
from typing import List, Tuple, Optional


class HardConcepts:
    """
    Hard concepts: Deterministic, interpretable rules.
    
    No leakage because concepts are computed from explicit rules,
    not learned embeddings.
    
    Concepts for Gridworld:
    1. near_goal: 1 if Manhattan distance to goal <= 2
    2. in_wind_zone: 1 if current column has wind > 0
    3. near_wall: 1 if adjacent to any wall
    4. high_wind: 1 if wind strength >= 2
    5. goal_reachable_soon: 1 if distance <= 4
    """
    
    def __init__(self, env):
        self.env = env
        self.n_concepts = 5
        self.concept_names = [
            'near_goal',
            'in_wind_zone', 
            'near_wall',
            'high_wind',
            'goal_reachable_soon'
        ]
    
    def __call__(self, state: np.ndarray) -> np.ndarray:
        """Compute hard concepts for a state."""
        row, col = int(state[0]), int(state[1])
        
        # Concept 1: Near goal (Manhattan distance <= 2)
        goal_dist = abs(row - self.env.goal[0]) + abs(col - self.env.goal[1])
        near_goal = 1.0 if goal_dist <= 2 else 0.0
        
        # Concept 2: In wind zone
        wind = self.env.wind[min(col, len(self.env.wind) - 1)]
        in_wind_zone = 1.0 if wind > 0 else 0.0
        
        # Concept 3: Near wall (adjacent to boundary)
        near_wall = 1.0 if (row == 0 or row == self.env.height - 1 or 
                           col == 0 or col == self.env.width - 1) else 0.0
        
        # Concept 4: High wind
        high_wind = 1.0 if wind >= 2 else 0.0
        
        # Concept 5: Goal reachable soon (within 4 steps optimally)
        goal_reachable_soon = 1.0 if goal_dist <= 4 else 0.0
        
        return np.array([near_goal, in_wind_zone, near_wall, high_wind, goal_reachable_soon], 
                        dtype=np.float32)
    
    def batch_forward(self, states: np.ndarray) -> np.ndarray:
        """Compute concepts for a batch of states."""
        return np.array([self(s) for s in states])


class SoftConceptEncoder:
    """
    Soft concept encoder: Simple MLP that outputs concept embeddings.
    
    Numpy-based implementation.
    
    Unlike hard concepts, soft concepts:
    - Are learned from data
    - Can encode information beyond the concept definition (leakage)
    - The leakage becomes corrupted under distribution shift
    """
    
    def __init__(
        self,
        input_dim: int = 6,      # State features dimension
        n_concepts: int = 5,     # Number of concepts
        embedding_dim: int = 8,  # Dimension per concept embedding
        hidden_dim: int = 32,
        seed: int = 42
    ):
        self.n_concepts = n_concepts
        self.embedding_dim = embedding_dim
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Initialize weights randomly
        np.random.seed(seed)
        self.W1 = np.random.randn(input_dim, hidden_dim) * 0.1
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(hidden_dim, hidden_dim) * 0.1
        self.b2 = np.zeros(hidden_dim)
        
        # Concept heads: one per concept, outputs (prob, embedding)
        self.concept_weights = [
            np.random.randn(hidden_dim, embedding_dim + 1) * 0.1
            for _ in range(n_concepts)
        ]
        self.concept_biases = [
            np.zeros(embedding_dim + 1)
            for _ in range(n_concepts)
        ]
        
        self.concept_names = [
            'near_goal',
            'in_wind_zone',
            'near_wall', 
            'high_wind',
            'goal_reachable_soon'
        ]
        
        self._trained = False
    
    def _relu(self, x):
        return np.maximum(0, x)
    
    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Forward pass.
        
        Args:
            x: State features [batch_size, input_dim] or [input_dim]
            
        Returns:
            concept_probs: [batch_size, n_concepts] or [n_concepts]
            concept_embeddings: [batch_size, n_concepts, embedding_dim] or [n_concepts, embedding_dim]
        """
        single_input = False
        if len(x.shape) == 1:
            x = x.reshape(1, -1)
            single_input = True
        
        # Forward through shared encoder
        h = self._relu(x @ self.W1 + self.b1)
        h = self._relu(h @ self.W2 + self.b2)
        
        # Forward through concept heads
        concept_probs = []
        concept_embeddings = []
        
        for i in range(self.n_concepts):
            out = h @ self.concept_weights[i] + self.concept_biases[i]
            prob = self._sigmoid(out[:, 0])
            embedding = out[:, 1:]
            
            concept_probs.append(prob)
            concept_embeddings.append(embedding)
        
        concept_probs = np.stack(concept_probs, axis=1)
        concept_embeddings = np.stack(concept_embeddings, axis=1)
        
        if single_input:
            concept_probs = concept_probs[0]
            concept_embeddings = concept_embeddings[0]
        
        return concept_probs, concept_embeddings
    
    def get_concepts_with_leakage(self, x: np.ndarray) -> np.ndarray:
        """
        Get concept representation including leaked information.
        
        This is what causes problems under distribution shift!
        The embedding contains info beyond just the concept probability.
        """
        concept_probs, concept_embeddings = self.forward(x)
        
        if len(concept_probs.shape) == 1:
            # Single sample
            full_repr = np.concatenate([
                concept_probs.reshape(-1, 1),
                concept_embeddings
            ], axis=1)
            return full_repr.flatten()
        else:
            # Batch
            full_repr = np.concatenate([
                concept_probs[:, :, np.newaxis],
                concept_embeddings
            ], axis=2)
            return full_repr.reshape(full_repr.shape[0], -1)
    
    def get_concepts_no_leakage(self, x: np.ndarray) -> np.ndarray:
        """
        Get only concept probabilities (no leaked embeddings).
        
        This should be robust to distribution shift.
        """
        concept_probs, _ = self.forward(x)
        return concept_probs
    
    def train_on_data(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        epochs: int = 100,
        lr: float = 0.01
    ):
        """
        Train the encoder to predict concept targets.
        
        Simple gradient descent training.
        """
        n_samples = features.shape[0]
        
        for epoch in range(epochs):
            # Forward pass
            h1 = self._relu(features @ self.W1 + self.b1)
            h2 = self._relu(h1 @ self.W2 + self.b2)
            
            total_loss = 0.0
            
            for i in range(self.n_concepts):
                out = h2 @ self.concept_weights[i] + self.concept_biases[i]
                pred = self._sigmoid(out[:, 0])
                target = targets[:, i]
                
                # Binary cross entropy loss
                eps = 1e-7
                loss = -np.mean(target * np.log(pred + eps) + (1 - target) * np.log(1 - pred + eps))
                total_loss += loss
                
                # Gradient for concept head (simplified)
                error = pred - target
                grad_W = h2.T @ error.reshape(-1, 1) / n_samples
                self.concept_weights[i][:, 0] -= lr * grad_W.flatten()
                self.concept_biases[i][0] -= lr * np.mean(error)
            
            if (epoch + 1) % 20 == 0:
                print(f"    Epoch {epoch + 1}/{epochs}, Loss: {total_loss / self.n_concepts:.4f}")
        
        self._trained = True


class SoftConcepts:
    """
    Wrapper class for soft concepts that matches HardConcepts interface.
    """
    
    def __init__(
        self,
        env,
        use_leakage: bool = True,
        embedding_dim: int = 8
    ):
        self.env = env
        self.use_leakage = use_leakage
        self.embedding_dim = embedding_dim
        self.n_concepts = 5
        
        # Initialize encoder
        self.encoder = SoftConceptEncoder(
            input_dim=6,
            n_concepts=self.n_concepts,
            embedding_dim=embedding_dim
        )
        
        self.concept_names = self.encoder.concept_names
    
    def __call__(self, state: np.ndarray) -> np.ndarray:
        """Compute soft concepts for a state."""
        # Get state features
        row, col = int(state[0]), int(state[1])
        features = self._compute_features(row, col)
        
        if self.use_leakage:
            concepts = self.encoder.get_concepts_with_leakage(features)
        else:
            concepts = self.encoder.get_concepts_no_leakage(features)
        
        return concepts
    
    def _compute_features(self, row: int, col: int) -> np.ndarray:
        """Compute state features for the encoder."""
        norm_row = row / (self.env.height - 1) if self.env.height > 1 else 0.0
        norm_col = col / (self.env.width - 1) if self.env.width > 1 else 0.0
        
        goal_dist = (abs(row - self.env.goal[0]) + abs(col - self.env.goal[1]))
        max_dist = self.env.height + self.env.width - 2
        norm_goal_dist = goal_dist / max_dist if max_dist > 0 else 0.0
        
        wind_strength = self.env.wind[min(col, len(self.env.wind) - 1)] / 2.0
        in_wind_zone = 1.0 if self.env.wind[min(col, len(self.env.wind) - 1)] > 0 else 0.0
        
        dist_to_wall = min(row, self.env.height - 1 - row, col, self.env.width - 1 - col)
        norm_wall_dist = dist_to_wall / (min(self.env.height, self.env.width) / 2)
        
        return np.array([norm_row, norm_col, norm_goal_dist, 
                        wind_strength, in_wind_zone, norm_wall_dist], dtype=np.float32)
    
    def batch_forward(self, states: np.ndarray) -> np.ndarray:
        """Compute concepts for a batch of states."""
        return np.array([self(s) for s in states])
    
    def train_on_trajectories(
        self,
        trajectories: List[List[dict]],
        hard_concepts,
        epochs: int = 100
    ):
        """
        Train soft concept encoder to predict hard concepts.
        
        The encoder will learn to match hard concepts but will also
        encode additional information in the embeddings (leakage).
        """
        # Collect training data
        features_list = []
        concepts_list = []
        
        for traj in trajectories:
            for step in traj:
                row, col = int(step['state'][0]), int(step['state'][1])
                features_list.append(self._compute_features(row, col))
                concepts_list.append(hard_concepts(step['state']))
        
        features = np.array(features_list)
        targets = np.array(concepts_list)
        
        # Train
        self.encoder.train_on_data(features, targets, epochs=epochs)


def measure_leakage(
    soft_concepts,
    states: np.ndarray,
    features: np.ndarray
) -> float:
    """
    Measure information leakage by training a probe to predict raw state from concepts.
    
    High R² means concepts leak a lot of state information.
    This is the metric from Espinosa Zarlenga et al. (2025).
    """
    from sklearn.linear_model import Ridge
    from sklearn.metrics import r2_score
    
    # Get soft concept representations
    concept_repr = soft_concepts.batch_forward(states)
    
    # Train probe to predict raw state features from concepts
    probe = Ridge(alpha=1.0)
    probe.fit(concept_repr, features)
    
    # Measure R²
    pred_features = probe.predict(concept_repr)
    r2 = r2_score(features, pred_features)
    
    return r2


if __name__ == "__main__":
    import sys
    sys.path.insert(0, '/home/claude/temporal-leakage-ope/src')
    
    from gridworld import WindyGridworld, collect_trajectory
    from policies import EpsilonGreedyPolicy
    
    print("Testing Concept Modules")
    print("=" * 50)
    
    env = WindyGridworld()
    
    # Test hard concepts
    hard = HardConcepts(env)
    test_state = np.array([3, 0])  # Start
    print(f"\nHard concepts at start {test_state}:")
    concepts = hard(test_state)
    for name, val in zip(hard.concept_names, concepts):
        print(f"  {name}: {val}")
    
    # Test soft concepts
    print("\n" + "=" * 50)
    print("Soft Concepts")
    
    soft = SoftConcepts(env, use_leakage=True)
    concepts_with_leak = soft(test_state)
    print(f"\nSoft concepts (with leakage) shape: {concepts_with_leak.shape}")
    
    soft_no_leak = SoftConcepts(env, use_leakage=False)
    concepts_no_leak = soft_no_leak(test_state)
    print(f"Soft concepts (no leakage) shape: {concepts_no_leak.shape}")
    
    print("\n✓ Concepts module ready!")
