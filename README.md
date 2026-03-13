# Temporal Leakage Poisoning in Concept-Based Off-Policy Evaluation

**Course:** ELEC70122 Machine Learning for Safety-Critical Decision Making  
**Imperial College London**

---

## Research Question

**Does leakage poisoning compound over time in sequential Concept-Based OPE?**

## Hypothesis

Soft concept embeddings leak information beyond concept labels. When the evaluation policy diverges from the behavior policy, states become out-of-distribution (OOD). At OOD states, the leaked information becomes corrupted, causing concept predictions to degrade. Since importance sampling ratios multiply over time, this error compounds—OPE error grows with trajectory length for soft concepts, but not for hard concepts.

---

## Key Result

| Timestep | Hard Concepts | Soft Concepts | Degradation |
|----------|---------------|---------------|-------------|
| t=0 (in-distribution) | 100% | 100% | 0% |
| t=10 (boundary) | 100% | 100% | 0% |
| t=15 (OOD) | 100% | 94% | **6%** |
| t=20 (OOD) | 100% | 78.5% | **21.5%** |
| t=25 (OOD) | 100% | 72% | **28%** |
| t=30 (OOD) | 100% | 68% | **32%** |

**Hard concepts remain stable. Soft concepts degrade as timestep increases.**

---

## Method

### Setup
- **Environment:** Windy Gridworld (7×10)
- **Behavior policy:** ε-greedy (ε=0.4)
- **Evaluation policy:** Near-optimal (ε=0.05)
- **Train horizon:** Soft concepts trained on states from t < 10 only

### Concepts
- **Hard concepts:** Rule-based binary features (near_goal, high_wind, etc.) — no leakage
- **Soft concepts:** Neural network encoder outputting concept probabilities + hidden embeddings — leaks state information

### Measurement
1. Collect 500 trajectories using behavior policy
2. Train soft concept encoder on early timesteps (t < 10)
3. At each timestep t, measure:
   - Concept prediction accuracy (soft vs hard ground truth)
   - Information leakage (R² of linear probe: embeddings → raw features)
   - Distribution shift (KL divergence from t=0)

---

## Core Equations

### Standard PDIS
$$\hat{V}^{\text{PDIS}} = \frac{1}{N} \sum_{n=1}^{N} \sum_{t=0}^{T} \gamma^t \rho_{0:t} r_t$$

where $\rho_{0:t} = \prod_{t'=0}^{t} \frac{\pi_e(a_{t'}|s_{t'})}{\pi_b(a_{t'}|s_{t'})}$

### Concept-Based PDIS (CPDIS)
$$\hat{V}^{\text{CPDIS}} = \frac{1}{N} \sum_{n=1}^{N} \sum_{t=0}^{T} \gamma^t \rho^c_{0:t} r_t$$

where $\rho^c_{0:t} = \prod_{t'=0}^{t} \frac{\pi^c_e(a_{t'}|c_{t'})}{\pi^c_b(a_{t'}|c_{t'})}$

**Key difference:** Policies conditioned on concepts $c_t = \phi(s_t)$ instead of states.

### Leakage Measurement
Train linear probe: concept embeddings → raw state features  
High R² = high leakage (embeddings encode information beyond concept labels)

---

## Project Structure

```
temporal-leakage-ope/
├── src/
│   ├── gridworld.py      # Windy Gridworld environment
│   ├── policies.py       # Behavior/evaluation policies, ConceptPolicy
│   ├── concepts.py       # HardConcepts, SoftConcepts, measure_leakage
│   ├── ope.py            # PDIS, CPDIS, Monte Carlo ground truth
│   └── utils.py          # Trajectory statistics, distribution shift
├── experiments/
│   └── temporal_leakage_experiment.py   # Main experiment
└── results/
    └── temporal_leakage.png             # Output figure
```

---

## Running the Experiment

```bash
# Install dependencies
pip install numpy matplotlib scikit-learn

# Run experiment
cd temporal-leakage-ope
python experiments/temporal_leakage_experiment.py
```

---

## References

1. **Concept-Based OPE:** Majumdar, Teversham, Parbhoo. "Concept-Based Off-Policy Evaluation." RLC 2025. arXiv:2411.19395

2. **Leakage Poisoning:** Espinosa Zarlenga et al. "Avoiding Leakage Poisoning: Concept Interventions Under Distribution Shifts." ICML 2025. arXiv:2504.17921

3. **Concept Embedding Models:** Espinosa Zarlenga et al. "Concept Embedding Models." NeurIPS 2022. arXiv:2209.09056

---

## Conclusion

Soft concept embeddings leak information that becomes corrupted under distribution shift. In sequential settings, this causes OPE error to compound over time. Hard (rule-based) concepts do not suffer from this effect. For safety-critical OPE, practitioners should either use hard concepts or ensure soft concepts are robust to the distribution shift induced by policy divergence.