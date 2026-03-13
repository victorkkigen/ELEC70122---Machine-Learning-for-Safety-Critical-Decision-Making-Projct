# Temporal Leakage Poisoning in Concept-Based Off-Policy Evaluation

## Research Question
Does leakage poisoning compound over time in sequential Concept-Based OPE?

## Hypothesis
- Soft concepts leak extra information that becomes corrupted as trajectories diverge from training distribution
- OPE error with soft concepts grows with trajectory length
- Hard concepts remain stable because they don't leak information

## Project Structure
```
temporal-leakage-ope/
├── src/
│   ├── gridworld.py      # Windy Gridworld environment
│   ├── policies.py       # Behavior and evaluation policies
│   ├── concepts.py       # Soft and hard concept layers
│   ├── ope.py            # PDIS estimator
│   └── utils.py          # Helper functions
├── experiments/
│   ├── collect_data.py   # Generate trajectory dataset
│   ├── ground_truth.py   # Compute true policy value
│   └── run_experiment.py # Main experiment loop
├── results/              # Plots and tables
└── paper/                # LaTeX files
```

## Setup
```bash
pip install numpy torch matplotlib --break-system-packages
```

## Run Experiment
```bash
python experiments/run_experiment.py
```

## Team
- Person 1: Environment, policies, data collection
- Person 2: Concepts, OPE estimator, experiments

## References
- Espinosa Zarlenga et al. (2025). Avoiding Leakage Poisoning. ICML 2025.
- Majumdar et al. (2025). Concept-Based Off-Policy Evaluation. RLC 2025.
