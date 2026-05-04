# Temporal Leakage Poisoning in Sequential Concept-Based OPE
**Institution:** Imperial College London
**Project:** Safety-Critical Machine Learning (ELEC70122)
**Authors:** Victor Kigen & Naimeh Fakhr Vaezi
**Academic Supervisor:** Pietro Ferraro

---

## ## Executive Summary
This research identifies a critical vulnerability in **Concept-Based Off-Policy Evaluation (OPE)**. While soft concepts offer interpretability, they leak latent state information that becomes corrupted under distribution shifts. In sequential settings, this corruption compounds, leading to a **24x increase** in evaluation error.

---

## ## 1. The Core Discovery: Temporal Leakage
The study utilizes a **$7 \times 10$ Windy Gridworld** to track how error propagates as the evaluation policy deviates from the behavior policy.

### ### Comparison of Concept Types
*   **Hard Concepts:** Rule-based and binary; these show **0% degradation** because they do not leak latent state information.
*   **Soft Concepts:** Learned neural embeddings; these suffer from **32% accuracy loss** when states move Out-of-Distribution (OOD).

### ### Multiplicative Failure
Because Importance Sampling (IS) ratios are cumulative, a small error in concept prediction at $t=15$ poisons the entire remaining trajectory.
> **Key Finding:** Error is binary. Once a state enters an "unknown" bin, the soft concept model fails completely, causing the OPE estimate to diverge exponentially.

---

## ## 2. Technical Results: Compounding Degradation
The following table illustrates how soft concept accuracy drops and OPE error grows as the agent moves further into OOD territory:

| Timestep ($t$) | State Distribution | Soft Concept Accuracy | OPE Error Growth |
| :--- | :--- | :--- | :--- |
| $0 \le t < 10$ | In-Distribution | 100% | Baseline |
| $t = 15$ | Boundary / OOD | 94% | $6 \times$ |
| $t = 20$ | OOD | 78.5% | $14 \times$ |
| **$t = 30$** | **Deep OOD** | **68%** | **$24 \times$** |

---

## ## 3. The Solution: Conformal Gating
To mitigate these failures, a **Conformal Prediction-based gating** mechanism was implemented to detect and filter unreliable concept predictions.

### ### Why it is Useful in the Real World
1.  **Safety Fuse:** It acts as a "circuit breaker," identifying when the environment has shifted and the model's internal sensors are no longer reliable.
2.  **Overconfidence Correction:** Traditional neural networks often stay "confident" even when wrong; Conformal Gating uses statistical calibration to detect these silent failures.
3.  **Performance Gains:** Implementation resulted in a **71% reduction in Mean Squared Error (MSE)** at long horizons.

---

## ## 4. Real-World Applications
*   **Industrial Digitization:** Ensuring AI operating systems in complex value chains (e.g., tea or coffee production) don't make decisions based on "poisoned" historical data.
*   **Healthcare:** Preventing diagnostic models from providing confident but incorrect interpretations when patient data differs from the training set.
*   **Autonomous Systems:** Allowing robots to recognize when they have entered a state space that requires human intervention or a more conservative policy.

---

## ## 5. Methodology & Formulas
The study compares standard **Per-Decision Importance Sampling (PDIS)** against **Concept-Based PDIS (CPDIS)** to measure the impact of poisoned concept ratios:

$$V^{CPDIS} = \frac{1}{N} \sum_{n=1}^{N} \sum_{t=0}^{T} \gamma^t \left( \prod_{t'=0}^{t} \frac{\pi_e(a_{t'}|c_{t'})}{\pi_b(a_{t'}|c_{t'})} \right) r_t$$

---

