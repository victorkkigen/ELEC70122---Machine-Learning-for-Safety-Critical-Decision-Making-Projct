# Temporal Leakage Poisoning in Sequential Concept-Based OPE
**Institution:** Imperial College London[cite: 1]  
**Project:** Safety-Critical Machine Learning (ELEC70122)[cite: 1]  
**Lead Researcher:** Victor Kigen[cite: 1]  
**Academic Supervisor:** Pietro Ferraro[cite: 1]

---

## ## Executive Summary
This research identifies a critical vulnerability in **Concept-Based Off-Policy Evaluation (OPE)**. While soft concepts offer interpretability, they leak latent state information that becomes corrupted under distribution shifts[cite: 1]. In sequential settings, this corruption compounds, leading to a **24x increase** in evaluation error[cite: 1].

---

## ## 1. The Core Discovery: Temporal Leakage
The study utilizes a **$7 \times 10$ Windy Gridworld** to track how error propagates as the evaluation policy deviates from the behavior policy[cite: 1].

### ### Comparison of Concept Types
*   **Hard Concepts:** Rule-based and binary; these show **0% degradation** because they do not leak latent state information[cite: 1].
*   **Soft Concepts:** Learned neural embeddings; these suffer from **32% accuracy loss** when states move Out-of-Distribution (OOD)[cite: 1].

### ### Multiplicative Failure
Because Importance Sampling (IS) ratios are cumulative, a small error in concept prediction at $t=15$ poisons the entire remaining trajectory[cite: 1].
> **Key Finding:** Error is binary. Once a state enters an "unknown" bin, the soft concept model fails completely, causing the OPE estimate to diverge exponentially[cite: 1].

---

## ## 2. Technical Results: Compounding Degradation
The following table illustrates how soft concept accuracy drops and OPE error grows as the agent moves further into OOD territory:

| Timestep ($t$) | State Distribution | Soft Concept Accuracy | OPE Error Growth |
| :--- | :--- | :--- | :--- |
| $0 \le t < 10$ | In-Distribution | 100%[cite: 1] | Baseline[cite: 1] |
| $t = 15$ | Boundary / OOD | 94%[cite: 1] | $6 \times$[cite: 1] |
| $t = 20$ | OOD | 78.5%[cite: 1] | $14 \times$[cite: 1] |
| **$t = 30$** | **Deep OOD** | **68%**[cite: 1] | **$24 \times$**[cite: 1] |

---

## ## 3. The Solution: Conformal Gating
To mitigate these failures, a **Conformal Prediction-based gating** mechanism was implemented to detect and filter unreliable concept predictions[cite: 1].

### ### Why it is Useful in the Real World
1.  **Safety Fuse:** It acts as a "circuit breaker," identifying when the environment has shifted and the model's internal sensors are no longer reliable[cite: 1].
2.  **Overconfidence Correction:** Traditional neural networks often stay "confident" even when wrong; Conformal Gating uses statistical calibration to detect these silent failures[cite: 1].
3.  **Performance Gains:** Implementation resulted in a **71% reduction in Mean Squared Error (MSE)** at long horizons[cite: 1].

---

## ## 4. Real-World Applications
*   **Industrial Digitization:** Ensuring AI operating systems in complex value chains (e.g., tea or coffee production) don't make decisions based on "poisoned" historical data[cite: 1].
*   **Healthcare:** Preventing diagnostic models from providing confident but incorrect interpretations when patient data differs from the training set[cite: 1].
*   **Autonomous Systems:** Allowing robots to recognize when they have entered a state space that requires human intervention or a more conservative policy[cite: 1].

---

## ## 5. Methodology & Formulas
The study compares standard **Per-Decision Importance Sampling (PDIS)** against **Concept-Based PDIS (CPDIS)** to measure the impact of poisoned concept ratios:

$$V^{CPDIS} = \frac{1}{N} \sum_{n=1}^{N} \sum_{t=0}^{T} \gamma^t \left( \prod_{t'=0}^{t} \frac{\pi_e(a_{t'}|c_{t'})}{\pi_b(a_{t'}|c_{t'})} \right) r_t$$[cite: 1]

---
*Generated for the Imperial College London Master's Thesis Project (2026).*[cite: 1]
