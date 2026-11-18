# PathSnap: Distributed Map-Matching & Route Reconstruction

**Hackathon Project** | PES University UE24CS352A Machine Learning  
**Team Members:** Prarthana Shivakumar • Prema • Priyansh • Prateek  

---

## 1. Project Overview

PathSnap is a distributed machine learning and sequential map-matching system that reconstructs realistic vehicle trajectories by matching noisy, irregularly sampled GPS observations to a high-density directed road network.

Urban GPS tracking frequently suffers from multipath reflections, signal attenuation, canyon effects, and low sampling intervals, causing raw trajectories to drift across parallel roads or jump across buildings. PathSnap solves this by coupling machine learning candidate classification with Hidden Markov Model (HMM) dynamic programming via the Viterbi algorithm.

---

## 2. Team Workstream Ownership

| Member | Primary Ownership | Core Module Responsibilities |
|---|---|---|
| **Prarthana** | Data Foundation | Reusable trip loader, EDA, coordinate projections, network integrity, Folium exploration maps |
| **Prema** | Candidate Classification | Road-segment candidate generation, feature extraction, training 5 distinct ML models, metrics comparison |
| **Prateek** | **Sequential Modelling & Optimisation** | **Directed arc connectivity, HMM transition/emission modeling, log-space Viterbi decoding, invalid transition handling, GPS gap attenuation, Bayesian hyperparameter optimization (Optuna)** |
| **Priyansh** | Application & Presentation | Evaluator-facing UI shell, trip selection flow, interactive map visualization, project packaging & integration |

---

## 3. Sequential Modelling & Optimization Architecture (`route_reconstruction/` & `optimization/`)

### A. Hidden States
- For each GPS observation $t \in \{0, \dots, T-1\}$, candidate road segments $C_t = \{a_{t, 1}, \dots, a_{t, K_t}\}$ act as the discrete hidden states.
- Reduces state space from $|\mathcal{A}| = 165,790$ full road segments to local high-probability candidates while strictly preserving segment identities.

### B. Emission Probabilities
- Normalised log emission scores:
  $$\ln e_t(a) = \ln \left( \frac{p(a \mid O_t)}{\sum_k p(a_k \mid O_t) + \epsilon} \right)$$
- Clipped at $\epsilon = 10^{-8}$ to prevent $-\infty$ underflow.

### C. Topological Transition Modeling
- **Same Segment ($a_i = a_j$):** Log baseline $\ln p_{stay}$. Attenuated if $\Delta t > \Delta t_{threshold}$.
- **Direct Successor ($to(a_i) = from(a_j)$):** Driving across intersection. Log baseline $\ln p_{direct}$.
- **Multi-hop Reachability:** BFS search over directed graph up to $k_{max}$ hops:
  $$P(a_j \mid a_i) = p_{direct} \cdot (\text{hop\_decay})^{k - 1}$$
- **Invalid / Disconnected Transitions:** Controlled penalty policy $\epsilon_{invalid} = 10^{-5}$ prevents total Viterbi trellis collapse during extreme sensor noise.
- **GPS Signal Dropout / Gaps:** When $\Delta t > \Delta t_{threshold}$, transition penalties are dynamically relaxed, allowing the vehicle to re-anchor using emission probabilities.

### D. Viterbi Dynamic Programming
- Fully vectorized log-space dynamic programming:
  $$V_t(j) = \max_i [V_{t-1}(i) + \ln P(a_j \mid a_i)] + \ln e_t(a_j)$$
- Exact sequence backtracking from $t = T-1$ down to $0$.

### E. Bayesian Hyperparameter Optimization
- Framework powered by **Optuna TPESampler** (Tree-structured Parzen Estimator).
- Group-aware cross-validation across training trips to guarantee zero data leakage.
- Ready-to-connect pipelines for both classifier models and sequential decoder parameters.

---

## 4. First-Phase Empirical Results (Real Trip `00000001`)

Evaluation on Trip `00000001` (1,315 GPS observations, 88,809 nodes, 165,790 road segments):

| Metric | Standalone Classifier (Pointwise Argmax) | Sequential HMM / Viterbi Decoder | Gain / Impact |
|---|---|---|---|
| **Relaxed Route Continuity** | **72.91%** | **92.39%** | **+19.48% improvement** |
| **Strict 1-Hop Continuity** | 68.26% | 76.56% | +8.30% improvement |
| **Invalid Disconnected Jumps** | **356** | **100** | **256 invalid jumps eliminated** |
| **Reference Route Precision** | 55.56% | 67.90% | +12.34% gain |
| **Reference Route Recall** | 78.10% | 87.05% | +8.95% gain |
| **Reference Route Macro F1** | **64.92%** | **76.29%** | **+11.37% gain** |
| **Execution Latency** | 2.99 ms | 197.98 ms | Real-time production throughput |

---

## 5. Repository Structure

```
PathSnap/
├── route_reconstruction/         # Prateek
│   ├── __init__.py
│   ├── hmm.py                   # RoadGraph, TransitionModel, EmissionModel, MapMatchingHMM
│   ├── viterbi.py               # Log-space Viterbi dynamic programming recurrence
│   └── route_decoder.py         # Standard decode_route interface & continuity metrics
├── optimization/                 # Prateek
│   ├── __init__.py
│   ├── bayesian_optimization.py # Optuna TPE optimizer & GroupKFold evaluator
│   ├── hyperparameter_search.py # Decoder & Classifier search spaces
│   └── optimization_results.py  # Structured trial logging & reporting
├── tests/                        # Prateek
│   ├── test_hmm.py              # Transition, emission, and graph tests
│   ├── test_viterbi.py          # Recurrence, backtracking, and gap tests
│   ├── test_route_decoder.py    # Interface compliance and metric tests
│   └── test_optimization.py     # Bayesian optimization & Group-CV tests
├── data_processing/              # Prarthana
├── models/                       # Prema
├── backend/                      # Joint API Ownership
├── frontend/                     # Priyansh
├── results/                      # Phase 1 evidence JSON & report
├── artifacts/                    # Shared summary artifacts
├── run_phase1_demonstration.py   # Full verification script on real trip
├── requirements.txt
└── README.md
```

---

## 6. How to Run & Verify

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Run Automated Unit Tests (16/16 Passing)
```bash
python -m pytest tests/ -v
```

### Run First-Phase Real Trip Demonstration
```bash
python run_phase1_demonstration.py
```
Outputs and evidence files are automatically written to `results/phase1_decoder_evidence.json` and `results/first_phase_decoder_report.md`.
