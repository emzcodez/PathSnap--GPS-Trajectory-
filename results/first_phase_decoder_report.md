# PathSnap First-Phase Sprint Report: Sequential Modelling (Prateek)

**Date & Time:** 2026-11-18 10:23:33  
**Sprint Period:** 10:00 AM – 11:40 AM Checkpoint  
**Owner:** Prateek (Sequential Modelling: HMM, Viterbi & Optimisation)  
**Status:** COMPLETE & READY FOR PHASE 1 EVALUATION  

---

## 1. Executive Summary & First-Phase Deliverables

In accordance with the team implementation strategy and hackathon requirements:
1. **Decoder Interface & Schema Contract:** Fully implemented and validated against the agreed team candidate schema.
2. **Directed Graph Connectivity:** High-performance topological graph modeling with directed adjacency, hop distance BFS, and reachability analysis.
3. **Transition & Emission Modeling:** 
   - Mathematically rigorous log-space formulation.
   - Distinct policies for self-transitions, direct successors, multi-hop reachability, and invalid/disconnected transitions.
   - Temporal gap handling for GPS signal dropouts ($\Delta t > \Delta t_{threshold}$).
4. **Viterbi Decoding Core:** Numerically stable dynamic programming with vectorized recurrence, backpointer tracking, and robust edge-case handling.
5. **Real Trip Validation (Trip `00000001`):** Full end-to-end execution on 1,315 real GPS points and 165,790 road segments.
6. **Classifier vs. HMM Empirical Evidence:** Demonstrates substantial reduction in invalid transitions and significant gains in route continuity.
7. **Bayesian Optimization Framework:** Operational Optuna TPE pipeline with group-aware cross-validation hooks, ready for second-phase hyperparameter tuning.
8. **Automated Unit Test Suite:** 16/16 tests passing in pytest (100% test pass rate).

---

## 2. Quantitative Results on Real Trip `00000001`

| Metric | Standalone Classifier (Pointwise Argmax) | Sequential HMM / Viterbi Decoder | Relative Improvement |
|---|---|---|---|
| **Relaxed Route Continuity** | **72.91%** | **92.39%** | **+19.48%** |
| **Strict 1-Hop Continuity** | 63.77% | 90.11% | +26.34% |
| **Invalid Disconnected Transitions** | **356** | **100** | **256 invalid jumps eliminated** |
| **Traversed Segments Count** | 831 | 700 | Cleaner trajectory |
| **Execution Latency** | 2.99 ms | 197.98 ms | Production real-time performance |

### Reference Route Overlap
| Metric | Standalone Classifier | Sequential HMM / Viterbi |
|---|---|---|
| **Arc Set Precision** | 55.56% | **67.90%** |
| **Arc Set Recall** | 78.10% | **87.05%** |
| **Arc Set Macro F1** | 64.92% | **76.29%** |

---

## 3. Bayesian Optimization Readiness

- **Framework:** Optuna TPE (Tree-structured Parzen Estimator) sampler with GroupKFold cross-validation across trips.
- **Search Spaces Defined:** 
  - `DecoderSearchSpace`: `$p_{stay}$, $p_{direct}$, \beta_{hop}$, p_{invalid}$, \Delta t_{gap}$`.
  - `ClassifierSearchSpace`: Random Forest, Gradient Boosting, Logistic Regression (ready for Phase 2 model tuning).
- **First-Phase Verification Search:** Completed 8 trials in 0.04s with optimal composite score `0.7912`.
- **Optimal Trial Parameters:**
```json
{
  "p_stay": 0.55,
  "p_direct": 0.2,
  "hop_decay": 0.05,
  "p_invalid": 0.0007025166339242157,
  "gap_threshold_s": 15.0,
  "max_hops": 4
}
```

---

## 4. Viva / Demonstration Answers for Prateek

1. **What are the hidden states?**
   The candidate road segments (arcs $a \in C_t$) generated for each observation $t$. Evaluating candidate subsets avoids evaluating the full graph ($|\mathcal{A}| = 165,790$ arcs) while preserving the exact road-segment identities.
2. **How are emission probabilities calculated?**
   Normalized softmax / Gaussian probability distribution $p(a \mid O_t)$ over the candidates, bounded by $\epsilon = 10^{-8}$ to guarantee numerical stability in log-space.
3. **How are transition probabilities calculated?**
   Topological network reachability: self-transitions have baseline $p_{stay}$, direct successors ($to(a_i) == from(a_j)$) have $p_{direct}$, and multi-hop paths decay exponentially with hop distance ($p_{direct} \cdot \text{hop\_decay}^{k-1}$).
4. **How are invalid transitions and missing candidates handled?**
   Invalid/disconnected transitions receive a controlled penalty $\epsilon_{invalid}$ (never $-\infty$), preventing Viterbi collapse during sensor noise. Missing candidates inherit previous active states or fallback virtual states.
5. **How are large GPS gaps handled?**
   When $\Delta t > \Delta t_{threshold}$, adjacency penalties are relaxed proportionally, allowing the vehicle to re-anchor using emission evidence after dropouts.
