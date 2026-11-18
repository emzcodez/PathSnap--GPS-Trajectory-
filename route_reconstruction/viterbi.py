"""Viterbi Algorithm for Map-Matching Route Reconstruction

Author: Prateek (Sequential Modelling: HMM, Viterbi & Optimisation)
Hackathon: PathSnap Distributed Map-Matching

Provides numerically stable log-space Viterbi decoding over road-network candidate trellises.
Guarantees:
- Fully vectorized log-space dynamic programming recurrence.
- Strict numerical stability without underflow.
- Robust handling of variable candidate sizes, empty candidate sets, and graph disconnects.
- Detailed diagnostic tracking (transition types, path scores, execution timings).
"""

from typing import List, Dict, Tuple, Optional, Any
import numpy as np
import time
from .hmm import RoadGraph, TransitionModel, EmissionModel


def viterbi_decode(
    candidate_outputs: List[Dict[str, Any]],
    graph: RoadGraph,
    transition_model: Optional[TransitionModel] = None,
    emission_model: Optional[EmissionModel] = None,
    timestamps: Optional[List[float]] = None
) -> Tuple[List[int], float, Dict[str, Any]]:
    """
    Perform exact log-space Viterbi decoding on candidate observations.

    Parameters
    ----------
    candidate_outputs : List[Dict[str, Any]]
        List of dicts per observation, following the team candidate schema:
        {
            "observation_index": int,
            "candidate_arc_ids": List[int],
            "probabilities": List[float]
        }
    graph : RoadGraph
        Road network graph containing directed arc connections.
    transition_model : TransitionModel, optional
        Transition probability calculator.
    emission_model : EmissionModel, optional
        Emission probability calculator.
    timestamps : List[float], optional
        Observation timestamps in seconds for GPS gap detection.

    Returns
    -------
    best_path : List[int]
        Optimal sequence of predicted arc_ids, one per observation.
    best_score : float
        Total log-likelihood score of the decoded path.
    diagnostics : Dict[str, Any]
        Performance and structural metrics (timing, trellis size, transition types).
    """
    start_time = time.perf_counter()
    trans_model = transition_model or TransitionModel()
    emiss_model = emission_model or EmissionModel()

    T = len(candidate_outputs)
    if T == 0:
        return [], 0.0, {"error": "Empty observation sequence", "execution_time_ms": 0.0}

    # Handle single observation edge case
    if T == 1:
        cands = candidate_outputs[0].get("candidate_arc_ids", [])
        probs = candidate_outputs[0].get("probabilities", [])
        if not cands:
            return [-1], 0.0, {"execution_time_ms": 0.0, "total_observations": 1}
        best_idx = int(np.argmax(probs)) if probs else 0
        exec_ms = (time.perf_counter() - start_time) * 1000.0
        return [cands[best_idx]], 0.0, {"execution_time_ms": exec_ms, "total_observations": 1}

    # Step 0: Pre-process and validate candidates per timestep
    # If any step has empty candidates, fill with a dummy fallback or neighbors
    processed_candidates: List[List[int]] = []
    processed_emissions: List[np.ndarray] = []

    for t, item in enumerate(candidate_outputs):
        cands = item.get("candidate_arc_ids", [])
        probs = item.get("probabilities", [])

        if not cands:
            # Policy for missing candidates:
            # If step t has 0 candidates, inherit candidates from previous step or dummy arc -1
            if processed_candidates and processed_candidates[-1]:
                cands = list(processed_candidates[-1])
                probs = [1.0 / len(cands)] * len(cands)
            else:
                cands = [-1]
                probs = [1.0]

        log_emiss = emiss_model.normalize_log_emissions(probs)
        processed_candidates.append(cands)
        processed_emissions.append(log_emiss)

    # Step 1: Initialize Viterbi Trellis at t = 0
    # Prior distribution: uniform over initial candidates (or proportional to initial emission)
    K_0 = len(processed_candidates[0])
    viterbi_trellis: List[np.ndarray] = [processed_emissions[0].copy()]
    backpointers: List[np.ndarray] = [np.zeros(K_0, dtype=np.int32)]

    # Diagnostic tracking for transition behavior
    transition_counts = {
        "self_transitions": 0,
        "direct_successors": 0,
        "multi_hop_transitions": 0,
        "invalid_transitions": 0,
        "gap_transitions": 0,
    }

    # Step 2: Forward Dynamic Programming Recurrence
    for t in range(1, T):
        prev_cands = processed_candidates[t - 1]
        curr_cands = processed_candidates[t]
        K_prev = len(prev_cands)
        K_curr = len(curr_cands)

        # Delta t calculation for GPS gap handling
        delta_t = None
        if timestamps is not None and len(timestamps) > t:
            try:
                delta_t = float(timestamps[t]) - float(timestamps[t - 1])
                if delta_t > trans_model.gap_threshold_s:
                    transition_counts["gap_transitions"] += 1
            except (ValueError, TypeError):
                delta_t = None

        # Compute |K_prev| x |K_curr| log-transition matrix
        log_trans_mat = trans_model.compute_transition_matrix(
            prev_cands, curr_cands, graph, delta_t=delta_t
        )

        # Recurrence: V_t(j) = max_i [ V_{t-1}(i) + log_trans(i, j) ] + log_emiss(j)
        # Vectorized addition: V_{t-1} column-vector + log_trans_mat -> shape (K_prev, K_curr)
        prev_v = viterbi_trellis[t - 1][:, np.newaxis]
        candidate_scores = prev_v + log_trans_mat

        best_prev_indices = np.argmax(candidate_scores, axis=0)
        max_prev_scores = candidate_scores[best_prev_indices, np.arange(K_curr)]

        curr_v = max_prev_scores + processed_emissions[t]

        viterbi_trellis.append(curr_v)
        backpointers.append(best_prev_indices)

    # Step 3: Termination
    best_final_idx = int(np.argmax(viterbi_trellis[-1]))
    best_score = float(viterbi_trellis[-1][best_final_idx])

    # Step 4: Backtracking
    best_path_indices = [0] * T
    best_path_indices[-1] = best_final_idx

    for t in range(T - 1, 0, -1):
        best_path_indices[t - 1] = int(backpointers[t][best_path_indices[t]])

    best_path: List[int] = [
        processed_candidates[t][best_path_indices[t]] for t in range(T)
    ]

    # Step 5: Tally transition metrics along the decoded path
    for t in range(T - 1):
        a_from = best_path[t]
        a_to = best_path[t + 1]
        if a_from == a_to:
            transition_counts["self_transitions"] += 1
        else:
            hops = graph.get_hop_distance(a_from, a_to, max_hops=trans_model.max_hops)
            if hops == 1:
                transition_counts["direct_successors"] += 1
            elif hops > 1:
                transition_counts["multi_hop_transitions"] += 1
            else:
                transition_counts["invalid_transitions"] += 1

    exec_ms = (time.perf_counter() - start_time) * 1000.0

    diagnostics = {
        "execution_time_ms": round(exec_ms, 2),
        "total_observations": T,
        "best_path_log_likelihood": round(best_score, 4),
        "transition_counts": transition_counts,
        "average_candidates_per_step": round(float(np.mean([len(c) for c in processed_candidates])), 2),
    }

    return best_path, best_score, diagnostics

