"""Standard Route Decoder Interface and Evaluation Metrics for PathSnap

Author: Prateek (Sequential Modelling: HMM, Viterbi & Optimisation)
Hackathon: PathSnap Distributed Map-Matching

Implements the agreed team decoder contract:
    predicted_arc_sequence = decode_route(
        candidates=candidate_output_by_observation,
        arcs=trip["arcs"],
        timestamps=trip["gps"]["timestamp"]
    )

Also provides:
    - compute_route_continuity: Evaluates topological consistency.
    - extract_traversed_route: Deduplicates consecutive segments to produce road trajectory.
    - compare_classifier_vs_hmm: Directly measures HMM improvement over standalone classifier.
"""

from typing import List, Dict, Tuple, Optional, Any, Union
import numpy as np
import pandas as pd
import time
from .hmm import RoadGraph, TransitionModel, EmissionModel
from .viterbi import viterbi_decode


def compute_route_continuity(
    arc_sequence: List[int],
    graph: RoadGraph,
    allow_multihop: bool = True,
    max_hops: int = 4
) -> Dict[str, Any]:
    """
    Calculate route continuity metrics along an arc sequence.

    Parameters
    ----------
    arc_sequence : List[int]
        Sequence of arc IDs (either pointwise observations or traversed route).
    graph : RoadGraph
        Road network graph.
    allow_multihop : bool
        If True, transitions reachable within max_hops are counted as continuous.
    max_hops : int
        Maximum hops to consider continuous.

    Returns
    -------
    Dict[str, Any]
        - strict_continuity: float in [0, 1] (self or direct successor)
        - relaxed_continuity: float in [0, 1] (self, direct, or multi-hop reachable)
        - total_transitions: int
        - valid_transitions: int
        - invalid_transitions: int
    """
    T = len(arc_sequence)
    if T <= 1:
        return {
            "strict_continuity": 1.0,
            "relaxed_continuity": 1.0,
            "total_transitions": 0,
            "valid_transitions": 0,
            "invalid_transitions": 0,
            "self_transitions": 0,
            "direct_transitions": 0,
            "multihop_transitions": 0,
        }

    self_trans = 0
    direct_trans = 0
    multihop_trans = 0
    invalid_trans = 0

    for t in range(T - 1):
        a1 = arc_sequence[t]
        a2 = arc_sequence[t + 1]

        if a1 == a2:
            self_trans += 1
        else:
            hops = graph.get_hop_distance(a1, a2, max_hops=max_hops)
            if hops == 1:
                direct_trans += 1
            elif hops > 1:
                multihop_trans += 1
            else:
                invalid_trans += 1

    total = T - 1
    strict_valid = self_trans + direct_trans
    relaxed_valid = strict_valid + multihop_trans

    return {
        "strict_continuity": round(strict_valid / total, 4),
        "relaxed_continuity": round(relaxed_valid / total, 4),
        "total_transitions": total,
        "valid_transitions": relaxed_valid if allow_multihop else strict_valid,
        "invalid_transitions": invalid_trans,
        "self_transitions": self_trans,
        "direct_transitions": direct_trans,
        "multihop_transitions": multihop_trans,
    }


def extract_traversed_route(pointwise_arcs: List[int]) -> List[int]:
    """
    Deduplicate consecutive identical arc IDs to obtain the sequence of road segments traversed.
    E.g. [2641, 2641, 2641, 64973, 64973] -> [2641, 64973]
    """
    if not pointwise_arcs:
        return []
    traversed = [pointwise_arcs[0]]
    for arc in pointwise_arcs[1:]:
        if arc != traversed[-1]:
            traversed.append(arc)
    return traversed


def decode_route(
    candidates: List[Dict[str, Any]],
    arcs: Union[pd.DataFrame, RoadGraph, List[Any]],
    timestamps: Optional[Union[pd.Series, List[float], np.ndarray]] = None,
    graph: Optional[RoadGraph] = None,
    transition_model: Optional[TransitionModel] = None,
    emission_model: Optional[EmissionModel] = None,
    return_full_result: bool = False
) -> Union[List[int], Dict[str, Any]]:
    """
    Standard team decoder interface.

    Parameters
    ----------
    candidates : List[Dict[str, Any]]
        List of candidate dictionaries:
        [{"observation_index": int, "candidate_arc_ids": [...], "probabilities": [...]}, ...]
    arcs : pd.DataFrame or RoadGraph or List
        The arcs table containing from_node and to_node connections.
    timestamps : pd.Series or List[float], optional
        GPS observation timestamps for time gap handling.
    graph : RoadGraph, optional
        Pre-built RoadGraph if already initialized (for high performance).
    transition_model : TransitionModel, optional
        Custom transition probability configuration.
    emission_model : EmissionModel, optional
        Custom emission normalization configuration.
    return_full_result : bool, optional
        If True, returns a comprehensive dictionary with traversed route, metrics, and diagnostics.
        If False (default), returns the list of predicted arc IDs per observation.

    Returns
    -------
    List[int] or Dict[str, Any]
        Predicted arc sequence or full detailed result dictionary.
    """
    # 1. Ensure RoadGraph is ready
    if graph is None:
        if isinstance(arcs, RoadGraph):
            rg = arcs
        else:
            rg = RoadGraph(arcs)
    else:
        rg = graph

    # 2. Extract timestamps list if provided
    ts_list = None
    if timestamps is not None:
        if isinstance(timestamps, (pd.Series, np.ndarray)):
            ts_list = timestamps.tolist()
        elif isinstance(timestamps, (list, tuple)):
            ts_list = list(timestamps)

    # 3. Execute Viterbi decoding
    best_path, best_score, diagnostics = viterbi_decode(
        candidate_outputs=candidates,
        graph=rg,
        transition_model=transition_model,
        emission_model=emission_model,
        timestamps=ts_list
    )

    if not return_full_result:
        return best_path

    # Full structured output
    continuity_info = compute_route_continuity(best_path, rg)
    traversed_route = extract_traversed_route(best_path)
    traversed_continuity = compute_route_continuity(traversed_route, rg)

    return {
        "pointwise_arcs": best_path,
        "traversed_arcs": traversed_route,
        "best_score": best_score,
        "pointwise_continuity": continuity_info,
        "traversed_continuity": traversed_continuity,
        "diagnostics": diagnostics,
    }


def compare_classifier_vs_hmm(
    candidates: List[Dict[str, Any]],
    arcs: Union[pd.DataFrame, RoadGraph, List[Any]],
    timestamps: Optional[Any] = None,
    reference_route: Optional[List[int]] = None,
    graph: Optional[RoadGraph] = None
) -> Dict[str, Any]:
    """
    Directly compares standalone classifier predictions (pointwise argmax)
    with sequential HMM/Viterbi route decoding.
    
    Demonstrates the exact value added by sequential modeling and graph topology.
    """
    if graph is None:
        rg = arcs if isinstance(arcs, RoadGraph) else RoadGraph(arcs)
    else:
        rg = graph

    # 1. Standalone Classifier Prediction (Pointwise Argmax)
    clf_start = time.perf_counter()
    clf_pointwise: List[int] = []
    for item in candidates:
        cands = item.get("candidate_arc_ids", [])
        probs = item.get("probabilities", [])
        if cands and probs:
            best_idx = int(np.argmax(probs))
            clf_pointwise.append(cands[best_idx])
        elif cands:
            clf_pointwise.append(cands[0])
        else:
            clf_pointwise.append(-1)
    clf_time_ms = (time.perf_counter() - clf_start) * 1000.0

    clf_traversed = extract_traversed_route(clf_pointwise)
    clf_continuity = compute_route_continuity(clf_pointwise, rg)
    clf_traversed_cont = compute_route_continuity(clf_traversed, rg)

    # 2. Sequential HMM/Viterbi Decoding
    hmm_start = time.perf_counter()
    full_result = decode_route(
        candidates=candidates,
        arcs=rg,
        timestamps=timestamps,
        graph=rg,
        return_full_result=True
    )
    hmm_time_ms = (time.perf_counter() - hmm_start) * 1000.0

    hmm_pointwise = full_result["pointwise_arcs"]
    hmm_traversed = full_result["traversed_arcs"]
    hmm_continuity = full_result["pointwise_continuity"]
    hmm_traversed_cont = full_result["traversed_continuity"]

    comparison: Dict[str, Any] = {
        "classifier_only": {
            "execution_time_ms": round(clf_time_ms, 2),
            "pointwise_strict_continuity": clf_continuity["strict_continuity"],
            "pointwise_relaxed_continuity": clf_continuity["relaxed_continuity"],
            "pointwise_invalid_transitions": clf_continuity["invalid_transitions"],
            "traversed_segments_count": len(clf_traversed),
            "traversed_relaxed_continuity": clf_traversed_cont["relaxed_continuity"],
        },
        "classifier_plus_hmm": {
            "execution_time_ms": round(hmm_time_ms, 2),
            "pointwise_strict_continuity": hmm_continuity["strict_continuity"],
            "pointwise_relaxed_continuity": hmm_continuity["relaxed_continuity"],
            "pointwise_invalid_transitions": hmm_continuity["invalid_transitions"],
            "traversed_segments_count": len(hmm_traversed),
            "traversed_relaxed_continuity": hmm_traversed_cont["relaxed_continuity"],
            "best_path_log_likelihood": full_result["best_score"],
        },
        "improvements": {
            "continuity_gain": round(
                hmm_continuity["relaxed_continuity"] - clf_continuity["relaxed_continuity"], 4
            ),
            "invalid_transitions_eliminated": (
                clf_continuity["invalid_transitions"] - hmm_continuity["invalid_transitions"]
            ),
        }
    }

    # Reference route comparison if supplied (for training/development analysis only)
    if reference_route is not None:
        ref_set = set(reference_route)
        clf_set = set(clf_traversed)
        hmm_set = set(hmm_traversed)

        clf_precision = len(clf_set & ref_set) / max(1, len(clf_set))
        clf_recall = len(clf_set & ref_set) / max(1, len(ref_set))
        hmm_precision = len(hmm_set & ref_set) / max(1, len(hmm_set))
        hmm_recall = len(hmm_set & ref_set) / max(1, len(ref_set))

        comparison["reference_route_metrics"] = {
            "reference_route_length": len(reference_route),
            "classifier_only": {
                "arc_set_precision": round(clf_precision, 4),
                "arc_set_recall": round(clf_recall, 4),
                "arc_set_f1": round(
                    2 * clf_precision * clf_recall / max(1e-8, clf_precision + clf_recall), 4
                ),
            },
            "classifier_plus_hmm": {
                "arc_set_precision": round(hmm_precision, 4),
                "arc_set_recall": round(hmm_recall, 4),
                "arc_set_f1": round(
                    2 * hmm_precision * hmm_recall / max(1e-8, hmm_precision + hmm_recall), 4
                ),
            }
        }

    return comparison

