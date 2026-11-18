"""Unit Tests for Viterbi Algorithm

Author: Prateek (Sequential Modelling: HMM, Viterbi & Optimisation)
"""

import os
import sys
import pytest
import numpy as np

# Support running from root or inside PathSnap directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
grandparent_dir = os.path.abspath(os.path.join(parent_dir, ".."))
if grandparent_dir not in sys.path:
    sys.path.insert(0, grandparent_dir)

try:
    from PathSnap.route_reconstruction.hmm import RoadGraph, TransitionModel, EmissionModel
    from PathSnap.route_reconstruction.viterbi import viterbi_decode
except ModuleNotFoundError:
    from route_reconstruction.hmm import RoadGraph, TransitionModel, EmissionModel
    from route_reconstruction.viterbi import viterbi_decode


@pytest.fixture
def synthetic_graph():
    """
    Creates a simple synthetic directed road graph:
    Nodes: 0 -> 1 -> 2 -> 3
    Arcs:
      0: 0 -> 1
      1: 1 -> 2
      2: 2 -> 3
      3: 0 -> 2 (shortcut)
      4: 3 -> 0 (loop back)
    """
    arcs = [
        {"arc_id": 0, "from_node": 0, "to_node": 1},
        {"arc_id": 1, "from_node": 1, "to_node": 2},
        {"arc_id": 2, "from_node": 2, "to_node": 3},
        {"arc_id": 3, "from_node": 0, "to_node": 2},
        {"arc_id": 4, "from_node": 3, "to_node": 0},
    ]
    return RoadGraph(arcs)


def test_viterbi_single_observation(synthetic_graph):
    candidates = [
        {"observation_index": 0, "candidate_arc_ids": [0, 1, 2], "probabilities": [0.1, 0.7, 0.2]}
    ]
    best_path, score, diag = viterbi_decode(candidates, synthetic_graph)
    assert best_path == [1], "Should select highest probability candidate for T=1"
    assert diag["total_observations"] == 1


def test_viterbi_deterministic_linear_path(synthetic_graph):
    """
    Observations traversing 0 -> 1 -> 2 with small noise.
    Candidate emissions favor arc 0 at t=0, arc 1 at t=1, arc 2 at t=2.
    """
    candidates = [
        {"observation_index": 0, "candidate_arc_ids": [0, 3], "probabilities": [0.8, 0.2]},
        {"observation_index": 1, "candidate_arc_ids": [0, 1], "probabilities": [0.3, 0.7]},
        {"observation_index": 2, "candidate_arc_ids": [1, 2], "probabilities": [0.2, 0.8]},
    ]
    best_path, score, diag = viterbi_decode(candidates, synthetic_graph)
    assert best_path == [0, 1, 2], f"Expected [0, 1, 2], got {best_path}"
    assert diag["transition_counts"]["direct_successors"] == 2


def test_viterbi_overcomes_emission_noise_with_transition_topology(synthetic_graph):
    """
    At t=1, noisy emission gives arc 4 slightly higher probability than arc 1,
    but arc 4 is disconnected from arc 0 (head of arc 0 is node 1; tail of arc 4 is node 3).
    HMM transition penalty should override the noisy emission and pick arc 1!
    """
    candidates = [
        {"observation_index": 0, "candidate_arc_ids": [0, 3], "probabilities": [0.9, 0.1]},
        {"observation_index": 1, "candidate_arc_ids": [1, 4], "probabilities": [0.45, 0.55]},
        {"observation_index": 2, "candidate_arc_ids": [2], "probabilities": [1.0]},
    ]
    tm = TransitionModel(p_stay=0.4, p_direct=0.5, p_invalid=1e-5)
    best_path, score, diag = viterbi_decode(candidates, synthetic_graph, transition_model=tm)
    assert best_path == [0, 1, 2], "Viterbi should penalize disconnected Arc 4 and maintain valid route [0, 1, 2]"


def test_viterbi_missing_candidate_policy(synthetic_graph):
    """
    Tests handling policy when an intermediate observation has empty candidates.
    """
    candidates = [
        {"observation_index": 0, "candidate_arc_ids": [0], "probabilities": [1.0]},
        {"observation_index": 1, "candidate_arc_ids": [], "probabilities": []},  # Empty!
        {"observation_index": 2, "candidate_arc_ids": [1], "probabilities": [1.0]},
    ]
    best_path, score, diag = viterbi_decode(candidates, synthetic_graph)
    assert len(best_path) == 3
    assert best_path[0] == 0
    assert best_path[2] == 1


def test_viterbi_disconnected_penalty_policy():
    """
    When all candidates are disconnected, Viterbi must not crash with -inf;
    it should apply the invalid transition penalty and return the best available sequence.
    """
    arcs = [
        {"arc_id": 0, "from_node": 0, "to_node": 1},
        {"arc_id": 99, "from_node": 90, "to_node": 91},  # Truly disconnected
    ]
    graph = RoadGraph(arcs)
    candidates = [
        {"observation_index": 0, "candidate_arc_ids": [0], "probabilities": [1.0]},
        {"observation_index": 1, "candidate_arc_ids": [99], "probabilities": [1.0]},  # Disconnected from 0
    ]
    best_path, score, diag = viterbi_decode(candidates, graph)
    assert best_path == [0, 99]
    assert diag["transition_counts"]["invalid_transitions"] == 1
    assert np.isfinite(score), "Score should remain finite"
