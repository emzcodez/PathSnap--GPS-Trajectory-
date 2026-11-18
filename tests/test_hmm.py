"""Unit Tests for HMM Transition and Emission Models

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
except ModuleNotFoundError:
    from route_reconstruction.hmm import RoadGraph, TransitionModel, EmissionModel


@pytest.fixture
def road_graph():
    arcs = [
        {"arc_id": 10, "from_node": 100, "to_node": 101},
        {"arc_id": 11, "from_node": 101, "to_node": 102},
        {"arc_id": 12, "from_node": 102, "to_node": 103},
        {"arc_id": 99, "from_node": 900, "to_node": 901}, # Disconnected arc
    ]
    return RoadGraph(arcs)


def test_road_graph_connectivity(road_graph):
    # Self
    assert road_graph.get_hop_distance(10, 10) == 0
    # Direct
    assert road_graph.is_direct_successor(10, 11) is True
    assert road_graph.get_hop_distance(10, 11) == 1
    # Multi-hop (10 -> 11 -> 12 is 2 hops)
    assert road_graph.get_hop_distance(10, 12) == 2
    # Disconnected
    assert road_graph.is_direct_successor(10, 99) is False
    assert road_graph.get_hop_distance(10, 99) == -1


def test_transition_probabilities_ordering(road_graph):
    tm = TransitionModel(p_stay=0.50, p_direct=0.35, hop_decay=0.25, p_invalid=1e-5)

    log_stay = tm.compute_log_transition(10, 10, road_graph)
    log_direct = tm.compute_log_transition(10, 11, road_graph)
    log_multihop = tm.compute_log_transition(10, 12, road_graph)
    log_invalid = tm.compute_log_transition(10, 99, road_graph)

    # Topological ordering of log likelihoods:
    # Stay >= Direct > Multi-hop > Invalid
    assert log_stay > log_direct
    assert log_direct > log_multihop
    assert log_multihop > log_invalid


def test_gps_gap_handling(road_graph):
    tm = TransitionModel(gap_threshold_s=5.0, p_invalid=1e-5)

    # Normal step: delta_t = 1.0s
    log_normal_invalid = tm.compute_log_transition(10, 99, road_graph, delta_t=1.0)

    # Large GPS gap: delta_t = 30.0s (dropout occurred)
    log_gap_invalid = tm.compute_log_transition(10, 99, road_graph, delta_t=30.0)

    # Gap policy relaxes invalid penalty during dropout so emissions can guide recovery
    assert log_gap_invalid > log_normal_invalid


def test_emission_normalization():
    em = EmissionModel()
    raw_probs = [0.2, 0.6, 0.2]
    log_emiss = em.normalize_log_emissions(raw_probs)
    assert len(log_emiss) == 3
    assert np.all(np.isfinite(log_emiss))
    # Highest probability candidate should have highest log emission
    assert np.argmax(log_emiss) == 1

    # Zero prob test (should not fail with -inf)
    zero_probs = [0.0, 0.0]
    log_zero = em.normalize_log_emissions(zero_probs)
    assert np.all(np.isfinite(log_zero))
