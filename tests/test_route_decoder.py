"""Unit Tests for Route Decoder Interface and Comparison Metrics

Author: Prateek (Sequential Modelling: HMM, Viterbi & Optimisation)
"""

import os
import sys
import pytest
import pandas as pd

# Support running from root or inside PathSnap directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
grandparent_dir = os.path.abspath(os.path.join(parent_dir, ".."))
if grandparent_dir not in sys.path:
    sys.path.insert(0, grandparent_dir)

try:
    from PathSnap.route_reconstruction.hmm import RoadGraph
    from PathSnap.route_reconstruction.route_decoder import (
        decode_route,
        compute_route_continuity,
        extract_traversed_route,
        compare_classifier_vs_hmm
    )
except ModuleNotFoundError:
    from route_reconstruction.hmm import RoadGraph
    from route_reconstruction.route_decoder import (
        decode_route,
        compute_route_continuity,
        extract_traversed_route,
        compare_classifier_vs_hmm
    )


@pytest.fixture
def mock_trip():
    # Synthetic road network: 0 -> 1 -> 2 -> 3
    arcs_df = pd.DataFrame([
        {"arc_id": 0, "from_node": 100, "to_node": 101},
        {"arc_id": 1, "from_node": 101, "to_node": 102},
        {"arc_id": 2, "from_node": 102, "to_node": 103},
        {"arc_id": 9, "from_node": 999, "to_node": 998}, # Disconnected noise arc
    ])
    timestamps = [1.0, 2.0, 3.0, 4.0, 5.0]
    return {"arcs": arcs_df, "timestamps": timestamps}


def test_decode_route_default_contract(mock_trip):
    candidates = [
        {"observation_index": 0, "candidate_arc_ids": [0, 9], "probabilities": [0.8, 0.2]},
        {"observation_index": 1, "candidate_arc_ids": [0, 1], "probabilities": [0.7, 0.3]},
        {"observation_index": 2, "candidate_arc_ids": [1, 9], "probabilities": [0.85, 0.15]},
        {"observation_index": 3, "candidate_arc_ids": [1, 2], "probabilities": [0.6, 0.4]},
        {"observation_index": 4, "candidate_arc_ids": [2], "probabilities": [1.0]},
    ]
    # Standard team contract call
    predicted_arcs = decode_route(
        candidates=candidates,
        arcs=mock_trip["arcs"],
        timestamps=mock_trip["timestamps"]
    )
    assert isinstance(predicted_arcs, list)
    assert len(predicted_arcs) == 5
    # The vehicle stays on arc 0, transitions to 1, stays on 1, transitions to 2
    assert predicted_arcs == [0, 0, 1, 1, 2]


def test_extract_traversed_route():
    pointwise = [10, 10, 10, 11, 11, 12, 12, 12]
    traversed = extract_traversed_route(pointwise)
    assert traversed == [10, 11, 12]


def test_compute_route_continuity(mock_trip):
    graph = RoadGraph(mock_trip["arcs"])
    # Fully continuous sequence
    continuous_seq = [0, 0, 1, 1, 2]
    res_cont = compute_route_continuity(continuous_seq, graph)
    assert res_cont["strict_continuity"] == 1.0
    assert res_cont["invalid_transitions"] == 0

    # Discontinuous sequence (jumping to arc 9)
    discontinuous_seq = [0, 9, 2]
    res_disc = compute_route_continuity(discontinuous_seq, graph)
    assert res_disc["invalid_transitions"] == 2
    assert res_disc["strict_continuity"] == 0.0


def test_compare_classifier_vs_hmm(mock_trip):
    # Scenario: Pointwise classifier makes an erroneous prediction (arc 9) at t=1
    # because of local sensor noise
    candidates = [
        {"observation_index": 0, "candidate_arc_ids": [0, 9], "probabilities": [0.9, 0.1]},
        # Classifier gives noisy arc 9 highest probability:
        {"observation_index": 1, "candidate_arc_ids": [0, 9], "probabilities": [0.4, 0.6]},
        {"observation_index": 2, "candidate_arc_ids": [1, 9], "probabilities": [0.9, 0.1]},
    ]
    ref_route = [0, 1]

    comp = compare_classifier_vs_hmm(
        candidates=candidates,
        arcs=mock_trip["arcs"],
        reference_route=ref_route
    )
    # HMM should correct the noisy jump to arc 9 and produce higher continuity than classifier
    assert comp["classifier_plus_hmm"]["pointwise_relaxed_continuity"] > comp["classifier_only"]["pointwise_relaxed_continuity"]
    assert comp["improvements"]["invalid_transitions_eliminated"] >= 1
