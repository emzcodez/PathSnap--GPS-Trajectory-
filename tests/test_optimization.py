"""Unit Tests for Bayesian Optimization Framework

Author: Prateek (Sequential Modelling: HMM, Viterbi & Optimisation)
"""

import os
import sys
import pytest
import optuna

# Support running from root or inside PathSnap directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
grandparent_dir = os.path.abspath(os.path.join(parent_dir, ".."))
if grandparent_dir not in sys.path:
    sys.path.insert(0, grandparent_dir)

try:
    from PathSnap.optimization.bayesian_optimization import BayesianOptimizer, ObjectiveEvaluator
    from PathSnap.optimization.hyperparameter_search import run_decoder_hyperparameter_search
    from PathSnap.route_reconstruction.hmm import RoadGraph
except ModuleNotFoundError:
    from optimization.bayesian_optimization import BayesianOptimizer, ObjectiveEvaluator
    from optimization.hyperparameter_search import run_decoder_hyperparameter_search
    from route_reconstruction.hmm import RoadGraph


def test_bayesian_optimizer_core():
    # Simple quadratic objective: maximize -(x - 3)^2
    optimizer = BayesianOptimizer(study_name="test_quad", direction="maximize", seed=42)

    def objective(trial: optuna.Trial) -> float:
        x = trial.suggest_float("x", -10.0, 10.0)
        return -((x - 3.0) ** 2)

    res = optimizer.optimize(objective, n_trials=10)
    assert res["total_trials"] == 10
    assert abs(res["best_params"]["x"] - 3.0) < 3.0
    assert len(optimizer.trial_history) == 10

    report = optimizer.get_summary_report()
    assert report["total_trials"] == 10
    assert "best_params" in report


def test_group_aware_cross_validation_no_leakage():
    # 20 trips
    trip_ids = [f"trip_{i:03d}" for i in range(20)]
    folds = ObjectiveEvaluator.split_trip_groups(trip_ids, n_splits=5, seed=42)

    assert len(folds) == 5
    for train_trips, val_trips in folds:
        train_set = set(train_trips)
        val_set = set(val_trips)
        # Strict isolation: no trip in both train and val
        assert len(train_set & val_set) == 0
        assert len(train_set) + len(val_set) == 20


def test_decoder_hyperparameter_search_pipeline():
    arcs = [
        {"arc_id": 0, "from_node": 1, "to_node": 2},
        {"arc_id": 1, "from_node": 2, "to_node": 3},
        {"arc_id": 2, "from_node": 3, "to_node": 4},
    ]
    graph = RoadGraph(arcs)
    candidates = [
        {"observation_index": 0, "candidate_arc_ids": [0, 1], "probabilities": [0.8, 0.2]},
        {"observation_index": 1, "candidate_arc_ids": [0, 1], "probabilities": [0.3, 0.7]},
        {"observation_index": 2, "candidate_arc_ids": [1, 2], "probabilities": [0.2, 0.8]},
    ]
    timestamps = [1.0, 2.0, 3.0]
    ref_route = [0, 1, 2]

    # Run quick 5-trial search
    opt_res = run_decoder_hyperparameter_search(
        sample_candidates=candidates,
        graph=graph,
        timestamps=timestamps,
        reference_route=ref_route,
        n_trials=5,
        seed=42
    )

    assert opt_res["total_trials"] == 5
    assert "p_stay" in opt_res["best_params"]
    assert "p_direct" in opt_res["best_params"]
    assert opt_res["best_value"] > 0.0
