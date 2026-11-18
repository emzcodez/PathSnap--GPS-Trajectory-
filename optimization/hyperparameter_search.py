"""Hyperparameter Search Spaces and Pipelines for PathSnap

Author: Prateek (Sequential Modelling: HMM, Viterbi & Optimisation)
Hackathon: PathSnap Distributed Map-Matching

Defines search spaces and tuning routines for:
1. DecoderSearchSpace: HMM transition parameters (p_stay, p_direct, hop_decay, p_invalid, gap_threshold_s).
2. ClassifierSearchSpace: ML classifier hyperparameters (ready to connect with Prema's models in Phase 2).
3. run_decoder_hyperparameter_search: End-to-end Bayesian search pipeline over training trips.
"""

from typing import Dict, Any, List, Optional
import optuna
import numpy as np
from .bayesian_optimization import BayesianOptimizer
from ..route_reconstruction.hmm import RoadGraph, TransitionModel, EmissionModel
from ..route_reconstruction.viterbi import viterbi_decode
from ..route_reconstruction.route_decoder import compute_route_continuity, extract_traversed_route


class DecoderSearchSpace:
    """Defines the parameter search space for HMM/Viterbi route decoding."""

    @staticmethod
    def sample_parameters(trial: optuna.Trial) -> Dict[str, Any]:
        return {
            "p_stay": trial.suggest_float("p_stay", 0.20, 0.75, step=0.05),
            "p_direct": trial.suggest_float("p_direct", 0.15, 0.50, step=0.05),
            "hop_decay": trial.suggest_float("hop_decay", 0.05, 0.40, step=0.05),
            "p_invalid": trial.suggest_float("p_invalid", 1e-6, 1e-3, log=True),
            "gap_threshold_s": trial.suggest_float("gap_threshold_s", 2.0, 15.0, step=1.0),
            "max_hops": trial.suggest_int("max_hops", 2, 4),
        }


class ClassifierSearchSpace:
    """Search space definitions for candidate classification models."""

    @staticmethod
    def sample_random_forest(trial: optuna.Trial) -> Dict[str, Any]:
        return {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300, step=25),
            "max_depth": trial.suggest_int("max_depth", 5, 25),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 5),
            "criterion": trial.suggest_categorical("criterion", ["gini", "entropy"]),
        }

    @staticmethod
    def sample_gradient_boosting(trial: optuna.Trial) -> Dict[str, Any]:
        return {
            "n_estimators": trial.suggest_int("n_estimators", 50, 250, step=25),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.20, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0, step=0.1),
        }

    @staticmethod
    def sample_logistic_regression(trial: optuna.Trial) -> Dict[str, Any]:
        return {
            "C": trial.suggest_float("C", 1e-4, 1e2, log=True),
            "penalty": trial.suggest_categorical("penalty", ["l2"]),
            "solver": trial.suggest_categorical("solver", ["lbfgs"]),
            "max_iter": 500,
        }


def run_decoder_hyperparameter_search(
    sample_candidates: List[Dict[str, Any]],
    graph: RoadGraph,
    timestamps: Optional[List[float]] = None,
    reference_route: Optional[List[int]] = None,
    n_trials: int = 15,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Execute Bayesian Optimization over decoder parameters to find optimal transition hyperparameters.
    
    Objective:
    Composite score balancing route continuity and reference route overlap (when available).
    """
    optimizer = BayesianOptimizer(
        study_name="decoder_hyperparameter_search",
        direction="maximize",
        seed=seed
    )

    ref_set = set(reference_route) if reference_route is not None else None

    def objective(trial: optuna.Trial) -> float:
        params = DecoderSearchSpace.sample_parameters(trial)
        tm = TransitionModel(
            p_stay=params["p_stay"],
            p_direct=params["p_direct"],
            hop_decay=params["hop_decay"],
            p_invalid=params["p_invalid"],
            gap_threshold_s=params["gap_threshold_s"],
            max_hops=params["max_hops"]
        )

        best_path, _, _ = viterbi_decode(
            candidate_outputs=sample_candidates,
            graph=graph,
            transition_model=tm,
            timestamps=timestamps
        )

        continuity = compute_route_continuity(best_path, graph)["relaxed_continuity"]

        if ref_set is not None and len(ref_set) > 0:
            traversed = extract_traversed_route(best_path)
            traversed_set = set(traversed)
            intersection = len(traversed_set & ref_set)
            precision = intersection / max(1, len(traversed_set))
            recall = intersection / max(1, len(ref_set))
            f1 = 2 * precision * recall / max(1e-8, precision + recall)
            # 60% weight on continuity, 40% on reference F1
            score = 0.60 * continuity + 0.40 * f1
        else:
            score = continuity

        return score

    result = optimizer.optimize(objective, n_trials=n_trials)
    result["study_report"] = optimizer.get_summary_report()
    return result

