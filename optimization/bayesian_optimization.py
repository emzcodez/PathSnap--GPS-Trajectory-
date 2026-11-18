"""Bayesian Hyperparameter Optimization Framework for PathSnap

Author: Prateek (Sequential Modelling: HMM, Viterbi & Optimisation)
Hackathon: PathSnap Distributed Map-Matching

Provides:
- BayesianOptimizer: Core Bayesian optimization using Optuna's Tree-structured Parzen Estimator (TPE).
- ObjectiveEvaluator: Evaluation protocol supporting GroupKFold cross-validation across trips
  to strictly prevent data leakage between folds.
- Full tracking of search space, trial histories, objective trajectories, and wall-clock compute cost.
"""

from typing import Dict, Any, Callable, Optional, List, Tuple
import optuna
import numpy as np
import time
from dataclasses import dataclass, field
import json


# Suppress overly verbose optuna logs during automated trials
optuna.logging.set_verbosity(optuna.logging.WARNING)


@dataclass
class TrialResult:
    trial_number: int
    params: Dict[str, Any]
    value: float
    duration_seconds: float
    state: str = "COMPLETE"


class BayesianOptimizer:
    """
    Manages Bayesian hyperparameter search studies using Tree-structured Parzen Estimator (TPE).
    """

    def __init__(
        self,
        study_name: str = "pathsnap_optimization",
        direction: str = "maximize",
        seed: int = 42,
        storage: Optional[str] = None
    ):
        """
        Parameters
        ----------
        study_name : str
            Identifier for the study.
        direction : str
            'maximize' (for F1 / accuracy / continuity) or 'minimize' (for log loss / error).
        seed : int
            Reproducible random seed.
        storage : str, optional
            Database URL (e.g. sqlite:///optuna.db) if persistence is desired.
        """
        self.study_name = study_name
        self.direction = direction
        self.seed = seed
        self.sampler = optuna.samplers.TPESampler(seed=seed)
        self.study = optuna.create_study(
            study_name=study_name,
            direction=direction,
            sampler=self.sampler,
            storage=storage,
            load_if_exists=True
        )
        self.trial_history: List[TrialResult] = []
        self.total_optimization_time_s: float = 0.0

    def optimize(
        self,
        objective_fn: Callable[[optuna.Trial], float],
        n_trials: int = 20,
        timeout: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Execute Bayesian optimization trials.

        Parameters
        ----------
        objective_fn : Callable[[optuna.Trial], float]
            Objective function that samples parameters from trial and returns a scalar score.
        n_trials : int
            Number of optimization iterations.
        timeout : float, optional
            Maximum time limit in seconds.

        Returns
        -------
        Dict[str, Any]
            Summary of best parameters, best score, trial count, and compute cost.
        """
        start_time = time.perf_counter()

        def wrapped_objective(trial: optuna.Trial) -> float:
            t0 = time.perf_counter()
            score = objective_fn(trial)
            duration = time.perf_counter() - t0

            self.trial_history.append(
                TrialResult(
                    trial_number=trial.number,
                    params=trial.params,
                    value=float(score),
                    duration_seconds=round(duration, 3),
                    state="COMPLETE"
                )
            )
            return score

        self.study.optimize(wrapped_objective, n_trials=n_trials, timeout=timeout)
        self.total_optimization_time_s = time.perf_counter() - start_time

        best_trial = self.study.best_trial
        return {
            "best_trial_number": best_trial.number,
            "best_value": round(float(best_trial.value), 4),
            "best_params": best_trial.params,
            "total_trials": len(self.study.trials),
            "total_compute_time_seconds": round(self.total_optimization_time_s, 2),
            "direction": self.direction,
        }

    def get_trials_dataframe(self):
        """Return study trials as a Pandas DataFrame."""
        return self.study.trials_dataframe()

    def get_summary_report(self) -> Dict[str, Any]:
        """Generate structured optimization summary report."""
        if not self.study.trials:
            return {"status": "No trials executed yet."}

        return {
            "study_name": self.study_name,
            "direction": self.direction,
            "best_params": self.study.best_params,
            "best_value": round(float(self.study.best_value), 4),
            "total_trials": len(self.study.trials),
            "total_duration_s": round(self.total_optimization_time_s, 2),
            "trials": [
                {
                    "trial": t.trial_number,
                    "params": t.params,
                    "value": t.value,
                    "duration_s": t.duration_seconds
                }
                for t in self.trial_history
            ]
        }


class ObjectiveEvaluator:
    """
    Evaluator with Group-Aware Cross-Validation across trips.
    Guarantees no observation from any trip leaks into validation folds.
    """

    @staticmethod
    def split_trip_groups(
        trip_ids: List[str],
        n_splits: int = 5,
        seed: int = 42
    ) -> List[Tuple[List[str], List[str]]]:
        """
        Split unique trip IDs into K train/val folds.
        """
        unique_trips = np.array(sorted(list(set(trip_ids))))
        rng = np.random.RandomState(seed)
        rng.shuffle(unique_trips)

        folds = []
        splits = np.array_split(unique_trips, n_splits)
        for i in range(n_splits):
            val_trips = list(splits[i])
            train_trips = [t for t in unique_trips if t not in val_trips]
            folds.append((train_trips, val_trips))

        return folds

