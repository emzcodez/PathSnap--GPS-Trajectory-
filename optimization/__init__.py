"""PathSnap Optimization Package

Owner: Prateek (Sequential Modelling: HMM, Viterbi & Optimisation)
Provides:
    - BayesianOptimizer: Core Optuna-powered Bayesian optimization manager.
    - HyperparameterSearch: Search space definitions and tuning pipelines.
    - OptimizationResults: Experiment logging, serialization, and report generation.
"""

from .bayesian_optimization import BayesianOptimizer, ObjectiveEvaluator
from .hyperparameter_search import DecoderSearchSpace, ClassifierSearchSpace, run_decoder_hyperparameter_search
from .optimization_results import OptimizationTrialRecord, OptimizationExperimentReport

__all__ = [
    "BayesianOptimizer",
    "ObjectiveEvaluator",
    "DecoderSearchSpace",
    "ClassifierSearchSpace",
    "run_decoder_hyperparameter_search",
    "OptimizationTrialRecord",
    "OptimizationExperimentReport",
]
