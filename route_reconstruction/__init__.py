"""PathSnap Route Reconstruction Package

Owner: Prateek (Sequential Modelling: HMM, Viterbi & Optimisation)
Provides:
    - RoadGraph: Topological road network graph and connectivity analysis.
    - TransitionModel: Dynamic transition probabilities with topology and gap handling.
    - EmissionModel: Observation emission probability normalization.
    - MapMatchingHMM: Hidden Markov Model for map-matching.
    - viterbi_decode: Numerically stable log-space Viterbi algorithm.
    - decode_route: Standardized shared decoder interface.
    - compute_route_continuity: Metric to quantify topological route validity.
"""

from .hmm import RoadGraph, TransitionModel, EmissionModel, MapMatchingHMM
from .viterbi import viterbi_decode
from .route_decoder import decode_route, compute_route_continuity, compare_classifier_vs_hmm

__all__ = [
    "RoadGraph",
    "TransitionModel",
    "EmissionModel",
    "MapMatchingHMM",
    "viterbi_decode",
    "decode_route",
    "compute_route_continuity",
    "compare_classifier_vs_hmm",
]

