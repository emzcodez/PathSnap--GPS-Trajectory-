"""Feature preparation for candidate-level binary classification/ranking."""
from __future__ import annotations
import pandas as pd

FEATURE_COLUMNS = [
    "distance_m",
    "candidate_rank",
    "projection_fraction",
    "arc_length_m",
    "heading_difference_deg",
    "distance_to_from_node_m",
    "distance_to_to_node_m",
]


def build_feature_table(candidate_labels: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    if candidate_labels.empty:
        return pd.DataFrame(columns=FEATURE_COLUMNS), pd.Series(dtype=int)
    X = candidate_labels[FEATURE_COLUMNS].copy()
    X["heading_difference_deg"] = X["heading_difference_deg"].fillna(180.0)
    y = candidate_labels["label"].astype(int).copy()
    return X, y
