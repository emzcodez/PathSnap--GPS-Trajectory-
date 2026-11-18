"""First-phase baseline: Random Forest candidate classifier."""
from __future__ import annotations
import time
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, log_loss

FEATURE_COLUMNS = [
    "distance_m", "candidate_rank", "projection_fraction", "arc_length_m",
    "heading_difference_deg", "distance_to_from_node_m", "distance_to_to_node_m"
]


def train_baseline(train_df: pd.DataFrame, random_state=42):
    X = train_df[FEATURE_COLUMNS].copy()
    X["heading_difference_deg"] = X["heading_difference_deg"].fillna(180)
    y = train_df.label.astype(int)
    model = RandomForestClassifier(
        n_estimators=120,
        max_depth=12,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
    )
    start = time.perf_counter()
    model.fit(X, y)
    train_time = time.perf_counter()-start
    return model, train_time


def evaluate_candidate_classifier(model, eval_df: pd.DataFrame):
    X = eval_df[FEATURE_COLUMNS].copy()
    X["heading_difference_deg"] = X["heading_difference_deg"].fillna(180)
    y = eval_df.label.astype(int).to_numpy()
    start = time.perf_counter()
    prob = model.predict_proba(X)[:,1]
    infer_time = time.perf_counter()-start
    eval_df = eval_df.copy()
    eval_df["predicted_score"] = prob
    group_cols = ["trip_id", "observation_index"] if "trip_id" in eval_df.columns else ["observation_index"]
    top = eval_df.sort_values(group_cols + ["predicted_score"],ascending=[True]*len(group_cols)+[False]).groupby(group_cols).head(1)
    # Observation-level prediction is correct if top candidate is the reference arc.
    top1 = (top.candidate_arc_id == top.reference_arc_id).mean()
    metrics = {
        "candidate_rows": len(eval_df),
        "observations": eval_df.groupby(group_cols).ngroups,
        "candidate_accuracy": accuracy_score(y, (prob >= 0.5).astype(int)),
        "candidate_precision": precision_score(y, (prob >= 0.5).astype(int), zero_division=0),
        "candidate_recall": recall_score(y, (prob >= 0.5).astype(int), zero_division=0),
        "candidate_f1": f1_score(y, (prob >= 0.5).astype(int), zero_division=0),
        "candidate_log_loss": log_loss(y, np.clip(prob,1e-7,1-1e-7)) if len(np.unique(y))>1 else float("nan"),
        "observation_top1_accuracy": float(top1),
        "inference_time_s": infer_time,
    }
    return metrics, eval_df
