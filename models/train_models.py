"""Train All Five Models for PathSnap

Hackathon: PathSnap Distributed Map-Matching
Repository Path: PathSnap/models/train_models.py

Trains and persists all 5 distinct model families required by the hackathon:
1. Linear: Logistic Regression
2. Probabilistic: Gaussian Naive Bayes
3. Tree-based: Decision Tree
4. Ensemble: Random Forest (Baseline)
5. Boosting: HistGradientBoosting (Selected for Phase 2)

Saves trained model artifacts (.joblib) to artifacts/models/
and records evaluation metrics to results/five_model_training_summary.json.
"""

from __future__ import annotations
import os
import sys
import time
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, log_loss

# Add parent directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

FEATURE_COLUMNS = [
    "distance_m",
    "candidate_rank",
    "projection_fraction",
    "arc_length_m",
    "heading_difference_deg",
    "distance_to_from_node_m",
    "distance_to_to_node_m",
]


def get_model_definitions():
    """Returns the 5 distinct model architectures and hyperparameter configurations."""
    return {
        "Logistic Regression": {
            "family": "Linear & Probabilistic",
            "filename": "logistic_regression.joblib",
            "model": LogisticRegression(
                C=1.0,
                max_iter=500,
                solver="lbfgs",
                random_state=42
            ),
            "description": "L2-regularized logistic regression for linear decision boundaries"
        },
        "Gaussian Naive Bayes": {
            "family": "Probabilistic / Generative",
            "filename": "gaussian_nb.joblib",
            "model": GaussianNB(var_smoothing=1e-9),
            "description": "Generative probabilistic classifier assuming feature conditional independence"
        },
        "Decision Tree": {
            "family": "Tree-based",
            "filename": "decision_tree.joblib",
            "model": DecisionTreeClassifier(
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                criterion="gini",
                random_state=42
            ),
            "description": "Single decision tree with depth pruning to mitigate overfitting"
        },
        "Random Forest (Baseline)": {
            "family": "Ensemble (Bagging)",
            "filename": "random_forest.joblib",
            "model": RandomForestClassifier(
                n_estimators=100,
                max_depth=12,
                min_samples_leaf=2,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1
            ),
            "description": "Bagged ensemble of randomized decision trees with balanced class weighting"
        },
        "HistGradientBoosting": {
            "family": "Boosting",
            "filename": "hist_gradient_boosting.joblib",
            "model": HistGradientBoostingClassifier(
                max_iter=100,
                max_depth=8,
                learning_rate=0.1,
                min_samples_leaf=20,
                random_state=42
            ),
            "description": "Histogram-based gradient boosted decision trees for fast inference and calibrated probabilities"
        }
    }


def train_and_evaluate_all_models(
    train_csv_path: str = None,
    test_csv_path: str = None,
    output_artifacts_dir: str = None,
    results_dir: str = None
):
    """
    Trains all 5 models, persists their weights, and evaluates on the test partition.
    """
    if train_csv_path is None:
        train_csv_path = os.path.join(parent_dir, "results", "phase1_train_candidates.csv")
    if test_csv_path is None:
        test_csv_path = os.path.join(parent_dir, "results", "phase1_student_test_candidates_scored.csv")
    if output_artifacts_dir is None:
        output_artifacts_dir = os.path.join(parent_dir, "artifacts", "models")
    if results_dir is None:
        results_dir = os.path.join(parent_dir, "results")

    os.makedirs(output_artifacts_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    print("=" * 75)
    print("PathSnap: Training All 5 Model Architectures")
    print("=" * 75)
    print(f"Loading training data from: {train_csv_path}")
    train_df = pd.read_csv(train_csv_path)
    test_df = pd.read_csv(test_csv_path)

    print(f"Training set: {len(train_df)} candidate rows | Positive labels: {int(train_df['label'].sum())}")
    print(f"Student test set: {len(test_df)} candidate rows | Positive labels: {int(test_df['label'].sum())}")

    X_train = train_df[FEATURE_COLUMNS].copy()
    X_train["heading_difference_deg"] = X_train["heading_difference_deg"].fillna(180.0)
    y_train = train_df["label"].astype(int).to_numpy()

    X_test = test_df[FEATURE_COLUMNS].copy()
    X_test["heading_difference_deg"] = X_test["heading_difference_deg"].fillna(180.0)
    y_test = test_df["label"].astype(int).to_numpy()

    models_dict = get_model_definitions()
    results_summary = []
    group_cols = ["trip_id", "observation_index"] if "trip_id" in test_df.columns else ["observation_index"]

    for name, conf in models_dict.items():
        clf = conf["model"]
        filename = conf["filename"]
        family = conf["family"]

        print(f"\n[{family}] Training {name}...")
        t0 = time.perf_counter()
        clf.fit(X_train, y_train)
        train_time_s = time.perf_counter() - t0

        # Persist model weights
        save_path = os.path.join(output_artifacts_dir, filename)
        joblib.dump(clf, save_path)
        file_size_kb = os.path.getsize(save_path) / 1024.0
        print(f"  --> Saved model artifact to: {save_path} ({file_size_kb:.1f} KB)")

        # Inference on student test set
        t0 = time.perf_counter()
        probs = clf.predict_proba(X_test)[:, 1]
        infer_time_s = time.perf_counter() - t0
        infer_time_ms = infer_time_s * 1000.0

        preds = (probs >= 0.5).astype(int)
        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds, zero_division=0)
        rec = recall_score(y_test, preds, zero_division=0)
        f1 = f1_score(y_test, preds, zero_division=0)
        ll = log_loss(y_test, np.clip(probs, 1e-7, 1.0 - 1e-7))

        # Observation-level Top-1 Accuracy: candidate with highest probability must equal reference arc
        temp_eval = test_df.copy()
        temp_eval["predicted_score"] = probs
        top_cands = temp_eval.sort_values(
            group_cols + ["predicted_score"],
            ascending=[True] * len(group_cols) + [False]
        ).groupby(group_cols).head(1)
        top1_acc = (top_cands["candidate_arc_id"] == top_cands["reference_arc_id"]).mean()

        print(f"  Training time: {train_time_s:.2f}s | Inference latency: {infer_time_ms:.1f}ms")
        print(f"  Candidate Acc: {acc*100:.2f}% | Top-1 Acc: {top1_acc*100:.2f}% | F1: {f1:.4f} | Log Loss: {ll:.4f}")

        results_summary.append({
            "model_name": name,
            "algorithm_family": family,
            "artifact_file": filename,
            "artifact_size_kb": round(file_size_kb, 1),
            "training_time_s": round(train_time_s, 3),
            "inference_time_ms": round(infer_time_ms, 2),
            "candidate_accuracy": round(float(acc), 4),
            "observation_top1_accuracy": round(float(top1_acc), 4),
            "candidate_precision": round(float(prec), 4),
            "candidate_recall": round(float(rec), 4),
            "candidate_f1": round(float(f1), 4),
            "log_loss": round(float(ll), 4),
            "configuration": str(clf)
        })

    # Save consolidated results JSON
    summary_path = os.path.join(results_dir, "five_model_training_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results_summary, f, indent=2)

    print("\n" + "=" * 75)
    print("CONSOLIDATED 5-MODEL TRAINING & EVALUATION SUMMARY")
    print("=" * 75)
    header = f"{'Model':<25} {'Accuracy':<10} {'Top-1 Acc':<11} {'F1-Score':<10} {'Log Loss':<10} {'Train(s)':<9} {'Infer(ms)':<10}"
    print(header)
    print("-" * 85)
    for r in results_summary:
        line = f"{r['model_name']:<25} {r['candidate_accuracy']*100:<9.2f}% {r['observation_top1_accuracy']*100:<10.2f}% {r['candidate_f1']:<10.4f} {r['log_loss']:<10.4f} {r['training_time_s']:<9.2f} {r['inference_time_ms']:<10.1f}"
        print(line)

    print(f"\nAll 5 trained model artifacts saved in: {output_artifacts_dir}")
    print(f"Summary report written to: {summary_path}")
    return results_summary


if __name__ == "__main__":
    train_and_evaluate_all_models()
