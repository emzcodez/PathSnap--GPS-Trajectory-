"""Five-Model Experimentation and Comparison for PathSnap

Hackathon: PathSnap Distributed Map-Matching
Evaluates 5 distinct algorithm families on genuine candidate data:
1. Linear: Logistic Regression
2. Probabilistic: Gaussian Naive Bayes
3. Tree-based: Decision Tree
4. Ensemble: Random Forest (Baseline)
5. Boosting: HistGradientBoosting
"""

import time
import json
import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, log_loss

FEATURE_COLUMNS = [
    "distance_m",
    "candidate_rank",
    "projection_fraction",
    "arc_length_m",
    "heading_difference_deg",
    "distance_to_from_node_m",
    "distance_to_to_node_m",
]


def run_five_model_comparison(data_dir: str = "results"):
    train_path = os.path.join(data_dir, "phase1_train_candidates.csv")
    test_path = os.path.join(data_dir, "phase1_student_test_candidates_scored.csv")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    X_train = train_df[FEATURE_COLUMNS].copy()
    X_train["heading_difference_deg"] = X_train["heading_difference_deg"].fillna(180)
    y_train = train_df["label"].astype(int)

    X_test = test_df[FEATURE_COLUMNS].copy()
    X_test["heading_difference_deg"] = X_test["heading_difference_deg"].fillna(180)
    y_test = test_df["label"].astype(int)

    models = {
        "Logistic Regression": {
            "family": "Linear & Probabilistic",
            "model": LogisticRegression(max_iter=500, random_state=42)
        },
        "Gaussian Naive Bayes": {
            "family": "Probabilistic / Generative",
            "model": GaussianNB()
        },
        "Decision Tree": {
            "family": "Tree-based",
            "model": DecisionTreeClassifier(max_depth=10, random_state=42)
        },
        "Random Forest (Baseline)": {
            "family": "Ensemble (Bagging)",
            "model": RandomForestClassifier(n_estimators=100, max_depth=12, min_samples_leaf=2, random_state=42, n_jobs=-1)
        },
        "HistGradientBoosting": {
            "family": "Boosting",
            "model": HistGradientBoostingClassifier(max_iter=100, max_depth=8, random_state=42)
        }
    }

    results = []
    group_cols = ["trip_id", "observation_index"]

    for name, entry in models.items():
        clf = entry["model"]
        fam = entry["family"]

        t0 = time.perf_counter()
        clf.fit(X_train, y_train)
        train_time = time.perf_counter() - t0

        t0 = time.perf_counter()
        prob = clf.predict_proba(X_test)[:, 1]
        infer_time = time.perf_counter() - t0

        pred = (prob >= 0.5).astype(int)
        acc = accuracy_score(y_test, pred)
        f1 = f1_score(y_test, pred, zero_division=0)
        prec = precision_score(y_test, pred, zero_division=0)
        rec = recall_score(y_test, pred, zero_division=0)
        ll = log_loss(y_test, np.clip(prob, 1e-7, 1 - 1e-7))

        eval_df = test_df.copy()
        eval_df["predicted_score"] = prob
        top = eval_df.sort_values(group_cols + ["predicted_score"], ascending=[True]*len(group_cols)+[False]).groupby(group_cols).head(1)
        top1 = (top["candidate_arc_id"] == top["reference_arc_id"]).mean()

        results.append({
            "model_name": name,
            "algorithm_family": fam,
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
            "log_loss": round(ll, 4),
            "top1_accuracy": round(top1, 4),
            "train_time_s": round(train_time, 2),
            "infer_time_ms": round(infer_time * 1000, 2)
        })

    out_file = os.path.join(data_dir, "five_model_comparison.json")
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved five-model comparison to {out_file}")
    return results


if __name__ == "__main__":
    run_five_model_comparison()
