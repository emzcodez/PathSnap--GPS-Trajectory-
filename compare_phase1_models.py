"""Train and compare five Phase-1 classifiers on existing candidate CSVs."""
from pathlib import Path
import json, time
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, log_loss

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
TRAIN_CSV = RESULTS / "phase1_train_candidates.csv"
TEST_CSV = RESULTS / "phase1_student_test_candidates_scored.csv"
FEATURES = [
    "distance_m", "candidate_rank", "projection_fraction", "arc_length_m",
    "heading_difference_deg", "distance_to_from_node_m", "distance_to_to_node_m"
]

def prep(df):
    X = df[FEATURES].copy()
    X["heading_difference_deg"] = X["heading_difference_deg"].fillna(180)
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
    return X, df["label"].astype(int).to_numpy()

def evaluate(name, model, X_test, test_df, y_test, train_time):
    start = time.perf_counter()
    prob = model.predict_proba(X_test)[:, 1]
    infer_time = time.perf_counter() - start
    pred = (prob >= 0.5).astype(int)
    scored = test_df.copy()
    scored["predicted_score"] = prob
    group_cols = ["trip_id", "observation_index"]
    top = scored.sort_values(group_cols + ["predicted_score"], ascending=[True, True, False]).groupby(group_cols).head(1)
    metrics = {
        "model": name,
        "candidate_rows": len(test_df),
        "observations": int(test_df.groupby(group_cols).ngroups),
        "candidate_accuracy": float(accuracy_score(y_test, pred)),
        "candidate_precision": float(precision_score(y_test, pred, zero_division=0)),
        "candidate_recall": float(recall_score(y_test, pred, zero_division=0)),
        "candidate_f1": float(f1_score(y_test, pred, zero_division=0)),
        "candidate_log_loss": float(log_loss(y_test, np.clip(prob, 1e-7, 1-1e-7))) if len(np.unique(y_test)) > 1 else None,
        "observation_top1_accuracy": float((top.candidate_arc_id == top.reference_arc_id).mean()),
        "training_time_s": float(train_time),
        "inference_time_s": float(infer_time),
    }
    return metrics, scored

def main():
    train_df = pd.read_csv(TRAIN_CSV)
    test_df = pd.read_csv(TEST_CSV)
    X_train, y_train = prep(train_df)
    X_test, y_test = prep(test_df)
    models = {
        "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=500, class_weight="balanced", random_state=42)),
        "Decision Tree": DecisionTreeClassifier(max_depth=12, min_samples_leaf=2, class_weight="balanced", random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=120, max_depth=12, min_samples_leaf=2, class_weight="balanced", random_state=42, n_jobs=-1),
        "Extra Trees": ExtraTreesClassifier(n_estimators=120, max_depth=12, min_samples_leaf=2, class_weight="balanced", random_state=42, n_jobs=-1),
        "Hist Gradient Boosting": HistGradientBoostingClassifier(max_iter=100, max_leaf_nodes=31, learning_rate=0.1, random_state=42),
    }
    results=[]
    scored_outputs=[]
    for name, model in models.items():
        print(f"\nTraining {name}...", flush=True)
        start=time.perf_counter()
        if name == "Hist Gradient Boosting":
            # Compensate for class imbalance using per-row weights.
            counts=np.bincount(y_train)
            weights=np.where(y_train == 1, len(y_train)/(2*counts[1]), len(y_train)/(2*counts[0]))
            model.fit(X_train, y_train, sample_weight=weights)
        else:
            model.fit(X_train, y_train)
        train_time=time.perf_counter()-start
        metrics, scored=evaluate(name, model, X_test, test_df, y_test, train_time)
        results.append(metrics)
        scored["model"] = name
        scored_outputs.append(scored)
        print(json.dumps(metrics, indent=2), flush=True)
    summary=pd.DataFrame(results).sort_values("observation_top1_accuracy", ascending=False)
    summary.to_csv(RESULTS/"phase1_model_comparison.csv", index=False)
    pd.concat(scored_outputs, ignore_index=True).to_csv(RESULTS/"phase1_all_models_scored.csv", index=False)
    with open(RESULTS/"phase1_model_comparison.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\n=== MODEL COMPARISON (sorted by observation top-1 accuracy) ===")
    print(summary.to_string(index=False))
    print(f"\nSaved: {RESULTS/'phase1_model_comparison.csv'}")

if __name__ == "__main__":
    main()
