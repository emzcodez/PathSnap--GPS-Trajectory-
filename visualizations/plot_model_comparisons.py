"""Model Performance & Accuracy Visualizations for PathSnap

Author: Prateek
Hackathon: PathSnap Distributed Map-Matching

Generates publication-quality comparison plots across all 5 trained models:
1. model_accuracies_comparison.png: Candidate Accuracy, Observation Top-1 Accuracy, Macro F1
2. model_log_loss_and_latency.png: Log Loss and Inference Latency trade-offs
3. model_confusion_matrices.png: Normalized confusion matrices for candidate classification
"""

import os
import sys
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

# Add workspace to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
for p in [current_dir, parent_dir]:
    if p not in sys.path:
        sys.path.insert(0, p)


def plot_all_model_metrics(output_dir: str = None):
    if output_dir is None:
        output_dir = os.path.join(parent_dir, "visualizations")
    os.makedirs(output_dir, exist_ok=True)

    results_path = os.path.join(parent_dir, "results", "five_model_comparison.json")
    if not os.path.exists(results_path):
        from models.model_comparison import run_five_model_comparison
        data = run_five_model_comparison(os.path.join(parent_dir, "results"))
    else:
        with open(results_path, "r") as f:
            data = json.load(f)

    df = pd.DataFrame(data)

    # Style configuration
    plt.rcParams["font.sans-serif"] = "DejaVu Sans"
    plt.rcParams["axes.edgecolor"] = "#cccccc"
    plt.rcParams["axes.linewidth"] = 0.8

    # -------------------------------------------------------------
    # PLOT 1: Accuracies & F1 Comparison Bar Chart
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(11, 6), dpi=300)
    x = np.arange(len(df))
    width = 0.25

    rects1 = ax.bar(x - width, df["accuracy"] * 100, width, label="Candidate Accuracy (%)", color="#2b5c8f", edgecolor="none")
    rects2 = ax.bar(x, df["top1_accuracy"] * 100, width, label="Observation Top-1 Accuracy (%)", color="#2a9d8f", edgecolor="none")
    rects3 = ax.bar(x + width, df["f1_score"] * 100, width, label="Candidate Macro F1 (%)", color="#e76f51", edgecolor="none")

    ax.set_ylabel("Score (%)", fontsize=12, fontweight="bold")
    ax.set_title("Performance Comparison Across All 5 Trained Model Families", fontsize=14, fontweight="bold", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(df["model_name"], fontsize=10, fontweight="bold")
    ax.legend(frameon=True, facecolor="#f8f9fa", edgecolor="#e5e5e5", loc="lower right", fontsize=10)
    ax.set_ylim(40, 105)
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    # Label bars with values
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f"{height:.1f}%",
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha="center", va="bottom", fontsize=8, fontweight="bold")

    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)

    plt.tight_layout()
    path_acc = os.path.join(output_dir, "model_accuracies_comparison.png")
    plt.savefig(path_acc)
    plt.close()
    print(f"Saved: {path_acc}")

    # -------------------------------------------------------------
    # PLOT 2: Log Loss vs Inference Latency Trade-Off
    # -------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)

    # Subplot A: Log Loss (Lower is better)
    colors_loss = ["#2b5c8f", "#d62828", "#f4a261", "#2a9d8f", "#1d3557"]
    bars_loss = ax1.bar(df["model_name"], df["log_loss"], color=colors_loss, width=0.55)
    ax1.set_ylabel("Log Loss (Cross-Entropy)", fontsize=11, fontweight="bold")
    ax1.set_title("Log Loss Comparison (Lower is Better)", fontsize=12, fontweight="bold")
    ax1.set_xticklabels(df["model_name"], rotation=25, ha="right", fontsize=9, fontweight="bold")
    ax1.grid(axis="y", linestyle="--", alpha=0.3)

    for bar in bars_loss:
        h = bar.get_height()
        ax1.annotate(f"{h:.4f}",
                     xy=(bar.get_x() + bar.get_width() / 2, h),
                     xytext=(0, 3), textcoords="offset points",
                     ha="center", va="bottom", fontsize=9, fontweight="bold")

    # Subplot B: Inference Latency (ms)
    colors_time = ["#457b9d", "#457b9d", "#457b9d", "#e63946", "#2a9d8f"]
    bars_time = ax2.bar(df["model_name"], df["infer_time_ms"], color=colors_time, width=0.55)
    ax2.set_ylabel("Inference Latency per Trip (ms)", fontsize=11, fontweight="bold")
    ax2.set_title("Inference Latency Comparison (ms)", fontsize=12, fontweight="bold")
    ax2.set_xticklabels(df["model_name"], rotation=25, ha="right", fontsize=9, fontweight="bold")
    ax2.grid(axis="y", linestyle="--", alpha=0.3)

    for bar in bars_time:
        h = bar.get_height()
        ax2.annotate(f"{h:.1f}ms",
                     xy=(bar.get_x() + bar.get_width() / 2, h),
                     xytext=(0, 3), textcoords="offset points",
                     ha="center", va="bottom", fontsize=9, fontweight="bold")

    plt.tight_layout()
    path_loss_time = os.path.join(output_dir, "model_log_loss_and_latency.png")
    plt.savefig(path_loss_time)
    plt.close()
    print(f"Saved: {path_loss_time}")

    # -------------------------------------------------------------
    # PLOT 3: Confusion Matrices for Top Models
    # -------------------------------------------------------------
    # Load candidate labels and compute confusion matrices for Random Forest and HistGradientBoosting
    train_path = os.path.join(parent_dir, "results", "phase1_train_candidates.csv")
    test_path = os.path.join(parent_dir, "results", "phase1_student_test_candidates_scored.csv")

    if os.path.exists(train_path) and os.path.exists(test_path):
        from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
        features = [
            "distance_m", "candidate_rank", "projection_fraction", "arc_length_m",
            "heading_difference_deg", "distance_to_from_node_m", "distance_to_to_node_m"
        ]
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)

        X_tr = train_df[features].fillna(180)
        y_tr = train_df["label"].astype(int)
        X_te = test_df[features].fillna(180)
        y_te = test_df["label"].astype(int)

        rf = RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1)
        rf.fit(X_tr, y_tr)
        y_pred_rf = (rf.predict_proba(X_te)[:, 1] >= 0.5).astype(int)
        cm_rf = confusion_matrix(y_te, y_pred_rf, normalize="true")

        hgb = HistGradientBoostingClassifier(max_iter=100, max_depth=8, random_state=42)
        hgb.fit(X_tr, y_tr)
        y_pred_hgb = (hgb.predict_proba(X_te)[:, 1] >= 0.5).astype(int)
        cm_hgb = confusion_matrix(y_te, y_pred_hgb, normalize="true")

        fig, (cax1, cax2) = plt.subplots(1, 2, figsize=(11, 4.5), dpi=300)

        im1 = cax1.imshow(cm_rf, interpolation="nearest", cmap="Blues")
        cax1.set_title("Random Forest (Baseline)\nNormalized Confusion Matrix", fontsize=11, fontweight="bold")
        cax1.set_xlabel("Predicted Label", fontweight="bold")
        cax1.set_ylabel("True Reference Label", fontweight="bold")
        cax1.set_xticks([0, 1])
        cax1.set_yticks([0, 1])
        cax1.set_xticklabels(["Non-Ref (0)", "Ref Arc (1)"])
        cax1.set_yticklabels(["Non-Ref (0)", "Ref Arc (1)"])
        for i in range(2):
            for j in range(2):
                cax1.text(j, i, f"{cm_rf[i, j]*100:.1f}%", ha="center", va="center",
                          color="white" if cm_rf[i, j] > 0.5 else "black", fontweight="bold", fontsize=11)

        im2 = cax2.imshow(cm_hgb, interpolation="nearest", cmap="Greens")
        cax2.set_title("HistGradientBoosting (Selected)\nNormalized Confusion Matrix", fontsize=11, fontweight="bold")
        cax2.set_xlabel("Predicted Label", fontweight="bold")
        cax2.set_ylabel("True Reference Label", fontweight="bold")
        cax2.set_xticks([0, 1])
        cax2.set_yticks([0, 1])
        cax2.set_xticklabels(["Non-Ref (0)", "Ref Arc (1)"])
        cax2.set_yticklabels(["Non-Ref (0)", "Ref Arc (1)"])
        for i in range(2):
            for j in range(2):
                cax2.text(j, i, f"{cm_hgb[i, j]*100:.1f}%", ha="center", va="center",
                          color="white" if cm_hgb[i, j] > 0.5 else "black", fontweight="bold", fontsize=11)

        plt.tight_layout()
        path_cm = os.path.join(output_dir, "model_confusion_matrices.png")
        plt.savefig(path_cm)
        plt.close()
        print(f"Saved: {path_cm}")

    print("All model comparison visualisations successfully generated.")


if __name__ == "__main__":
    plot_all_model_metrics()
