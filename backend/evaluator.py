"""Server-Side Evaluation Service for PathSnap

Joint Ownership: Prateek & Priyansh
Hackathon: PathSnap Distributed Map-Matching

Maintains ground-truth reference routes strictly server-side.
Calculates prediction accuracy and sequence fidelity without ever leaking
ground-truth labels into model inputs, API payloads, or the user interface.
"""

import os
from typing import Dict, Any, List, Optional
import pandas as pd


class ServerSideEvaluator:
    """Calculates evaluation metrics against server-side reference data."""

    def __init__(self, data_root: Optional[str] = None):
        self.data_root = data_root or self._find_data_root()

    def _find_data_root(self) -> str:
        candidates = [
            "Student_folder/Student folder",
            "../Student_folder/Student folder",
            "Student folder",
            "../Student folder",
        ]
        for p in candidates:
            if os.path.exists(p):
                return p
        return "Student_folder/Student folder"

    def get_reference_route(self, trip_id: str, partition: str = "train") -> Optional[List[int]]:
        """Load reference route for training or development evaluation."""
        path = os.path.join(self.data_root, partition, trip_id, f"{trip_id}.route")
        if not os.path.exists(path):
            # Try alternate partition
            alt_partition = "student_test" if partition == "train" else "train"
            alt_path = os.path.join(self.data_root, alt_partition, trip_id, f"{trip_id}.route")
            if os.path.exists(alt_path):
                path = alt_path
            else:
                return None

        try:
            route_df = pd.read_csv(path, header=None, names=["arc_id"])
            return route_df["arc_id"].tolist()
        except Exception:
            return None

    def evaluate_predictions(
        self,
        predicted_arcs: List[int],
        reference_route: List[int]
    ) -> Dict[str, Any]:
        """
        Evaluate predicted arc sequence against the true reference route.
        Server-side calculation only.
        """
        if not reference_route:
            return {"error": "Reference route unavailable"}

        pred_set = set(predicted_arcs)
        ref_set = set(reference_route)

        tp = len(pred_set & ref_set)
        precision = tp / max(1, len(pred_set))
        recall = tp / max(1, len(ref_set))
        f1 = 2 * precision * recall / max(1e-8, precision + recall)

        return {
            "reference_route_length": len(reference_route),
            "predicted_unique_arcs": len(pred_set),
            "matched_arcs": tp,
            "arc_precision": round(precision, 4),
            "arc_recall": round(recall, 4),
            "macro_f1": round(f1, 4),
        }
