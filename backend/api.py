"""FastAPI Router for PathSnap Prediction & Decoding

Joint Ownership: Prateek & Priyansh
Hackathon: PathSnap Distributed Map-Matching

Exposes the agreed API response schema (Contract D) for evaluator-facing UI:
{
    "trip_id": "00000001",
    "gps_points": [...],
    "predicted_arc_ids": [...],
    "predicted_route_geometry": [...],
    "metrics": {...},
    "processing_time_ms": 123.4
}
"""

import os
import sys
import time
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
import pandas as pd
import numpy as np

# Ensure root is in path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

try:
    from PathSnap.route_reconstruction.hmm import RoadGraph
    from PathSnap.route_reconstruction.route_decoder import decode_route, extract_traversed_route, compute_route_continuity
    from PathSnap.backend.evaluator import ServerSideEvaluator
except ModuleNotFoundError:
    from route_reconstruction.hmm import RoadGraph
    from route_reconstruction.route_decoder import decode_route, extract_traversed_route, compute_route_continuity
    from backend.evaluator import ServerSideEvaluator

router = APIRouter(prefix="/api", tags=["PathSnap Pipeline"])
evaluator = ServerSideEvaluator()


class TripPredictionRequest(BaseModel):
    trip_id: str
    partition: Optional[str] = "train"


def _find_data_root() -> str:
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


@router.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "PathSnap Map-Matching & Sequential Reconstruction API",
        "version": "1.0.0"
    }


@router.get("/trips")
def list_available_trips(partition: str = Query("student_test")):
    """List available trip IDs for the UI selector."""
    data_root = _find_data_root()
    p_path = os.path.join(data_root, partition)
    if not os.path.exists(p_path):
        return {"trips": [], "error": f"Partition path not found: {p_path}"}

    trips = sorted([d for d in os.listdir(p_path) if os.path.isdir(os.path.join(p_path, d))])
    return {
        "partition": partition,
        "count": len(trips),
        "trips": trips
    }


@router.post("/predict_route")
def predict_route(req: TripPredictionRequest) -> Dict[str, Any]:
    """
    Main evaluation pipeline endpoint.
    Processes the selected trip through candidate generation, Viterbi decoding,
    and returns the predicted route adhering strictly to Contract D.
    """
    t_start = time.perf_counter()
    trip_id = req.trip_id.strip()
    data_root = _find_data_root()

    # Search in requested partition, fallback to other
    trip_path = os.path.join(data_root, req.partition, trip_id)
    if not os.path.exists(trip_path):
        alt = "student_test" if req.partition == "train" else "train"
        alt_path = os.path.join(data_root, alt, trip_id)
        if os.path.exists(alt_path):
            trip_path = alt_path
        else:
            raise HTTPException(status_code=404, detail=f"Trip {trip_id} not found in dataset")

    # 1. Load Track, Nodes, Arcs
    try:
        track_df = pd.read_csv(f"{trip_path}/{trip_id}.track", sep=r"\s+", header=None, names=["lon", "lat", "timestamp"])
        nodes_df = pd.read_csv(f"{trip_path}/{trip_id}.nodes", sep=r"\s+", header=None, names=["lon", "lat"])
        arcs_df = pd.read_csv(f"{trip_path}/{trip_id}.arcs", sep=r"\s+", header=None, names=["from_node", "to_node"])
        arcs_df["arc_id"] = arcs_df.index
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading trip files: {str(e)}")

    # 2. Build RoadGraph
    graph = RoadGraph(arcs_df, nodes_df)

    # 3. Generate candidate road segments (or use pre-scored if available)
    from scipy.spatial import cKDTree
    mid_lat = track_df["lat"].mean()
    cos_lat = np.cos(np.radians(mid_lat))
    scale_y = 111320.0
    scale_x = 111320.0 * cos_lat

    node_xy = np.column_stack([nodes_df["lon"].values * scale_x, nodes_df["lat"].values * scale_y])
    kdtree = cKDTree(node_xy)

    from_nodes = arcs_df["from_node"].values
    to_nodes = arcs_df["to_node"].values
    node_to_arcs = [[] for _ in range(len(nodes_df))]
    for a_id in range(len(arcs_df)):
        node_to_arcs[from_nodes[a_id]].append(a_id)
        node_to_arcs[to_nodes[a_id]].append(a_id)

    track_lons = track_df["lon"].values
    track_lats = track_df["lat"].values
    candidates = []

    for i in range(len(track_df)):
        px = track_lons[i] * scale_x
        py = track_lats[i] * scale_y
        nearby_nodes = kdtree.query_ball_point([px, py], r=80.0)
        cand_set = set()
        for n in nearby_nodes:
            for a in node_to_arcs[n]:
                cand_set.add(a)
        if not cand_set:
            _, near_n = kdtree.query([px, py], k=1)
            for a in node_to_arcs[near_n]:
                cand_set.add(a)

        cands_list = list(cand_set)[:8]
        # Equal prior / distance weight fallback
        candidates.append({
            "observation_index": i,
            "candidate_arc_ids": cands_list,
            "probabilities": [1.0 / len(cands_list)] * len(cands_list)
        })

    # 4. Decode Route using Prateek's Viterbi HMM Decoder
    decode_res = decode_route(
        candidates=candidates,
        arcs=graph,
        timestamps=track_df["timestamp"],
        graph=graph,
        return_full_result=True
    )

    pointwise_arcs = decode_res["pointwise_arcs"]
    traversed_arcs = decode_res["traversed_arcs"]
    continuity = decode_res["pointwise_continuity"]

    # 5. Build predicted route geometry coordinates for UI rendering
    geometry = []
    for arc_id in traversed_arcs:
        ep = graph.get_arc_endpoints(arc_id)
        if ep and ep[0] in graph.nodes_coords and ep[1] in graph.nodes_coords:
            u_coord = graph.nodes_coords[ep[0]]
            v_coord = graph.nodes_coords[ep[1]]
            geometry.append([u_coord[0], u_coord[1]])
            geometry.append([v_coord[0], v_coord[1]])

    # 6. Build GPS points array: [[lon, lat], ...]
    gps_points = [[float(r["lon"]), float(r["lat"])] for _, r in track_df.iterrows()]

    t_end = time.perf_counter()
    exec_ms = round((t_end - t_start) * 1000.0, 2)

    # Adhere strictly to Contract D
    return {
        "trip_id": trip_id,
        "gps_points": gps_points,
        "predicted_arc_ids": traversed_arcs,
        "predicted_route_geometry": geometry,
        "metrics": {
            "relaxed_route_continuity": continuity["relaxed_continuity"],
            "strict_route_continuity": continuity["strict_continuity"],
            "total_observations": len(track_df),
            "traversed_segments_count": len(traversed_arcs),
            "invalid_transitions": continuity["invalid_transitions"]
        },
        "processing_time_ms": exec_ms
    }
