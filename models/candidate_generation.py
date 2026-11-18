"""Generate road-arc candidates for each GPS observation.

Expected trip contract:
  gps: DataFrame columns longitude, latitude, timestamp
  nodes: DataFrame columns longitude, latitude (node row index is node_id)
  arcs: DataFrame columns from_node_id, to_node_id (row index is arc_id)
  reference_route: sequence of arc IDs; training/development only
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

EARTH_RADIUS_M = 6_371_008.8


def _local_xy(lon, lat, lon0, lat0):
    """Approximate local metric coordinates around a trip's mean location."""
    lon = np.asarray(lon, dtype=float)
    lat = np.asarray(lat, dtype=float)
    x = np.radians(lon - lon0) * EARTH_RADIUS_M * np.cos(np.radians(lat0))
    y = np.radians(lat - lat0) * EARTH_RADIUS_M
    return np.column_stack([x, y])


def _bearing_deg(a_xy, b_xy):
    delta = b_xy - a_xy
    return (np.degrees(np.arctan2(delta[:, 0], delta[:, 1])) + 360.0) % 360.0


def _angle_difference_deg(a, b):
    return np.abs((a - b + 180.0) % 360.0 - 180.0)


class CandidateGenerator:
    def __init__(self, max_candidates=8, node_search_k=12, max_distance_m=100.0):
        self.max_candidates = int(max_candidates)
        self.node_search_k = int(node_search_k)
        self.max_distance_m = float(max_distance_m)

    def generate(self, gps: pd.DataFrame, nodes: pd.DataFrame, arcs: pd.DataFrame) -> pd.DataFrame:
        gps = gps.reset_index(drop=True)
        nodes = nodes.reset_index(drop=True)
        arcs = arcs.reset_index(drop=True)
        lon0, lat0 = float(gps.longitude.mean()), float(gps.latitude.mean())
        node_xy = _local_xy(nodes.longitude, nodes.latitude, lon0, lat0)
        gps_xy = _local_xy(gps.longitude, gps.latitude, lon0, lat0)
        tree = cKDTree(node_xy)
        arc_from = arcs.from_node_id.to_numpy(dtype=int)
        arc_to = arcs.to_node_id.to_numpy(dtype=int)
        if np.any(arc_from >= len(nodes)) or np.any(arc_to >= len(nodes)):
            raise ValueError("Arc endpoint IDs exceed node table size")
        arc_xy_from = node_xy[arc_from]
        arc_xy_to = node_xy[arc_to]
        arc_vec = arc_xy_to - arc_xy_from
        arc_len = np.linalg.norm(arc_vec, axis=1)
        arc_heading = _bearing_deg(arc_xy_from, arc_xy_to)
        # Node-to-arc adjacency, preserving arc row IDs.
        incident = [[] for _ in range(len(nodes))]
        for arc_id, (u, v) in enumerate(zip(arc_from, arc_to)):
            incident[u].append(arc_id)
            incident[v].append(arc_id)
        gps_heading = np.full(len(gps), np.nan)
        if len(gps) > 1:
            gps_heading[1:] = _bearing_deg(gps_xy[:-1], gps_xy[1:])
            gps_heading[0] = gps_heading[1]
        rows = []
        for obs_idx, point in enumerate(gps_xy):
            _, near_nodes = tree.query(point, k=min(self.node_search_k, len(nodes)))
            near_nodes = np.atleast_1d(near_nodes).astype(int)
            candidate_ids = sorted({a for n in near_nodes for a in incident[n]})
            if not candidate_ids:
                continue
            ids = np.asarray(candidate_ids, dtype=int)
            starts, vectors = arc_xy_from[ids], arc_vec[ids]
            denom = np.sum(vectors * vectors, axis=1)
            denom[denom == 0] = 1e-12
            t = np.clip(np.sum((point - starts) * vectors, axis=1) / denom, 0.0, 1.0)
            projected = starts + t[:, None] * vectors
            distance = np.linalg.norm(projected - point, axis=1)
            keep = distance <= self.max_distance_m
            ids, distance, t, projected = ids[keep], distance[keep], t[keep], projected[keep]
            if len(ids) == 0:
                # Keep nearest candidates even if outside the threshold, useful for diagnostics.
                nearest_order = np.argsort(np.linalg.norm((starts + np.clip(np.sum((point-starts)*vectors, axis=1)/denom,0,1)[:,None]*vectors)-point, axis=1))[:self.max_candidates]
                ids = np.asarray(candidate_ids, dtype=int)[nearest_order]
                t = np.clip(np.sum((point - arc_xy_from[ids]) * arc_vec[ids], axis=1) / np.maximum(np.sum(arc_vec[ids]**2, axis=1),1e-12),0,1)
                projected = arc_xy_from[ids] + t[:,None]*arc_vec[ids]
                distance = np.linalg.norm(projected-point,axis=1)
            order = np.argsort(distance)[:self.max_candidates]
            ids, distance, t, projected = ids[order], distance[order], t[order], projected[order]
            for rank, (arc_id, dist, proj_t, proj) in enumerate(zip(ids, distance, t, projected), start=1):
                rows.append({
                    "observation_index": obs_idx,
                    "candidate_arc_id": int(arc_id),
                    "candidate_rank": rank,
                    "distance_m": float(dist),
                    "projection_fraction": float(proj_t),
                    "arc_length_m": float(arc_len[arc_id]),
                    "arc_heading_deg": float(arc_heading[arc_id]),
                    "gps_heading_deg": float(gps_heading[obs_idx]),
                    "heading_difference_deg": float(_angle_difference_deg(arc_heading[arc_id], gps_heading[obs_idx])) if np.isfinite(gps_heading[obs_idx]) else np.nan,
                    "distance_to_from_node_m": float(np.linalg.norm(point-arc_xy_from[arc_id])),
                    "distance_to_to_node_m": float(np.linalg.norm(point-arc_xy_to[arc_id])),
                    "from_node_id": int(arc_from[arc_id]),
                    "to_node_id": int(arc_to[arc_id]),
                })
        return pd.DataFrame(rows)


def assign_reference_labels(candidates: pd.DataFrame, gps: pd.DataFrame, nodes: pd.DataFrame, arcs: pd.DataFrame, reference_route) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Assign each GPS observation a reference arc by nearest geometry on route arcs.

    This is an explicit approximation: route arc IDs are known, but the files do not
    provide one ground-truth arc per GPS sample. We map each sample to the nearest
    geometric arc in the reference route and then label candidate equality.
    """
    if candidates.empty or len(reference_route) == 0:
        return candidates.assign(label=0), pd.DataFrame()
    route_ids = np.asarray(reference_route, dtype=int)
    arcs = arcs.reset_index(drop=True)
    nodes = nodes.reset_index(drop=True)
    gps = gps.reset_index(drop=True)
    lon0, lat0 = float(gps.longitude.mean()), float(gps.latitude.mean())
    node_xy = _local_xy(nodes.longitude, nodes.latitude, lon0, lat0)
    gps_xy = _local_xy(gps.longitude, gps.latitude, lon0, lat0)
    route_ids = route_ids[(route_ids >= 0) & (route_ids < len(arcs))]
    route_from = arcs.iloc[route_ids].from_node_id.to_numpy(dtype=int)
    route_to = arcs.iloc[route_ids].to_node_id.to_numpy(dtype=int)
    starts, vectors = node_xy[route_from], node_xy[route_to]-node_xy[route_from]
    denom = np.sum(vectors*vectors, axis=1)
    denom[denom == 0] = 1e-12
    nearest_arc = []
    nearest_dist = []
    for point in gps_xy:
        t = np.clip(np.sum((point-starts)*vectors,axis=1)/denom,0,1)
        proj = starts+t[:,None]*vectors
        d = np.linalg.norm(proj-point,axis=1)
        i = int(np.argmin(d))
        nearest_arc.append(int(route_ids[i]))
        nearest_dist.append(float(d[i]))
    obs_labels = pd.DataFrame({"observation_index":np.arange(len(gps)),"reference_arc_id":nearest_arc,"reference_route_distance_m":nearest_dist})
    out = candidates.merge(obs_labels,on="observation_index",how="left")
    out["label"] = (out.candidate_arc_id == out.reference_arc_id).astype(int)
    return out, obs_labels


def candidate_miss_rate(obs_labels: pd.DataFrame, candidates: pd.DataFrame) -> float:
    if obs_labels.empty:
        return float("nan")
    available = candidates.groupby("observation_index").candidate_arc_id.apply(set).to_dict()
    misses = sum(int(row.reference_arc_id not in available.get(row.observation_index,set())) for row in obs_labels.itertuples())
    return misses / len(obs_labels)
