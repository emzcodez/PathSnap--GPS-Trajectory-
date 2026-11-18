"""HMM Formulation and Graph Transition Modeling for PathSnap

Author: Prateek (Sequential Modelling: HMM, Viterbi & Optimisation)
Hackathon: PathSnap Distributed Map-Matching

This module models the Hidden Markov Model (HMM) components for map-matching:
- Hidden States: Candidate road-network arcs for each GPS observation.
- Emissions: Candidate probabilities derived from the classifier or distance metrics.
- Transitions: Directed road-network topological connectivity, shortest hop distance,
  handling of invalid/disconnected transitions, and temporal GPS gap adaptation.
"""

from typing import List, Dict, Tuple, Optional, Set, Any
import numpy as np
import pandas as pd
from collections import defaultdict, deque


class RoadGraph:
    """Represents the directed road network graph for topological connectivity analysis."""

    def __init__(self, arcs_data: Any, nodes_data: Optional[Any] = None):
        """
        Initialize RoadGraph.

        Parameters
        ----------
        arcs_data : pd.DataFrame or list of tuples
            Contains columns or records with 'from_node' and 'to_node' (and optionally 'arc_id').
            If 0-indexed without 'arc_id', the row index is used as arc_id.
        nodes_data : pd.DataFrame, optional
            Contains columns or records with 'lon', 'lat' (0-indexed node_id).
        """
        self.arc_to_nodes: Dict[int, Tuple[int, int]] = {}
        self.node_outgoing_arcs: Dict[int, List[int]] = defaultdict(list)
        self.node_incoming_arcs: Dict[int, List[int]] = defaultdict(list)
        self.nodes_coords: Dict[int, Tuple[float, float]] = {}

        self._parse_arcs(arcs_data)
        if nodes_data is not None:
            self._parse_nodes(nodes_data)

        # LRU/memoization cache for hop distances between arcs
        self._hop_cache: Dict[Tuple[int, int], int] = {}

    def _parse_arcs(self, arcs_data: Any) -> None:
        if isinstance(arcs_data, pd.DataFrame):
            has_arc_id = 'arc_id' in arcs_data.columns
            from_col = 'from_node' if 'from_node' in arcs_data.columns else arcs_data.columns[0]
            to_col = 'to_node' if 'to_node' in arcs_data.columns else arcs_data.columns[1]

            for idx, row in arcs_data.iterrows():
                arc_id = int(row['arc_id']) if has_arc_id else int(idx)
                u = int(row[from_col])
                v = int(row[to_col])
                self.arc_to_nodes[arc_id] = (u, v)
                self.node_outgoing_arcs[u].append(arc_id)
                self.node_incoming_arcs[v].append(arc_id)
        elif isinstance(arcs_data, (list, tuple)):
            for idx, item in enumerate(arcs_data):
                if isinstance(item, dict):
                    arc_id = item.get('arc_id', idx)
                    u = int(item['from_node'])
                    v = int(item['to_node'])
                else:
                    arc_id = idx
                    u = int(item[0])
                    v = int(item[1])
                self.arc_to_nodes[arc_id] = (u, v)
                self.node_outgoing_arcs[u].append(arc_id)
                self.node_incoming_arcs[v].append(arc_id)

    def _parse_nodes(self, nodes_data: Any) -> None:
        if isinstance(nodes_data, pd.DataFrame):
            lon_col = 'lon' if 'lon' in nodes_data.columns else nodes_data.columns[0]
            lat_col = 'lat' if 'lat' in nodes_data.columns else nodes_data.columns[1]
            for idx, row in nodes_data.iterrows():
                node_id = int(row['node_id']) if 'node_id' in nodes_data.columns else int(idx)
                self.nodes_coords[node_id] = (float(row[lon_col]), float(row[lat_col]))

    @property
    def num_arcs(self) -> int:
        return len(self.arc_to_nodes)

    @property
    def num_nodes(self) -> int:
        return len(self.nodes_coords)

    def get_arc_endpoints(self, arc_id: int) -> Optional[Tuple[int, int]]:
        return self.arc_to_nodes.get(arc_id)

    def is_direct_successor(self, prev_arc: int, next_arc: int) -> bool:
        """Returns True if the head of prev_arc connects directly to the tail of next_arc."""
        if prev_arc not in self.arc_to_nodes or next_arc not in self.arc_to_nodes:
            return False
        return self.arc_to_nodes[prev_arc][1] == self.arc_to_nodes[next_arc][0]

    def get_hop_distance(self, prev_arc: int, next_arc: int, max_hops: int = 4) -> int:
        """
        Calculate directed topological hop distance from prev_arc to next_arc.
        
        Returns:
            0 if prev_arc == next_arc (self-transition)
            1 if next_arc directly succeeds prev_arc (head(u->v) == tail(v->w))
            k if reachable in k hops (where 1 < k <= max_hops)
            -1 if unreachable within max_hops or disconnected
        """
        if prev_arc == next_arc:
            return 0
        
        cache_key = (prev_arc, next_arc)
        if cache_key in self._hop_cache:
            return self._hop_cache[cache_key]

        if prev_arc not in self.arc_to_nodes or next_arc not in self.arc_to_nodes:
            self._hop_cache[cache_key] = -1
            return -1

        u_prev, v_prev = self.arc_to_nodes[prev_arc]
        u_next, v_next = self.arc_to_nodes[next_arc]

        if v_prev == u_next:
            self._hop_cache[cache_key] = 1
            return 1

        # Breadth-First Search on directed arc topology up to max_hops
        # Queue holds: (current_node, current_hops)
        queue = deque([(v_prev, 1)])
        visited_nodes: Set[int] = {v_prev}

        while queue:
            curr_node, hops = queue.popleft()
            if hops >= max_hops:
                continue

            for out_arc in self.node_outgoing_arcs.get(curr_node, []):
                next_u, next_v = self.arc_to_nodes[out_arc]
                if out_arc == next_arc:
                    dist = hops
                    self._hop_cache[cache_key] = dist
                    return dist
                if next_u == u_next:
                    dist = hops + 1
                    self._hop_cache[cache_key] = dist
                    return dist
                if next_v not in visited_nodes:
                    visited_nodes.add(next_v)
                    queue.append((next_v, hops + 1))

        self._hop_cache[cache_key] = -1
        return -1


class TransitionModel:
    """
    Computes mathematically rigorous transition log-probabilities between candidate road segments.
    
    Handles:
    - Same-segment self-transition (staying on same arc)
    - 1-step direct successor transition (driving across connected intersection)
    - Multi-hop reachable transition (intermediate segments traversed)
    - Invalid / disconnected transitions (fallback penalty policy)
    - Temporal GPS gap adaptation (attenuating transition strictness during dropouts)
    """

    def __init__(
        self,
        p_stay: float = 0.50,
        p_direct: float = 0.35,
        hop_decay: float = 0.25,
        p_invalid: float = 1e-5,
        gap_threshold_s: float = 5.0,
        max_hops: int = 4
    ):
        """
        Parameters
        ----------
        p_stay : float
            Baseline probability of staying on the same arc between consecutive observations.
        p_direct : float
            Baseline probability of transitioning to a direct successor arc.
        hop_decay : float
            Decay factor per additional topological hop (p_direct * (hop_decay ** (hop - 1))).
        p_invalid : float
            Fallback penalty probability for disconnected or invalid transitions.
            Crucial policy: Never set to 0 (-inf) to prevent trellis collapse under GPS noise.
        gap_threshold_s : float
            Time gap threshold (in seconds) beyond which GPS dropout is detected.
        max_hops : int
            Maximum BFS depth for network reachability search.
        """
        self.p_stay = p_stay
        self.p_direct = p_direct
        self.hop_decay = hop_decay
        self.p_invalid = max(p_invalid, 1e-9)
        self.gap_threshold_s = gap_threshold_s
        self.max_hops = max_hops

        # Precompute log baselines
        self.log_p_stay = float(np.log(self.p_stay))
        self.log_p_direct = float(np.log(self.p_direct))
        self.log_p_invalid = float(np.log(self.p_invalid))

    def compute_log_transition(
        self,
        prev_arc: int,
        next_arc: int,
        graph: RoadGraph,
        delta_t: Optional[float] = None
    ) -> float:
        """
        Compute the log transition probability ln P(S_{t+1} = next_arc | S_t = prev_arc).
        """
        # 1. Check self-transition
        if prev_arc == next_arc:
            # If large time gap occurred, staying on the exact same short arc is less likely
            if delta_t is not None and delta_t > self.gap_threshold_s:
                # Attenuate stay probability proportional to elapsed time
                gap_factor = min(1.0, self.gap_threshold_s / delta_t)
                return float(np.log(max(self.p_stay * gap_factor, self.p_invalid * 10)))
            return self.log_p_stay

        # 2. Check directed graph reachability
        hops = graph.get_hop_distance(prev_arc, next_arc, max_hops=self.max_hops)

        if hops == 1:
            # Direct successor
            return self.log_p_direct
        elif hops > 1:
            # Multi-hop reachable: exponential decay with hop count
            prob = self.p_direct * (self.hop_decay ** (hops - 1))
            return float(np.log(max(prob, self.p_invalid * 10)))

        # 3. Invalid / disconnected transition (hops == -1)
        # Apply handling policy for invalid transitions
        if delta_t is not None and delta_t > self.gap_threshold_s:
            # In a GPS dropout/gap, the vehicle might have traversed multiple unseen segments.
            # Relax the penalty during gaps so emission probability guides the state recovery.
            relaxed_p_invalid = min(0.01, self.p_invalid * (delta_t / self.gap_threshold_s))
            return float(np.log(relaxed_p_invalid))

        return self.log_p_invalid

    def compute_transition_matrix(
        self,
        prev_candidates: List[int],
        next_candidates: List[int],
        graph: RoadGraph,
        delta_t: Optional[float] = None
    ) -> np.ndarray:
        """
        Construct a (|prev_candidates| x |next_candidates|) log-transition matrix.
        """
        n_prev = len(prev_candidates)
        n_next = len(next_candidates)
        log_mat = np.empty((n_prev, n_next), dtype=np.float64)

        for i, prev_arc in enumerate(prev_candidates):
            for j, next_arc in enumerate(next_candidates):
                log_mat[i, j] = self.compute_log_transition(prev_arc, next_arc, graph, delta_t)

        return log_mat


class EmissionModel:
    """
    Normalizes candidate observation probabilities into stable log emission scores.
    """

    def __init__(self, eps: float = 1e-8):
        self.eps = eps

    def normalize_log_emissions(self, probabilities: List[float]) -> np.ndarray:
        """
        Normalize raw probabilities / confidence scores and return log emission array.
        """
        probs = np.array(probabilities, dtype=np.float64)
        total = np.sum(probs)
        if total > 0:
            probs = probs / total
        else:
            probs = np.ones_like(probs) / len(probs)
        probs = np.clip(probs, self.eps, 1.0)
        return np.log(probs)


class MapMatchingHMM:
    """
    Encapsulates the complete Hidden Markov Model for sequential map-matching.
    """

    def __init__(
        self,
        graph: RoadGraph,
        transition_model: Optional[TransitionModel] = None,
        emission_model: Optional[EmissionModel] = None
    ):
        self.graph = graph
        self.transition_model = transition_model or TransitionModel()
        self.emission_model = emission_model or EmissionModel()

