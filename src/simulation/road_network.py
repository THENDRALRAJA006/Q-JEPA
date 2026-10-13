"""
Road-network representation for Surakshit Bharat Q-JEPA.
Uses NetworkX directed graph with rich edge/node attributes.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np

from src.common.logging import get_logger

logger = get_logger(__name__)


@dataclass
class EdgeState:
    """Current and predicted state of a road segment."""
    segment_id: str
    length_m: float
    baseline_speed_kmh: float
    current_speed_kmh: float
    current_travel_time_s: float
    predicted_travel_times_s: Dict[int, float] = field(default_factory=dict)  # horizon_steps -> seconds
    congestion_level: float = 0.0          # 0 = free, 1 = standstill
    has_incident: bool = False
    is_closed: bool = False
    uncertainty: float = 0.0              # std-dev of travel-time prediction (seconds)
    last_updated: Optional[float] = None  # unix timestamp or sim step


@dataclass
class NodeState:
    """Intersection attributes."""
    node_id: str
    lat: float
    lon: float
    has_traffic_signal: bool = False
    signal_delay_s: float = 0.0


class RoadNetwork:
    """Directed road-graph with per-edge traffic state."""

    def __init__(self, seed: int = 42):
        self.graph = nx.DiGraph()
        self.seed = seed
        self._rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------

    @classmethod
    def create_synthetic_grid(
        cls,
        rows: int = 5,
        cols: int = 6,
        seed: int = 42,
        bbox: Tuple[float, float, float, float] = (28.50, 77.10, 28.65, 77.30),
    ) -> "RoadNetwork":
        """Create a grid road network with lat/lon positions."""
        net = cls(seed=seed)
        lat_min, lon_min, lat_max, lon_max = bbox
        rng = np.random.default_rng(seed)

        # Build nodes
        for r in range(rows):
            for c in range(cols):
                nid = f"N{r}_{c}"
                lat = lat_min + (lat_max - lat_min) * r / max(rows - 1, 1)
                lon = lon_min + (lon_max - lon_min) * c / max(cols - 1, 1)
                # Add small jitter for realism
                lat += rng.normal(0, 0.002)
                lon += rng.normal(0, 0.002)
                net.graph.add_node(
                    nid,
                    lat=float(lat),
                    lon=float(lon),
                    has_traffic_signal=bool(rng.random() > 0.6),
                    signal_delay_s=float(rng.uniform(5, 30)),
                )

        # Build edges (both directions for each adjacent pair)
        def _add_edge(u: str, v: str, speed: float, length: float):
            seg_id = f"{u}->{v}"
            t = length / (speed / 3.6)
            net.graph.add_edge(
                u, v,
                segment_id=seg_id,
                length_m=length,
                baseline_speed_kmh=speed,
                current_speed_kmh=speed,
                current_travel_time_s=t,
                expected_travel_time_s=t,
                predicted_travel_times_s={},
                predicted_speeds_kmh={},
                congestion_level=0.0,
                has_incident=False,
                is_closed=False,
                uncertainty=0.0,
                uncertainties_by_horizon={},
                historical_speed_kmh=speed,
                historical_travel_time_s=t,
                last_updated=0,
            )

        for r in range(rows):
            for c in range(cols):
                nid = f"N{r}_{c}"
                if c + 1 < cols:
                    nbr = f"N{r}_{c+1}"
                    lat1 = net.graph.nodes[nid]["lat"]
                    lon1 = net.graph.nodes[nid]["lon"]
                    lat2 = net.graph.nodes[nbr]["lat"]
                    lon2 = net.graph.nodes[nbr]["lon"]
                    length = _haversine(lat1, lon1, lat2, lon2)
                    speed = float(rng.uniform(30, 60))
                    _add_edge(nid, nbr, speed, length)
                    _add_edge(nbr, nid, speed, length)
                if r + 1 < rows:
                    nbr = f"N{r+1}_{c}"
                    lat1 = net.graph.nodes[nid]["lat"]
                    lon1 = net.graph.nodes[nid]["lon"]
                    lat2 = net.graph.nodes[nbr]["lat"]
                    lon2 = net.graph.nodes[nbr]["lon"]
                    length = _haversine(lat1, lon1, lat2, lon2)
                    speed = float(rng.uniform(30, 60))
                    _add_edge(nid, nbr, speed, length)
                    _add_edge(nbr, nid, speed, length)

        logger.info(
            "Created synthetic grid network: %d nodes, %d edges",
            net.graph.number_of_nodes(),
            net.graph.number_of_edges(),
        )
        return net

    # ------------------------------------------------------------------
    # State updates
    # ------------------------------------------------------------------

    def update_edge_state(
        self,
        u: str,
        v: str,
        current_speed_kmh: Optional[float] = None,
        has_incident: Optional[bool] = None,
        is_closed: Optional[bool] = None,
        predicted_travel_times_s: Optional[Dict[int, float]] = None,
        predicted_speeds_kmh: Optional[Dict[int, float]] = None,
        uncertainty: Optional[float] = None,
        uncertainties_by_horizon: Optional[Dict[int, float]] = None,
        historical_speed_kmh: Optional[float] = None,
        historical_travel_time_s: Optional[float] = None,
        step: int = 0,
    ) -> None:
        if not self.graph.has_edge(u, v):
            logger.warning("Edge %s->%s not found", u, v)
            return
        e = self.graph[u][v]
        if current_speed_kmh is not None:
            e["current_speed_kmh"] = max(1.0, current_speed_kmh)
            e["current_travel_time_s"] = e["length_m"] / (e["current_speed_kmh"] / 3.6)
            e["expected_travel_time_s"] = e["current_travel_time_s"]
            e["congestion_level"] = max(
                0.0,
                min(1.0, 1.0 - e["current_speed_kmh"] / e["baseline_speed_kmh"]),
            )
        if has_incident is not None:
            e["has_incident"] = has_incident
        if is_closed is not None:
            e["is_closed"] = is_closed
        if predicted_travel_times_s is not None:
            e["predicted_travel_times_s"] = predicted_travel_times_s
        if predicted_speeds_kmh is not None:
            e["predicted_speeds_kmh"] = predicted_speeds_kmh
        if uncertainty is not None:
            e["uncertainty"] = uncertainty
        if uncertainties_by_horizon is not None:
            e["uncertainties_by_horizon"] = uncertainties_by_horizon
        if historical_speed_kmh is not None:
            e["historical_speed_kmh"] = historical_speed_kmh
            e["historical_travel_time_s"] = e["length_m"] / (max(1.0, historical_speed_kmh) / 3.6)
        elif historical_travel_time_s is not None:
            e["historical_travel_time_s"] = historical_travel_time_s
        e["last_updated"] = step

    def close_edge(self, u: str, v: str, step: int = 0) -> None:
        self.update_edge_state(u, v, is_closed=True, step=step)

    def reopen_edge(self, u: str, v: str, step: int = 0) -> None:
        self.update_edge_state(u, v, is_closed=False, has_incident=False, step=step)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_edge_travel_time(self, u: str, v: str, horizon: int = 0) -> float:
        """Travel time in seconds. Use predicted if available."""
        e = self.graph[u][v]
        if e["is_closed"]:
            return float("inf")
        if horizon > 0 and e.get("predicted_travel_times_s"):
            return e["predicted_travel_times_s"].get(horizon, e["current_travel_time_s"])
        return e["current_travel_time_s"]

    def get_open_subgraph(self) -> nx.DiGraph:
        """Return subgraph with only open edges."""
        edges_to_keep = [
            (u, v) for u, v, d in self.graph.edges(data=True) if not d.get("is_closed", False)
        ]
        return self.graph.edge_subgraph(edges_to_keep).copy()

    def get_node_positions(self) -> Dict[str, Tuple[float, float]]:
        """lat/lon for each node."""
        return {
            n: (d["lat"], d["lon"])
            for n, d in self.graph.nodes(data=True)
        }

    def node_list(self) -> List[str]:
        return list(self.graph.nodes())

    def edge_list(self) -> List[Tuple[str, str]]:
        return list(self.graph.edges())

    def summary(self) -> Dict:
        return {
            "num_nodes": self.graph.number_of_nodes(),
            "num_edges": self.graph.number_of_edges(),
            "closed_edges": sum(
                1 for _, _, d in self.graph.edges(data=True) if d.get("is_closed")
            ),
            "incident_edges": sum(
                1 for _, _, d in self.graph.edges(data=True) if d.get("has_incident")
            ),
            "avg_congestion": float(
                np.mean([d.get("congestion_level", 0) for _, _, d in self.graph.edges(data=True)])
            ) if self.graph.number_of_edges() else 0.0,
        }


def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Distance in metres between two lat/lon points."""
    R = 6_371_000.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
