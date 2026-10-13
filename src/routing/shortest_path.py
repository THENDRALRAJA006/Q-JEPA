"""
Routing algorithms for Surakshit Bharat Q-JEPA.
Implements five router strategies + dynamic rerouting.
"""
from __future__ import annotations

import heapq
import math
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np

from src.simulation.road_network import RoadNetwork
from src.common.logging import get_logger

logger = get_logger(__name__)


class RouterType(str, Enum):
    SHORTEST_DISTANCE = "shortest_distance"
    CURRENT_TRAFFIC = "current_traffic"
    PREDICTED_TRAFFIC = "predicted_traffic"
    UNCERTAINTY_AWARE = "uncertainty_aware"
    QUANTUM_INSPIRED = "quantum_inspired"
    STATIC_HISTORICAL = "static_historical"
    TIME_DEPENDENT_PREDICTED = "time_dependent_predicted"
    TIME_DEPENDENT_UNCERTAINTY = "time_dependent_uncertainty"


@dataclass
class RouteResult:
    router_type: RouterType
    origin: str
    destination: str
    path: List[str]
    total_distance_m: float
    estimated_travel_time_s: float
    predicted_arrival_time_s: float
    uncertainty_s: float
    computation_time_ms: float
    feasible: bool
    explanation: str
    edge_travel_times: List[float] = field(default_factory=list)
    risk_score: float = 0.0


# ─────────────────────────────────────────────────
# Router A: Shortest distance
# ─────────────────────────────────────────────────

def route_shortest_distance(
    network: RoadNetwork, origin: str, destination: str
) -> RouteResult:
    """Dijkstra on raw edge lengths. Ignores traffic."""
    t0 = time.perf_counter()
    G = network.get_open_subgraph()

    def w(u, v, d):
        return d.get("length_m", 1.0)

    try:
        path = nx.shortest_path(G, origin, destination, weight=w)
        dist = sum(
            G[path[i]][path[i+1]]["length_m"] for i in range(len(path) - 1)
        )
        tt = sum(
            G[path[i]][path[i+1]]["current_travel_time_s"] for i in range(len(path) - 1)
        )
        edge_tts = [G[path[i]][path[i+1]]["current_travel_time_s"] for i in range(len(path) - 1)]
        feasible = True
        explanation = "Shortest geographic distance route; ignores live traffic."
    except nx.NetworkXNoPath:
        path, dist, tt, edge_tts = [], 0.0, float("inf"), []
        feasible = False
        explanation = "No feasible path found (network disconnected or roads closed)."

    elapsed_ms = (time.perf_counter() - t0) * 1000
    return RouteResult(
        router_type=RouterType.SHORTEST_DISTANCE,
        origin=origin, destination=destination,
        path=path, total_distance_m=dist,
        estimated_travel_time_s=tt, predicted_arrival_time_s=tt,
        uncertainty_s=0.0, computation_time_ms=elapsed_ms,
        feasible=feasible, explanation=explanation, edge_travel_times=edge_tts,
    )


# ─────────────────────────────────────────────────
# Router B: Current-traffic routing
# ─────────────────────────────────────────────────

def route_current_traffic(
    network: RoadNetwork, origin: str, destination: str
) -> RouteResult:
    """Dijkstra using current travel-time estimates."""
    t0 = time.perf_counter()
    G = network.get_open_subgraph()

    def w(u, v, d):
        return d.get("current_travel_time_s", d.get("length_m", 1.0))

    try:
        path = nx.shortest_path(G, origin, destination, weight=w)
        dist = sum(G[path[i]][path[i+1]]["length_m"] for i in range(len(path) - 1))
        tt = sum(G[path[i]][path[i+1]]["current_travel_time_s"] for i in range(len(path) - 1))
        edge_tts = [G[path[i]][path[i+1]]["current_travel_time_s"] for i in range(len(path) - 1)]
        feasible = True
        explanation = "Current-traffic route using live observed travel times."
    except nx.NetworkXNoPath:
        path, dist, tt, edge_tts = [], 0.0, float("inf"), []
        feasible = False
        explanation = "No feasible path (road closures or disconnected network)."

    elapsed_ms = (time.perf_counter() - t0) * 1000
    return RouteResult(
        router_type=RouterType.CURRENT_TRAFFIC,
        origin=origin, destination=destination,
        path=path, total_distance_m=dist,
        estimated_travel_time_s=tt, predicted_arrival_time_s=tt,
        uncertainty_s=0.0, computation_time_ms=elapsed_ms,
        feasible=feasible, explanation=explanation, edge_travel_times=edge_tts,
    )


# ─────────────────────────────────────────────────
# Router C: Predicted traffic routing
# ─────────────────────────────────────────────────

def route_predicted_traffic(
    network: RoadNetwork, origin: str, destination: str,
    horizon_steps: int = 3,
) -> RouteResult:
    """Dijkstra using forecasted travel times. Falls back to current if unavailable."""
    t0 = time.perf_counter()
    G = network.get_open_subgraph()

    def w(u, v, d):
        pred = d.get("predicted_travel_times_s", {})
        if pred and horizon_steps in pred:
            return pred[horizon_steps]
        return d.get("current_travel_time_s", d.get("length_m", 1.0))

    try:
        path = nx.shortest_path(G, origin, destination, weight=w)
        dist = sum(G[path[i]][path[i+1]]["length_m"] for i in range(len(path) - 1))
        tt = sum(w(*e, G.edges[e]) for e in zip(path, path[1:]))
        edge_tts = [w(*e, G.edges[e]) for e in zip(path, path[1:])]
        feasible = True
        explanation = f"Predicted-traffic route (horizon={horizon_steps} steps). Falls back to current if no forecast."
    except nx.NetworkXNoPath:
        path, dist, tt, edge_tts = [], 0.0, float("inf"), []
        feasible = False
        explanation = "No feasible path found."

    elapsed_ms = (time.perf_counter() - t0) * 1000
    return RouteResult(
        router_type=RouterType.PREDICTED_TRAFFIC,
        origin=origin, destination=destination,
        path=path, total_distance_m=dist,
        estimated_travel_time_s=tt, predicted_arrival_time_s=tt,
        uncertainty_s=0.0, computation_time_ms=elapsed_ms,
        feasible=feasible, explanation=explanation, edge_travel_times=edge_tts,
    )


# ─────────────────────────────────────────────────
# Router D: Uncertainty-aware routing
# ─────────────────────────────────────────────────

def route_uncertainty_aware(
    network: RoadNetwork, origin: str, destination: str,
    risk_penalty: float = 0.3, horizon_steps: int = 3,
) -> RouteResult:
    """
    Dijkstra with cost = predicted_travel_time + risk_penalty * uncertainty.
    Biases routes toward lower-variance paths.
    """
    t0 = time.perf_counter()
    G = network.get_open_subgraph()

    def w(u, v, d):
        pred = d.get("predicted_travel_times_s", {})
        if pred and horizon_steps in pred:
            tt = pred[horizon_steps]
        else:
            tt = d.get("current_travel_time_s", d.get("length_m", 1.0))
        sigma = d.get("uncertainty", 0.0)
        return tt + risk_penalty * sigma

    try:
        path = nx.shortest_path(G, origin, destination, weight=w)
        dist = sum(G[path[i]][path[i+1]]["length_m"] for i in range(len(path) - 1))
        tt = sum(G[path[i]][path[i+1]].get("current_travel_time_s", 0) for i in range(len(path) - 1))
        uncertainty = sum(G[path[i]][path[i+1]].get("uncertainty", 0) for i in range(len(path) - 1))
        edge_tts = [G[path[i]][path[i+1]].get("current_travel_time_s", 0) for i in range(len(path) - 1)]
        feasible = True
        explanation = f"Uncertainty-aware route (risk_penalty={risk_penalty}). Avoids high-variance segments."
    except nx.NetworkXNoPath:
        path, dist, tt, uncertainty, edge_tts = [], 0.0, float("inf"), 0.0, []
        feasible = False
        explanation = "No feasible path found."

    elapsed_ms = (time.perf_counter() - t0) * 1000
    return RouteResult(
        router_type=RouterType.UNCERTAINTY_AWARE,
        origin=origin, destination=destination,
        path=path, total_distance_m=dist,
        estimated_travel_time_s=tt, predicted_arrival_time_s=tt,
        uncertainty_s=uncertainty, computation_time_ms=elapsed_ms,
        feasible=feasible, explanation=explanation, edge_travel_times=edge_tts,
        risk_score=risk_penalty * uncertainty,
    )


# ─────────────────────────────────────────────────
# Router E: Quantum-inspired (Simulated Annealing)
# ─────────────────────────────────────────────────

def route_quantum_inspired(
    network: RoadNetwork, origin: str, destination: str,
    temperature: float = 100.0,
    cooling: float = 0.98,
    iterations: int = 500,
    seed: int = 42,
) -> RouteResult:
    """
    Simulated-annealing route optimizer.

    This is a CLASSICAL simulated-annealing implementation used as a
    quantum-inspired optimization baseline. It does NOT claim quantum
    speedup. For small networks, results are comparable to Dijkstra.
    """
    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)
    G = network.get_open_subgraph()

    # Start with a greedy current-traffic route
    try:
        current_path = nx.shortest_path(
            G, origin, destination, weight="current_travel_time_s"
        )
    except nx.NetworkXNoPath:
        elapsed_ms = (time.perf_counter() - t0) * 1000
        return RouteResult(
            router_type=RouterType.QUANTUM_INSPIRED,
            origin=origin, destination=destination,
            path=[], total_distance_m=0.0,
            estimated_travel_time_s=float("inf"), predicted_arrival_time_s=float("inf"),
            uncertainty_s=0.0, computation_time_ms=elapsed_ms,
            feasible=False, explanation="No initial path found for SA optimizer.",
        )

    def path_cost(p: List[str]) -> float:
        if len(p) < 2:
            return float("inf")
        cost = 0.0
        for i in range(len(p) - 1):
            if not G.has_edge(p[i], p[i+1]):
                return float("inf")
            d = G[p[i]][p[i+1]]
            pred = d.get("predicted_travel_times_s", {})
            tt = pred.get(3, d.get("current_travel_time_s", 0.0))
            sigma = d.get("uncertainty", 0.0)
            cost += tt + 0.2 * sigma
        return cost

    best_path = list(current_path)
    best_cost = path_cost(best_path)
    current_cost = best_cost
    T = temperature

    for _ in range(iterations):
        # Perturbation: try a random detour through a random intermediate node
        try:
            mid = str(rng.choice(list(G.nodes())))
            p1 = nx.shortest_path(G, origin, mid, weight="current_travel_time_s")
            p2 = nx.shortest_path(G, mid, destination, weight="current_travel_time_s")
            candidate = p1[:-1] + p2
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            candidate = list(current_path)

        candidate_cost = path_cost(candidate)
        delta = candidate_cost - current_cost

        if delta < 0 or (T > 0 and float(rng.random()) < math.exp(-delta / T)):
            current_path = candidate
            current_cost = candidate_cost
            if current_cost < best_cost:
                best_path = list(current_path)
                best_cost = current_cost

        T *= cooling

    dist = sum(G[best_path[i]][best_path[i+1]]["length_m"] for i in range(len(best_path) - 1) if G.has_edge(best_path[i], best_path[i+1]))
    tt = sum(G[best_path[i]][best_path[i+1]].get("current_travel_time_s", 0) for i in range(len(best_path) - 1) if G.has_edge(best_path[i], best_path[i+1]))
    edge_tts = [G[best_path[i]][best_path[i+1]].get("current_travel_time_s", 0) for i in range(len(best_path) - 1) if G.has_edge(best_path[i], best_path[i+1])]

    elapsed_ms = (time.perf_counter() - t0) * 1000
    return RouteResult(
        router_type=RouterType.QUANTUM_INSPIRED,
        origin=origin, destination=destination,
        path=best_path, total_distance_m=dist,
        estimated_travel_time_s=tt, predicted_arrival_time_s=tt,
        uncertainty_s=0.0, computation_time_ms=elapsed_ms,
        feasible=True,
        explanation=(
            f"Classical simulated-annealing optimizer ({iterations} iterations, "
            f"T0={temperature}, cooling={cooling}). Note: This is NOT a quantum algorithm."
        ),
        edge_travel_times=edge_tts,
    )


# ─────────────────────────────────────────────────
# Dynamic rerouter
# ─────────────────────────────────────────────────

class DynamicRouter:
    """Monitors an in-progress journey and triggers reroutes when warranted."""

    def __init__(
        self,
        network: RoadNetwork,
        router_type: RouterType = RouterType.CURRENT_TRAFFIC,
        reroute_threshold_pct: float = 10.0,
        min_reroute_interval_steps: int = 3,
        risk_penalty: float = 0.3,
    ):
        self.network = network
        self.router_type = router_type
        self.reroute_threshold_pct = reroute_threshold_pct
        self.min_reroute_interval_steps = min_reroute_interval_steps
        self.risk_penalty = risk_penalty
        self.current_route: Optional[RouteResult] = None
        self.reroute_count = 0
        self.last_reroute_step = -999

    def set_initial_route(self, origin: str, destination: str) -> RouteResult:
        route = self._compute_route(origin, destination)
        self.current_route = route
        return route

    def check_and_reroute(
        self, current_node: str, destination: str, step: int
    ) -> Optional[RouteResult]:
        """Returns a new route if rerouting is warranted, else None."""
        if self.current_route is None:
            return self.set_initial_route(current_node, destination)

        # Enforce minimum interval
        if step - self.last_reroute_step < self.min_reroute_interval_steps:
            return None

        # Compute expected remaining time on current route
        remaining = self._remaining_time(current_node)
        new_route = self._compute_route(current_node, destination)

        if not new_route.feasible:
            return None

        improvement_pct = 100.0 * (remaining - new_route.estimated_travel_time_s) / max(remaining, 1.0)

        if improvement_pct >= self.reroute_threshold_pct:
            self.current_route = new_route
            self.reroute_count += 1
            self.last_reroute_step = step
            logger.info("Rerouting triggered: %.1f%% improvement (step %d)", improvement_pct, step)
            return new_route

        return None

    def _remaining_time(self, current_node: str) -> float:
        if self.current_route is None or not self.current_route.path:
            return float("inf")
        route = self.current_route.path
        try:
            idx = route.index(current_node)
        except ValueError:
            return float("inf")
        remaining_path = route[idx:]
        total = 0.0
        G = self.network.graph
        for i in range(len(remaining_path) - 1):
            u, v = remaining_path[i], remaining_path[i + 1]
            if G.has_edge(u, v):
                total += G[u][v].get("current_travel_time_s", 0)
            else:
                return float("inf")
        return total

    def _compute_route(self, origin: str, destination: str) -> RouteResult:
        if self.router_type == RouterType.SHORTEST_DISTANCE:
            return route_shortest_distance(self.network, origin, destination)
        elif self.router_type == RouterType.TIME_DEPENDENT_PREDICTED:
            from src.routing.time_dependent import route_time_dependent
            return route_time_dependent(self.network, origin, destination, risk_penalty=0.0)
        elif self.router_type == RouterType.TIME_DEPENDENT_UNCERTAINTY:
            from src.routing.time_dependent import route_time_dependent
            return route_time_dependent(self.network, origin, destination, risk_penalty=self.risk_penalty)
        elif self.router_type == RouterType.STATIC_HISTORICAL:
            from src.routing.time_dependent import route_static_historical
            return route_static_historical(self.network, origin, destination)
        elif self.router_type == RouterType.PREDICTED_TRAFFIC:
            return route_predicted_traffic(self.network, origin, destination)
        elif self.router_type == RouterType.UNCERTAINTY_AWARE:
            return route_uncertainty_aware(self.network, origin, destination, self.risk_penalty)
        elif self.router_type == RouterType.QUANTUM_INSPIRED:
            return route_quantum_inspired(self.network, origin, destination)
        else:  # CURRENT_TRAFFIC default
            return route_current_traffic(self.network, origin, destination)


def compare_all_routes(
    network: RoadNetwork, origin: str, destination: str,
    risk_penalty: float = 0.3,
) -> List[RouteResult]:
    """Run all five routers and return results list."""
    results = [
        route_shortest_distance(network, origin, destination),
        route_current_traffic(network, origin, destination),
        route_predicted_traffic(network, origin, destination),
        route_uncertainty_aware(network, origin, destination, risk_penalty),
        route_quantum_inspired(network, origin, destination),
    ]
    return results
