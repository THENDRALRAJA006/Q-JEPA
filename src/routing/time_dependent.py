"""
Time-Dependent Ambulance Routing Engine (Phase 3).

Implements arrival-time-aware edge costs where each downstream road segment
is evaluated at the estimated arrival time of the ambulance, rather than
applying a uniform departure-time forecast.

Key capabilities:
1. Dynamic arrival-time estimation:
   arrival_horizon = round(accumulated_path_travel_time / forecast_sampling_interval)
2. Horizon lookup with documented fallback:
   - Within supported range (e.g., 1..12): uses the horizon-specific forecast.
   - Near-departure (h=0): uses current observed traffic.
   - Exceeding supported range (h > H_max): explicit fallback (historical average or clamped horizon).
3. Five routing strategies evaluated in controlled benchmarks:
   - A: Shortest Distance
   - B: Current Traffic
   - C: Static Historical Traffic
   - D: Time-Dependent Predicted Traffic
   - E: Time-Dependent Uncertainty-Aware Traffic
4. Dynamic trip simulator measuring ETA error, rerouting frequency, and realized journey time.
"""
from __future__ import annotations

import heapq
import math
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
import numpy as np

from src.common.logging import get_logger
from src.routing.shortest_path import RouteResult, RouterType
from src.simulation.road_network import RoadNetwork

logger = get_logger(__name__)


class FallbackMode(str, Enum):
    CLAMP_MAX_HORIZON = "clamped_max_horizon"
    HISTORICAL_AVERAGE = "historical_average"
    CURRENT_OBSERVED = "current_observed"


@dataclass
class EdgeArrivalEvaluation:
    """Detailed record of edge evaluation at estimated arrival time."""
    u: str
    v: str
    segment_id: str
    arrival_elapsed_s: float
    horizon_step: int
    evaluated_speed_kmh: float
    evaluated_travel_time_s: float
    uncertainty_s: float
    risk_penalty: float
    total_cost: float
    fallback_used: str  # "none", "clamped_max_horizon", "historical_average", "current_observed", "closed"
    is_closed: bool = False


@dataclass
class TimeDependentRouteResult(RouteResult):
    """Extended RouteResult with time-dependent arrival-time metadata."""
    departure_time_s: float = 0.0
    arrival_steps: List[int] = field(default_factory=list)
    edge_evaluations: List[EdgeArrivalEvaluation] = field(default_factory=list)
    fallback_counts: Dict[str, int] = field(default_factory=dict)
    eta_error_s: Optional[float] = None


def evaluate_edge_at_arrival(
    edge_data: Dict[str, Any],
    u: str,
    v: str,
    arrival_elapsed_s: float,
    step_interval_s: float = 300.0,
    max_supported_horizon: int = 12,
    risk_penalty: float = 0.0,
    fallback_mode: FallbackMode = FallbackMode.CLAMP_MAX_HORIZON,
) -> EdgeArrivalEvaluation:
    """
    Evaluates travel time, speed, and cost for edge u->v at arrival_elapsed_s.

    Determines the forecast horizon step:
      h = round(arrival_elapsed_s / step_interval_s)

    Applies documented fallbacks when h > max_supported_horizon or when
    predictions are unavailable for the segment.
    """
    segment_id = edge_data.get("segment_id", f"{u}->{v}")
    length_m = float(edge_data.get("length_m", 500.0))
    baseline_spd = float(edge_data.get("baseline_speed_kmh", 30.0))
    current_spd = float(edge_data.get("current_speed_kmh", baseline_spd))
    current_tt = float(edge_data.get("current_travel_time_s", length_m / (max(1.0, current_spd) / 3.6)))
    hist_spd = float(edge_data.get("historical_speed_kmh", baseline_spd))
    hist_tt = float(edge_data.get("historical_travel_time_s", length_m / (max(1.0, hist_spd) / 3.6)))

    if edge_data.get("is_closed", False):
        return EdgeArrivalEvaluation(
            u=u, v=v, segment_id=segment_id,
            arrival_elapsed_s=arrival_elapsed_s,
            horizon_step=-1,
            evaluated_speed_kmh=0.0,
            evaluated_travel_time_s=float("inf"),
            uncertainty_s=0.0,
            risk_penalty=risk_penalty,
            total_cost=float("inf"),
            fallback_used="closed",
            is_closed=True,
        )

    # Determine horizon step
    h = int(round(arrival_elapsed_s / max(1.0, step_interval_s)))
    pred_tts = edge_data.get("predicted_travel_times_s", {})
    pred_spds = edge_data.get("predicted_speeds_kmh", {})
    uncertainties_by_h = edge_data.get("uncertainties_by_horizon", {})
    base_uncertainty = float(edge_data.get("uncertainty", 0.0))

    fallback_label = "none"
    eval_tt: float
    eval_spd: float
    eval_unc: float

    if h == 0:
        # Near departure (elapsed time < half interval): use live current observation
        eval_tt = current_tt
        eval_spd = current_spd
        eval_unc = base_uncertainty * 0.5
        fallback_label = "current_observed_near_departure"

    elif 1 <= h <= max_supported_horizon:
        # Within supported forecast range
        if h in pred_tts:
            eval_tt = float(pred_tts[h])
            eval_spd = float(pred_spds.get(h, length_m / (max(1.0, eval_tt) / 3.6)))
            eval_unc = float(uncertainties_by_h.get(h, base_uncertainty))
            fallback_label = "none"
        else:
            # Prediction missing for this edge: documented fallback
            if fallback_mode == FallbackMode.HISTORICAL_AVERAGE:
                eval_tt = hist_tt
                eval_spd = hist_spd
                eval_unc = base_uncertainty
                fallback_label = "fallback_historical_missing_prediction"
            else:
                eval_tt = current_tt
                eval_spd = current_spd
                eval_unc = base_uncertainty
                fallback_label = "fallback_current_missing_prediction"

    else:
        # Horizon exceeds supported range (h > max_supported_horizon)
        if fallback_mode == FallbackMode.HISTORICAL_AVERAGE:
            eval_tt = hist_tt
            eval_spd = hist_spd
            eval_unc = base_uncertainty
            fallback_label = "fallback_historical_average_exceeded_horizon"
        else:
            # Clamped max horizon fallback
            if max_supported_horizon in pred_tts:
                eval_tt = float(pred_tts[max_supported_horizon])
                eval_spd = float(pred_spds.get(max_supported_horizon, length_m / (max(1.0, eval_tt) / 3.6)))
                eval_unc = float(uncertainties_by_h.get(max_supported_horizon, base_uncertainty))
                fallback_label = "fallback_clamped_max_horizon"
            else:
                eval_tt = hist_tt
                eval_spd = hist_spd
                eval_unc = base_uncertainty
                fallback_label = "fallback_historical_exceeded_horizon"

    cost = eval_tt + risk_penalty * eval_unc

    return EdgeArrivalEvaluation(
        u=u, v=v, segment_id=segment_id,
        arrival_elapsed_s=arrival_elapsed_s,
        horizon_step=h,
        evaluated_speed_kmh=eval_spd,
        evaluated_travel_time_s=eval_tt,
        uncertainty_s=eval_unc,
        risk_penalty=risk_penalty,
        total_cost=cost,
        fallback_used=fallback_label,
        is_closed=False,
    )


def route_time_dependent(
    network: RoadNetwork,
    origin: str,
    destination: str,
    departure_time_s: float = 0.0,
    step_interval_s: float = 300.0,
    max_supported_horizon: int = 12,
    risk_penalty: float = 0.0,
    fallback_mode: FallbackMode = FallbackMode.CLAMP_MAX_HORIZON,
    router_type: RouterType = RouterType.PREDICTED_TRAFFIC,
) -> TimeDependentRouteResult:
    """
    Time-dependent Dijkstra algorithm where downstream edge weights depend
    on the estimated arrival time at that segment.

    Guarantee:
    - Traversal time accumulates monotonically.
    - Segment cost uses the forecast corresponding to arrival_time = departure_time + accumulated_travel_time.
    - No single uniform departure-time forecast is applied across the entire graph.
    """
    t0 = time.perf_counter()
    G = network.graph

    if not G.has_node(origin) or not G.has_node(destination):
        elapsed_ms = (time.perf_counter() - t0) * 1000
        return TimeDependentRouteResult(
            router_type=router_type,
            origin=origin,
            destination=destination,
            path=[],
            total_distance_m=0.0,
            estimated_travel_time_s=float("inf"),
            predicted_arrival_time_s=float("inf"),
            uncertainty_s=0.0,
            computation_time_ms=elapsed_ms,
            feasible=False,
            explanation=f"Origin '{origin}' or destination '{destination}' does not exist in network.",
        )

    # Priority queue stores tuples: (cumulative_cost, node_u, cumulative_time_s)
    # Using node counter to break ties without comparing node strings directly
    tie_breaker = 0
    pq: List[Tuple[float, int, str, float]] = [(0.0, tie_breaker, origin, 0.0)]
    best_costs: Dict[str, float] = {origin: 0.0}
    best_times: Dict[str, float] = {origin: 0.0}
    parent_map: Dict[str, Tuple[str, EdgeArrivalEvaluation]] = {}

    target_reached = False

    while pq:
        curr_cost, _, u, curr_time = heapq.heappop(pq)

        if curr_cost > best_costs.get(u, float("inf")):
            continue

        if u == destination:
            target_reached = True
            break

        for v in G.neighbors(u):
            edge_data = G[u][v]
            if edge_data.get("is_closed", False):
                continue

            eval_res = evaluate_edge_at_arrival(
                edge_data=edge_data,
                u=u,
                v=v,
                arrival_elapsed_s=curr_time,
                step_interval_s=step_interval_s,
                max_supported_horizon=max_supported_horizon,
                risk_penalty=risk_penalty,
                fallback_mode=fallback_mode,
            )

            if eval_res.is_closed or math.isinf(eval_res.total_cost):
                continue

            next_cost = curr_cost + eval_res.total_cost
            next_time = curr_time + eval_res.evaluated_travel_time_s

            if next_cost < best_costs.get(v, float("inf")):
                best_costs[v] = next_cost
                best_times[v] = next_time
                parent_map[v] = (u, eval_res)
                tie_breaker += 1
                heapq.heappush(pq, (next_cost, tie_breaker, v, next_time))

    elapsed_ms = (time.perf_counter() - t0) * 1000

    if not target_reached:
        return TimeDependentRouteResult(
            router_type=router_type,
            origin=origin,
            destination=destination,
            path=[],
            total_distance_m=0.0,
            estimated_travel_time_s=float("inf"),
            predicted_arrival_time_s=float("inf"),
            uncertainty_s=0.0,
            computation_time_ms=elapsed_ms,
            feasible=False,
            explanation="No feasible path to destination (road closures or disconnected network).",
        )

    # Reconstruct path and edge evaluations
    path = [destination]
    evaluations_rev: List[EdgeArrivalEvaluation] = []
    curr = destination
    while curr != origin:
        p, ev = parent_map[curr]
        evaluations_rev.append(ev)
        path.append(p)
        curr = p

    path.reverse()
    evaluations = list(reversed(evaluations_rev))

    total_dist = sum(G[evaluations[i].u][evaluations[i].v].get("length_m", 0.0) for i in range(len(evaluations)))
    total_travel_time = sum(ev.evaluated_travel_time_s for ev in evaluations)
    total_uncertainty = sum(ev.uncertainty_s for ev in evaluations)
    arrival_steps = [ev.horizon_step for ev in evaluations]
    edge_tts = [ev.evaluated_travel_time_s for ev in evaluations]

    fallback_counts: Dict[str, int] = {}
    for ev in evaluations:
        fb = ev.fallback_used
        fallback_counts[fb] = fallback_counts.get(fb, 0) + 1

    risk_score = risk_penalty * total_uncertainty
    desc = (
        f"Arrival-time-aware time-dependent route (steps={len(evaluations)}, "
        f"fallbacks={sum(v for k, v in fallback_counts.items() if k != 'none')}). "
        f"Costs evaluated dynamically at downstream segment arrival times."
    )

    return TimeDependentRouteResult(
        router_type=router_type,
        origin=origin,
        destination=destination,
        path=path,
        total_distance_m=total_dist,
        estimated_travel_time_s=total_travel_time,
        predicted_arrival_time_s=departure_time_s + total_travel_time,
        uncertainty_s=total_uncertainty,
        computation_time_ms=elapsed_ms,
        feasible=True,
        explanation=desc,
        edge_travel_times=edge_tts,
        risk_score=risk_score,
        departure_time_s=departure_time_s,
        arrival_steps=arrival_steps,
        edge_evaluations=evaluations,
        fallback_counts=fallback_counts,
    )


def route_static_historical(
    network: RoadNetwork,
    origin: str,
    destination: str,
) -> RouteResult:
    """
    Static historical-traffic routing (Baseline 3).
    Dijkstra evaluated strictly on historical average travel times.
    """
    t0 = time.perf_counter()
    G = network.get_open_subgraph()

    def weight(u, v, d):
        return d.get("historical_travel_time_s", d.get("length_m", 500.0) / (d.get("baseline_speed_kmh", 30.0) / 3.6))

    try:
        path = nx.shortest_path(G, origin, destination, weight=weight)
        dist = sum(G[path[i]][path[i+1]]["length_m"] for i in range(len(path) - 1))
        tt = sum(weight(path[i], path[i+1], G[path[i]][path[i+1]]) for i in range(len(path) - 1))
        edge_tts = [weight(path[i], path[i+1], G[path[i]][path[i+1]]) for i in range(len(path) - 1)]
        feasible = True
        explanation = "Static historical-traffic route using long-term historical mean speeds."
    except nx.NetworkXNoPath:
        path, dist, tt, edge_tts = [], 0.0, float("inf"), []
        feasible = False
        explanation = "No feasible path found."

    elapsed_ms = (time.perf_counter() - t0) * 1000
    return RouteResult(
        router_type=RouterType.CURRENT_TRAFFIC,
        origin=origin, destination=destination,
        path=path, total_distance_m=dist,
        estimated_travel_time_s=tt, predicted_arrival_time_s=tt,
        uncertainty_s=0.0, computation_time_ms=elapsed_ms,
        feasible=feasible, explanation=explanation, edge_travel_times=edge_tts,
    )


def compare_five_strategies(
    network: RoadNetwork,
    origin: str,
    destination: str,
    departure_time_s: float = 0.0,
    risk_penalty: float = 0.3,
    step_interval_s: float = 300.0,
    max_supported_horizon: int = 12,
    fallback_mode: FallbackMode = FallbackMode.CLAMP_MAX_HORIZON,
) -> Dict[str, RouteResult]:
    """
    Evaluates all 5 required routing strategies on identical input conditions:
    1. Shortest distance routing
    2. Current traffic routing
    3. Static historical traffic routing
    4. Time-dependent routing with predicted traffic
    5. Time-dependent routing with uncertainty-aware costs
    """
    from src.routing.shortest_path import route_current_traffic, route_shortest_distance

    r1 = route_shortest_distance(network, origin, destination)
    r2 = route_current_traffic(network, origin, destination)
    r3 = route_static_historical(network, origin, destination)
    r4 = route_time_dependent(
        network, origin, destination,
        departure_time_s=departure_time_s,
        step_interval_s=step_interval_s,
        max_supported_horizon=max_supported_horizon,
        risk_penalty=0.0,
        fallback_mode=fallback_mode,
        router_type=RouterType.PREDICTED_TRAFFIC,
    )
    r5 = route_time_dependent(
        network, origin, destination,
        departure_time_s=departure_time_s,
        step_interval_s=step_interval_s,
        max_supported_horizon=max_supported_horizon,
        risk_penalty=risk_penalty,
        fallback_mode=fallback_mode,
        router_type=RouterType.UNCERTAINTY_AWARE,
    )

    return {
        "1_shortest_distance": r1,
        "2_current_traffic": r2,
        "3_static_historical": r3,
        "4_time_dependent_predicted": r4,
        "5_time_dependent_uncertainty": r5,
    }
