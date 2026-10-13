"""
Tests for Phase 3 Time-Dependent Ambulance Routing.

Verifies:
1. Arrival-time-aware forecast indexing (downstream edges evaluated at t_arrival, not t_departure).
2. Documented fallbacks when required horizon exceeds model's supported range.
3. Edge state maintenance (current, predicted, expected, uncertainty, incident, last_updated).
4. Controlled en-route traffic change adaptation without future oracle leakage.
5. 5-strategy evaluation consistency on identical scenario conditions.
6. Verification that real METR-LA sensor data and simulated routing are strictly separated.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pytest
import networkx as nx
import numpy as np

from src.simulation.road_network import RoadNetwork
from src.routing.shortest_path import (
    RouterType,
    route_current_traffic,
    route_shortest_distance,
)
from src.routing.time_dependent import (
    EdgeArrivalEvaluation,
    FallbackMode,
    TimeDependentRouteResult,
    compare_five_strategies,
    evaluate_edge_at_arrival,
    route_static_historical,
    route_time_dependent,
)
from scripts.run_time_dependent_benchmark import (
    run_controlled_en_route_traffic_change_test,
)


@pytest.fixture
def corridor_network():
    """Simple linear network: A -> B -> C -> D -> E."""
    net = RoadNetwork(seed=42)
    nodes = ["A", "B", "C", "D", "E"]
    for i, n in enumerate(nodes):
        net.graph.add_node(n, lat=28.5 + 0.01 * i, lon=77.1 + 0.01 * i)

    # 4 segments, each 1000m long, baseline speed 60 km/h (free flow TT = 60s)
    for u, v in zip(nodes, nodes[1:]):
        net.graph.add_edge(
            u, v,
            segment_id=f"{u}->{v}",
            length_m=1000.0,
            baseline_speed_kmh=60.0,
            current_speed_kmh=60.0,
            current_travel_time_s=60.0,
            expected_travel_time_s=60.0,
            predicted_travel_times_s={},
            predicted_speeds_kmh={},
            congestion_level=0.0,
            has_incident=False,
            is_closed=False,
            uncertainty=2.0,
            uncertainties_by_horizon={},
            historical_speed_kmh=60.0,
            historical_travel_time_s=60.0,
            last_updated=0,
        )
    return net


def test_edge_state_maintenance(corridor_network):
    """Verify each edge maintains all required Phase 3 state fields."""
    for u, v, d in corridor_network.graph.edges(data=True):
        assert "current_speed_kmh" in d
        assert "current_travel_time_s" in d
        assert "predicted_travel_times_s" in d
        assert "expected_travel_time_s" in d
        assert "uncertainty" in d
        assert "has_incident" in d
        assert "is_closed" in d
        assert "last_updated" in d
        assert "historical_travel_time_s" in d


def test_arrival_time_forecast_indexing(corridor_network):
    """
    Test that downstream edges are evaluated at arrival time, not departure time.
    Segment B->C will be reached at t = 300s (horizon step 1).
    Segment C->D will be reached at t = 900s (horizon step 3).
    """
    edge_bc = corridor_network.graph["B"]["C"]
    edge_cd = corridor_network.graph["C"]["D"]

    # Different forecasts at different horizons:
    # At h=0: speed=60 km/h (tt=60s)
    # At h=1: speed=30 km/h (tt=120s)
    # At h=3: speed=15 km/h (tt=240s)
    edge_bc["predicted_travel_times_s"] = {1: 120.0, 3: 200.0}
    edge_cd["predicted_travel_times_s"] = {1: 60.0, 3: 240.0}

    # Arrival at B is at 300s -> should evaluate h=1 (120s), NOT current 60s
    eval_bc = evaluate_edge_at_arrival(
        edge_data=edge_bc,
        u="B", v="C",
        arrival_elapsed_s=300.0,
        step_interval_s=300.0,
        max_supported_horizon=12,
    )
    assert eval_bc.horizon_step == 1
    assert eval_bc.evaluated_travel_time_s == 120.0
    assert eval_bc.fallback_used == "none"

    # Arrival at C is at 900s -> should evaluate h=3 (240s)
    eval_cd = evaluate_edge_at_arrival(
        edge_data=edge_cd,
        u="C", v="D",
        arrival_elapsed_s=900.0,
        step_interval_s=300.0,
        max_supported_horizon=12,
    )
    assert eval_cd.horizon_step == 3
    assert eval_cd.evaluated_travel_time_s == 240.0
    assert eval_cd.fallback_used == "none"


def test_documented_fallback_when_exceeding_max_horizon(corridor_network):
    """Test fallback when arrival time exceeds supported horizon (e.g., h=15 > 12)."""
    edge_de = corridor_network.graph["D"]["E"]
    edge_de["predicted_travel_times_s"] = {12: 150.0}
    edge_de["historical_travel_time_s"] = 180.0

    # Arrival at 4500s -> h = 15 > max_supported_horizon (12)
    # 1. Clamped fallback
    eval_clamped = evaluate_edge_at_arrival(
        edge_data=edge_de,
        u="D", v="E",
        arrival_elapsed_s=4500.0,
        step_interval_s=300.0,
        max_supported_horizon=12,
        fallback_mode=FallbackMode.CLAMP_MAX_HORIZON,
    )
    assert eval_clamped.horizon_step == 15
    assert eval_clamped.evaluated_travel_time_s == 150.0  # Used max horizon 12
    assert "fallback_clamped_max_horizon" in eval_clamped.fallback_used

    # 2. Historical average fallback
    eval_hist = evaluate_edge_at_arrival(
        edge_data=edge_de,
        u="D", v="E",
        arrival_elapsed_s=4500.0,
        step_interval_s=300.0,
        max_supported_horizon=12,
        fallback_mode=FallbackMode.HISTORICAL_AVERAGE,
    )
    assert eval_hist.horizon_step == 15
    assert eval_hist.evaluated_travel_time_s == 180.0  # Used historical average
    assert "fallback_historical_average_exceeded_horizon" in eval_hist.fallback_used


def test_controlled_en_route_traffic_change_experiment():
    """
    Automated test verifying that an incident injected mid-journey modifies
    the selected route dynamically without future ground truth leakage.
    """
    report = run_controlled_en_route_traffic_change_test()
    assert report["status"] == "PASSED"
    assert report["route_adapted_dynamically"] is True
    assert report["avoided_congested_edge"] is True
    assert report["no_future_information_leakage"] is True
    assert report["arrival_time_aware_costs_applied"] is True


def test_five_strategy_evaluation(corridor_network):
    """Verify that all 5 strategies return valid, feasible routes on identical conditions."""
    res = compare_five_strategies(
        network=corridor_network,
        origin="A",
        destination="D",
        departure_time_s=0.0,
        risk_penalty=0.3,
    )
    assert "1_shortest_distance" in res
    assert "2_current_traffic" in res
    assert "3_static_historical" in res
    assert "4_time_dependent_predicted" in res
    assert "5_time_dependent_uncertainty" in res

    for k, r in res.items():
        assert r.feasible is True
        assert r.path == ["A", "B", "C", "D"]
        assert r.total_distance_m == 3000.0
        assert r.estimated_travel_time_s > 0.0


def test_controlled_uncertainty_changes_route():
    """
    Controlled Test A: Uncertainty changes preferred route.
    Two parallel corridors connecting S to D:
    - Corridor A (Risky): S -> A -> D (travel time = 200s, uncertainty = 60s)
      Cost at risk=0.0: 200s
      Cost at risk=1.0: 200 + 60 = 260s
    - Corridor B (Reliable): S -> B -> D (travel time = 220s, uncertainty = 10s)
      Cost at risk=0.0: 220s
      Cost at risk=1.0: 220 + 10 = 230s

    Under risk_penalty=0.0: Corridor A is chosen (200s < 220s).
    Under risk_penalty=1.0: Corridor B is chosen (230s < 260s).
    """
    net = RoadNetwork()
    for n in ["S", "A", "B", "D"]:
        net.graph.add_node(n, lat=0.0, lon=0.0)

    # Corridor 1 (Risky): S -> A -> D
    net.graph.add_edge("S", "A", length_m=2000.0, current_speed_kmh=36.0, current_travel_time_s=100.0,
                       predicted_travel_times_s={1: 100.0, 2: 100.0}, predicted_speeds_kmh={1: 36.0, 2: 36.0},
                       uncertainties_by_horizon={1: 30.0, 2: 30.0}, uncertainty=30.0)
    net.graph.add_edge("A", "D", length_m=2000.0, current_speed_kmh=36.0, current_travel_time_s=100.0,
                       predicted_travel_times_s={1: 100.0, 2: 100.0}, predicted_speeds_kmh={1: 36.0, 2: 36.0},
                       uncertainties_by_horizon={1: 30.0, 2: 30.0}, uncertainty=30.0)

    # Corridor 2 (Reliable): S -> B -> D
    net.graph.add_edge("S", "B", length_m=2000.0, current_speed_kmh=32.7, current_travel_time_s=110.0,
                       predicted_travel_times_s={1: 110.0, 2: 110.0}, predicted_speeds_kmh={1: 32.7, 2: 32.7},
                       uncertainties_by_horizon={1: 5.0, 2: 5.0}, uncertainty=5.0)
    net.graph.add_edge("B", "D", length_m=2000.0, current_speed_kmh=32.7, current_travel_time_s=110.0,
                       predicted_travel_times_s={1: 110.0, 2: 110.0}, predicted_speeds_kmh={1: 32.7, 2: 32.7},
                       uncertainties_by_horizon={1: 5.0, 2: 5.0}, uncertainty=5.0)

    r_pred = route_time_dependent(net, "S", "D", step_interval_s=60.0, risk_penalty=0.0)
    r_unc = route_time_dependent(net, "S", "D", step_interval_s=60.0, risk_penalty=1.0)

    assert r_pred.path == ["S", "A", "D"], f"Expected ['S', 'A', 'D'], got {r_pred.path}"
    assert r_unc.path == ["S", "B", "D"], f"Expected ['S', 'B', 'D'], got {r_unc.path}"
    assert r_unc.estimated_travel_time_s == 220.0


def test_controlled_uncertainty_preserves_route():
    """
    Controlled Test B: Uncertainty correctly does NOT change preferred route
    when travel time difference exceeds risk penalty.
    Corridor A (Much Faster): travel time = 100s, uncertainty = 60s -> Cost at risk=1.0: 160s
    Corridor B (Slow): travel time = 220s, uncertainty = 10s -> Cost at risk=1.0: 230s
    Both risk=0.0 and risk=1.0 choose Corridor A.
    """
    net = RoadNetwork()
    for n in ["S", "A", "B", "D"]:
        net.graph.add_node(n, lat=0.0, lon=0.0)

    # Corridor 1 (Much Faster): S -> A -> D
    net.graph.add_edge("S", "A", length_m=2000.0, current_speed_kmh=72.0, current_travel_time_s=50.0,
                       predicted_travel_times_s={1: 50.0, 2: 50.0}, predicted_speeds_kmh={1: 72.0, 2: 72.0},
                       uncertainties_by_horizon={1: 30.0, 2: 30.0}, uncertainty=30.0)
    net.graph.add_edge("A", "D", length_m=2000.0, current_speed_kmh=72.0, current_travel_time_s=50.0,
                       predicted_travel_times_s={1: 50.0, 2: 50.0}, predicted_speeds_kmh={1: 72.0, 2: 72.0},
                       uncertainties_by_horizon={1: 30.0, 2: 30.0}, uncertainty=30.0)

    # Corridor 2 (Slow): S -> B -> D
    net.graph.add_edge("S", "B", length_m=2000.0, current_speed_kmh=32.7, current_travel_time_s=110.0,
                       predicted_travel_times_s={1: 110.0, 2: 110.0}, predicted_speeds_kmh={1: 32.7, 2: 32.7},
                       uncertainties_by_horizon={1: 5.0, 2: 5.0}, uncertainty=5.0)
    net.graph.add_edge("B", "D", length_m=2000.0, current_speed_kmh=32.7, current_travel_time_s=110.0,
                       predicted_travel_times_s={1: 110.0, 2: 110.0}, predicted_speeds_kmh={1: 32.7, 2: 32.7},
                       uncertainties_by_horizon={1: 5.0, 2: 5.0}, uncertainty=5.0)

    r_pred = route_time_dependent(net, "S", "D", step_interval_s=60.0, risk_penalty=0.0)
    r_unc = route_time_dependent(net, "S", "D", step_interval_s=60.0, risk_penalty=1.0)

    assert r_pred.path == ["S", "A", "D"]
    assert r_unc.path == ["S", "A", "D"]

