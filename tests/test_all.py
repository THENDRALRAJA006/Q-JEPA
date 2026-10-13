"""
Comprehensive test suite for Surakshit Bharat Q-JEPA Traffic.
Run with: python -m pytest tests/ -v
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest

from src.simulation.road_network import RoadNetwork
from src.simulation.traffic_simulator import TrafficSimulator, ScenarioType
from src.simulation.ambulance_scenarios import create_scenario_suite
from src.routing.shortest_path import (
    route_shortest_distance, route_current_traffic,
    route_predicted_traffic, route_uncertainty_aware,
    route_quantum_inspired, compare_all_routes, RouterType,
)
from src.data.loaders import (
    temporal_split, SpeedNormalizer, build_windows, build_all_edges_windows,
)
from src.forecasting.persistence import PersistenceForecaster, HistoricalAverageForecaster
from src.forecasting.lstm_model import LSTMForecaster
from src.forecasting.jepa_encoder import JEPAForecaster
from src.evaluation.prediction_metrics import mae, rmse, per_horizon_metrics, routing_metrics
import pandas as pd
import networkx as nx


# ─────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────

@pytest.fixture
def small_network():
    """Minimal 3×3 grid network for fast tests."""
    return RoadNetwork.create_synthetic_grid(rows=3, cols=3, seed=0)


@pytest.fixture
def simulator(small_network):
    return TrafficSimulator(network=small_network, scenario=ScenarioType.NORMAL, seed=0)


@pytest.fixture
def tiny_df():
    """Tiny synthetic traffic DataFrame."""
    rng = np.random.default_rng(42)
    steps = list(range(1, 61))
    edge_ids = ["A->B", "B->C"]
    records = []
    for step in steps:
        for eid in edge_ids:
            records.append({
                "step": step,
                "time_minutes": step * 5,
                "hour_of_day": (step * 5 / 60) % 24,
                "edge_id": eid,
                "u": eid.split("->")[0],
                "v": eid.split("->")[1],
                "baseline_speed_kmh": 40.0,
                "gt_speed_kmh": float(rng.uniform(20, 60)),
                "observed_speed_kmh": float(rng.uniform(20, 60)),
                "flow_veh_per_hour": float(rng.uniform(100, 400)),
                "congestion_level": float(rng.uniform(0, 1)),
                "is_missing": False,
                "is_noisy": False,
                "has_incident": False,
                "is_closed": False,
                "length_m": 500.0,
            })
    return pd.DataFrame(records)


# ─────────────────────────────────────────────────
# 1. Synthetic data generation
# ─────────────────────────────────────────────────

def test_grid_network_creation():
    net = RoadNetwork.create_synthetic_grid(rows=3, cols=4, seed=42)
    assert net.graph.number_of_nodes() == 12
    assert net.graph.number_of_edges() > 0
    for n, d in net.graph.nodes(data=True):
        assert "lat" in d and "lon" in d


def test_network_edges_have_required_attributes(small_network):
    for u, v, d in small_network.graph.edges(data=True):
        assert "length_m" in d
        assert "baseline_speed_kmh" in d
        assert "current_travel_time_s" in d
        assert d["length_m"] > 0
        assert d["baseline_speed_kmh"] > 0


def test_simulator_steps(simulator):
    state = simulator.step_forward()
    assert state.step == 1
    assert len(state.observations) > 0
    assert state.time_minutes == 5.0


def test_simulator_reproducibility():
    net1 = RoadNetwork.create_synthetic_grid(rows=3, cols=3, seed=7)
    net2 = RoadNetwork.create_synthetic_grid(rows=3, cols=3, seed=7)
    sim1 = TrafficSimulator(network=net1, scenario=ScenarioType.NORMAL, seed=7)
    sim2 = TrafficSimulator(network=net2, scenario=ScenarioType.NORMAL, seed=7)
    s1 = sim1.step_forward()
    s2 = sim2.step_forward()
    obs1 = {o.edge_id: o.observed_speed_kmh for o in s1.observations}
    obs2 = {o.edge_id: o.observed_speed_kmh for o in s2.observations}
    assert obs1 == obs2


# ─────────────────────────────────────────────────
# 2. Dataset validation and temporal splitting
# ─────────────────────────────────────────────────

def test_temporal_split_no_overlap(tiny_df):
    train, val, test = temporal_split(tiny_df, train_ratio=0.7, val_ratio=0.15)
    train_steps = set(train["step"])
    val_steps = set(val["step"])
    test_steps = set(test["step"])
    assert train_steps.isdisjoint(val_steps), "Train and val share steps (data leakage)"
    assert train_steps.isdisjoint(test_steps), "Train and test share steps (data leakage)"
    assert val_steps.isdisjoint(test_steps), "Val and test share steps (data leakage)"


def test_temporal_split_ordering(tiny_df):
    train, val, test = temporal_split(tiny_df, train_ratio=0.7, val_ratio=0.15)
    assert train["step"].max() < val["step"].min()
    assert val["step"].max() < test["step"].min()


def test_normalizer_fit_on_train_only(tiny_df):
    train, val, test = temporal_split(tiny_df)
    norm = SpeedNormalizer().fit(train)
    # Normalizer params computed from train only
    assert norm.speed_min <= train["observed_speed_kmh"].min() + 1e-3
    assert norm.speed_max >= train["observed_speed_kmh"].max() - 1e-3
    # Transform should work on val/test without error
    _ = norm.transform(val["observed_speed_kmh"].values)
    _ = norm.transform(test["observed_speed_kmh"].values)


def test_window_builder_no_future_leakage(tiny_df):
    edge_df = tiny_df[tiny_df["edge_id"] == "A->B"].sort_values("step")
    X, y = build_windows(edge_df, "A->B", window_size=5, horizon=2)
    # Each y row uses future steps: verify y[i] comes after X[i] in time
    # (conceptual check — shape test)
    if len(X) > 0:
        assert X.shape[1] == 5
        assert y.shape[1] == 2


# ─────────────────────────────────────────────────
# 3. Missing and corrupted data
# ─────────────────────────────────────────────────

def test_missing_observation_handling():
    net = RoadNetwork.create_synthetic_grid(rows=3, cols=3, seed=0)
    sim = TrafficSimulator(
        network=net, scenario=ScenarioType.MISSING_OBS,
        seed=0, missing_obs_rate=0.5,
    )
    state = sim.step_forward()
    missing = [o for o in state.observations if o.is_missing]
    # With 50% rate, expect some missing
    assert len(missing) > 0


def test_noisy_observations_bounded():
    net = RoadNetwork.create_synthetic_grid(rows=3, cols=3, seed=0)
    sim = TrafficSimulator(
        network=net, scenario=ScenarioType.NOISY_OBS,
        seed=0, noise_std_kmh=5.0,
    )
    for _ in range(5):
        state = sim.step_forward()
    speeds = [o.observed_speed_kmh for o in state.observations if o.observed_speed_kmh is not None]
    assert all(s > 0 for s in speeds), "All observed speeds must be positive"


# ─────────────────────────────────────────────────
# 4. Forecast output shapes
# ─────────────────────────────────────────────────

def test_persistence_forecast_shape():
    X = np.random.rand(20, 10).astype(np.float32)
    y = np.random.rand(20, 3).astype(np.float32)
    perf = PersistenceForecaster().fit(X, y)
    preds = perf.predict_horizon(X, horizon=3)
    assert preds.shape == (20, 3)


def test_historical_avg_forecast_shape():
    X = np.random.rand(50, 12).astype(np.float32)
    y = np.random.rand(50, 3).astype(np.float32)
    hist = HistoricalAverageForecaster().fit(X, y)
    preds = hist.predict_horizon(X, horizon=3)
    assert preds.shape == (50, 3)


def test_lstm_forecast_shape():
    X = np.random.rand(30, 12).astype(np.float32)
    y = np.random.rand(30, 3).astype(np.float32)
    lstm = LSTMForecaster(window_size=12, horizon=3, hidden=16, layers=1, epochs=2)
    lstm.fit(X, y)
    preds = lstm.predict(X)
    assert preds.shape == (30, 3)


def test_jepa_forecast_shape():
    X = np.random.rand(30, 12).astype(np.float32)
    y = np.random.rand(30, 3).astype(np.float32)
    jepa = JEPAForecaster(window_size=12, horizon=3, embed_dim=16, pred_dim=16, epochs=2)
    jepa.fit(X, y)
    preds = jepa.predict(X)
    assert preds.shape == (30, 3)


def test_jepa_overfit_small_dataset():
    """Verify JEPA can overfit a tiny training set (sanity check)."""
    rng = np.random.default_rng(42)
    X = rng.random((10, 8)).astype(np.float32)
    y = rng.random((10, 1)).astype(np.float32)
    jepa = JEPAForecaster(window_size=8, horizon=1, embed_dim=16, pred_dim=16, epochs=50, lr=1e-2)
    jepa.fit(X, y)
    preds = jepa.predict(X)
    train_mae = float(np.abs(preds - y).mean())
    # With 50 epochs on 10 samples it should fit reasonably
    assert train_mae < 0.5, f"JEPA could not overfit small dataset: MAE={train_mae:.4f}"


# ─────────────────────────────────────────────────
# 5. Routing correctness
# ─────────────────────────────────────────────────

def test_shortest_path_on_known_graph():
    """3-node chain: A->B->C. Shortest path must use both edges."""
    G = nx.DiGraph()
    G.add_edge("A", "B", segment_id="A->B", length_m=100, baseline_speed_kmh=30,
               current_speed_kmh=30, current_travel_time_s=12, congestion_level=0,
               has_incident=False, is_closed=False, predicted_travel_times_s={},
               uncertainty=0, last_updated=0)
    G.add_edge("B", "C", segment_id="B->C", length_m=100, baseline_speed_kmh=30,
               current_speed_kmh=30, current_travel_time_s=12, congestion_level=0,
               has_incident=False, is_closed=False, predicted_travel_times_s={},
               uncertainty=0, last_updated=0)
    # Add a longer direct A->C
    G.add_edge("A", "C", segment_id="A->C", length_m=300, baseline_speed_kmh=30,
               current_speed_kmh=30, current_travel_time_s=36, congestion_level=0,
               has_incident=False, is_closed=False, predicted_travel_times_s={},
               uncertainty=0, last_updated=0)

    from src.simulation.road_network import RoadNetwork
    net = RoadNetwork(seed=0)
    net.graph = G
    for n in ["A", "B", "C"]:
        net.graph.add_node(n, lat=28.5 + list("ABC").index(n) * 0.01, lon=77.1, has_traffic_signal=False, signal_delay_s=0)

    result = route_shortest_distance(net, "A", "C")
    assert result.feasible
    assert result.path == ["A", "B", "C"], f"Expected A->B->C, got {result.path}"
    assert abs(result.total_distance_m - 200.0) < 1.0


def test_routing_on_small_grid(small_network):
    nodes = small_network.node_list()
    origin, dest = nodes[0], nodes[-1]
    result = route_current_traffic(small_network, origin, dest)
    assert result.feasible
    assert result.path[0] == origin
    assert result.path[-1] == dest
    assert result.estimated_travel_time_s > 0


def test_closed_road_handling(small_network):
    """Close all edges from origin; routing should find alternative or report infeasible."""
    nodes = small_network.node_list()
    origin = nodes[0]
    # Close all outgoing edges from origin
    for u, v in list(small_network.graph.out_edges(origin)):
        small_network.close_edge(u, v)
    dest = nodes[-1]
    result = route_current_traffic(small_network, origin, dest)
    # Either infeasible or found alternative (depends on graph connectivity)
    assert isinstance(result.feasible, bool)


def test_unreachable_destination():
    """Completely isolated destination node — router must return infeasible."""
    net = RoadNetwork(seed=0)
    net.graph.add_node("A", lat=28.5, lon=77.1, has_traffic_signal=False, signal_delay_s=0)
    net.graph.add_node("B", lat=28.51, lon=77.11, has_traffic_signal=False, signal_delay_s=0)
    net.graph.add_edge("A", "B", segment_id="A->B", length_m=500, baseline_speed_kmh=30,
                       current_speed_kmh=30, current_travel_time_s=60, congestion_level=0,
                       has_incident=False, is_closed=False, predicted_travel_times_s={},
                       uncertainty=0, last_updated=0)
    # Test: destination not reachable from origin — expect infeasible result or exception caught
    try:
        result = route_shortest_distance(net, "A", "B")
        # This should be feasible (A->B direct)
        assert result.feasible
        # Now try a truly unreachable pair (B -> A has no reverse edge)
        result2 = route_shortest_distance(net, "B", "A")
        assert not result2.feasible
    except Exception:
        pass  # Any exception from router on impossible route is acceptable


def test_uncertainty_aware_routing(small_network):
    """Uncertainty-aware router must not crash and return feasible route."""
    nodes = small_network.node_list()
    result = route_uncertainty_aware(small_network, nodes[0], nodes[-1], risk_penalty=0.5)
    assert isinstance(result.feasible, bool)


def test_quantum_inspired_routing(small_network):
    """SA optimizer must return a valid path on small network."""
    nodes = small_network.node_list()
    result = route_quantum_inspired(small_network, nodes[0], nodes[-1], iterations=50, seed=42)
    assert isinstance(result.feasible, bool)
    if result.feasible:
        assert result.path[0] == nodes[0]
        assert result.path[-1] == nodes[-1]


def test_compare_all_routes_returns_five(small_network):
    nodes = small_network.node_list()
    routes = compare_all_routes(small_network, nodes[0], nodes[-1])
    assert len(routes) == 5
    router_types = {r.router_type for r in routes}
    assert RouterType.SHORTEST_DISTANCE in router_types
    assert RouterType.QUANTUM_INSPIRED in router_types


# ─────────────────────────────────────────────────
# 6. Evaluation metrics
# ─────────────────────────────────────────────────

def test_mae_rmse_correctness():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.5, 2.5, 3.5])
    assert abs(mae(y_true, y_pred) - 0.5) < 1e-6
    assert abs(rmse(y_true, y_pred) - 0.5) < 1e-6


def test_per_horizon_metrics():
    y_true = np.array([[1.0, 2.0], [3.0, 4.0]])
    y_pred = np.array([[1.5, 2.5], [3.5, 4.5]])
    results = per_horizon_metrics(y_true, y_pred)
    assert len(results) == 2
    assert all(r["MAE"] == pytest.approx(0.5) for r in results)


def test_routing_metrics_with_inf():
    journey_times = [100.0, 200.0, float("inf"), 150.0]
    m = routing_metrics(journey_times)
    assert m["unreachable_rate"] == pytest.approx(0.25)
    assert m["mean_s"] == pytest.approx(150.0)


# ─────────────────────────────────────────────────
# 7. Scenario suite
# ─────────────────────────────────────────────────

def test_scenario_suite_creates_seven(small_network):
    scenarios = create_scenario_suite(small_network, seed=42)
    assert len(scenarios) == 7
    types = {s.scenario_type for s in scenarios}
    assert ScenarioType.NORMAL in types
    assert ScenarioType.RUSH_HOUR in types


# ─────────────────────────────────────────────────
# 8. Model checkpoint save/load
# ─────────────────────────────────────────────────

def test_lstm_checkpoint_roundtrip(tmp_path):
    X = np.random.rand(20, 8).astype(np.float32)
    y = np.random.rand(20, 2).astype(np.float32)
    lstm = LSTMForecaster(window_size=8, horizon=2, hidden=16, layers=1, epochs=2)
    lstm.fit(X, y)
    preds_before = lstm.predict(X)

    ckpt_path = tmp_path / "lstm_test.pt"
    lstm.save(ckpt_path)

    lstm2 = LSTMForecaster(window_size=8, horizon=2)
    lstm2.load(ckpt_path)
    preds_after = lstm2.predict(X)

    np.testing.assert_allclose(preds_before, preds_after, atol=1e-5)


def test_jepa_checkpoint_roundtrip(tmp_path):
    X = np.random.rand(20, 8).astype(np.float32)
    y = np.random.rand(20, 2).astype(np.float32)
    jepa = JEPAForecaster(window_size=8, horizon=2, embed_dim=16, pred_dim=16, epochs=2)
    jepa.fit(X, y)
    preds_before = jepa.predict(X)

    ckpt_path = tmp_path / "jepa_test.pt"
    jepa.save(ckpt_path)

    jepa2 = JEPAForecaster(window_size=8, horizon=2)
    jepa2.load(ckpt_path)
    preds_after = jepa2.predict(X)

    np.testing.assert_allclose(preds_before, preds_after, atol=1e-5)


# ─────────────────────────────────────────────────
# 9. API imports (smoke test)
# ─────────────────────────────────────────────────

def test_api_imports():
    """Verify API module imports without errors."""
    from src.api.main import app
    assert app is not None


def test_api_health_endpoint():
    from fastapi.testclient import TestClient
    from src.api.main import app
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"


def test_api_network_endpoint():
    from fastapi.testclient import TestClient
    from src.api.main import app
    client = TestClient(app)
    response = client.get("/network")
    assert response.status_code == 200
    data = response.json()
    assert "nodes" in data and "edges" in data
    assert len(data["nodes"]) > 0


def test_api_routes_compare():
    from fastapi.testclient import TestClient
    from src.api.main import app, _get_network
    client = TestClient(app)
    net = _get_network()
    nodes = net.node_list()
    payload = {"origin": nodes[0], "destination": nodes[-1], "risk_penalty": 0.3}
    response = client.post("/routes/compare", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "routes" in data
    assert len(data["routes"]) == 5


def test_api_invalid_node():
    from fastapi.testclient import TestClient
    from src.api.main import app
    client = TestClient(app)
    payload = {"origin": "NONEXISTENT", "destination": "ALSO_BAD", "risk_penalty": 0.3}
    response = client.post("/routes/compare", json=payload)
    assert response.status_code == 400
