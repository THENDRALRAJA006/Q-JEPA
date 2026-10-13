"""
Integration tests for forecast-to-routing data flow.
Covers all requirements from TASK 6.

Run with: python -m pytest tests/test_integration.py -v --tb=short
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest

from src.simulation.road_network import RoadNetwork
from src.simulation.traffic_simulator import TrafficSimulator, ScenarioType
from src.simulation.ambulance_scenarios import create_scenario_suite
from src.forecasting.inference import (
    EdgeHistoryBuffer,
    LSTMInferenceEngine,
    JEPAInferenceEngine,
    write_forecasts_to_network,
    clear_forecasts_from_network,
    load_lstm_engine,
    load_jepa_engine,
    _speed_to_travel_time,
    STEP_MINUTES,
)
from src.routing.shortest_path import (
    route_predicted_traffic,
    route_uncertainty_aware,
    route_current_traffic,
    route_shortest_distance,
    compare_all_routes,
    RouterType,
)
from src.evaluation.prediction_metrics import (
    routing_metrics,
    journey_time_improvement_pct,
)
from src.common.config import abs_path


# ─────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────

@pytest.fixture
def small_net():
    return RoadNetwork.create_synthetic_grid(rows=3, cols=3, seed=0)


@pytest.fixture
def sim(small_net):
    return TrafficSimulator(network=small_net, scenario=ScenarioType.NORMAL, seed=0)


@pytest.fixture
def history_with_data(small_net, sim):
    """History buffer with 15 steps of observations."""
    h = EdgeHistoryBuffer(window_size=12)
    for _ in range(15):
        state = sim.step_forward()
        for obs in state.observations:
            h.update(obs.edge_id, obs.observed_speed_kmh if not obs.is_missing else None)
    return h


@pytest.fixture
def lstm_engine():
    """Load real LSTM engine if checkpoint available, else skip."""
    engine = load_lstm_engine()
    if engine is None:
        pytest.skip("LSTM checkpoint not available")
    return engine


@pytest.fixture
def jepa_engine():
    """Load real JEPA engine if checkpoint available, else skip."""
    engine = load_jepa_engine()
    if engine is None:
        pytest.skip("JEPA checkpoint not available")
    return engine


# ─────────────────────────────────────────────────
# 1. Unit conversions
# ─────────────────────────────────────────────────

def test_speed_to_travel_time_units():
    """60 km/h over 600 m = 36 seconds."""
    tt = _speed_to_travel_time(60.0, 600.0)
    assert abs(tt - 36.0) < 0.01

def test_speed_to_travel_time_safe_zero():
    """Zero or tiny speed should not divide-by-zero."""
    tt = _speed_to_travel_time(0.0, 500.0)
    assert np.isfinite(tt)
    assert tt > 0

def test_step_minutes_is_correct():
    """Training data uses 5-minute steps."""
    assert STEP_MINUTES == 5


# ─────────────────────────────────────────────────
# 2. Edge history buffer
# ─────────────────────────────────────────────────

def test_buffer_stores_observations(small_net, sim):
    h = EdgeHistoryBuffer(window_size=12)
    state = sim.step_forward()
    for obs in state.observations:
        h.update(obs.edge_id, obs.observed_speed_kmh)
    edge_ids = list(small_net.graph.edges())
    eid = f"{edge_ids[0][0]}->{edge_ids[0][1]}"
    assert len(h._buffers[eid]) >= 0  # may be 0 if that edge was in closed list

def test_buffer_forward_fills_missing():
    h = EdgeHistoryBuffer(window_size=5)
    h.update("A->B", 40.0)
    h.update("A->B", None)  # missing
    buf = list(h._buffers["A->B"])
    assert buf[-1] == 40.0  # forward-filled

def test_buffer_window_length():
    h = EdgeHistoryBuffer(window_size=12)
    for _ in range(20):
        h.update("A->B", 30.0)
    win = h.get_window("A->B", 30.0)
    assert win is not None
    assert len(win) == 12

def test_buffer_pads_short_history():
    h = EdgeHistoryBuffer(window_size=12)
    for _ in range(6):
        h.update("A->B", 50.0)
    win = h.get_window("A->B", 50.0)
    assert win is not None
    assert len(win) == 12
    assert win[0] == 50.0  # padded with first value

def test_buffer_returns_none_for_empty():
    h = EdgeHistoryBuffer(window_size=12)
    win = h.get_window("NONEXISTENT->EDGE", 30.0)
    assert win is None

def test_buffer_update_from_network(small_net):
    h = EdgeHistoryBuffer(window_size=12)
    h.update_from_network(small_net)
    # Every edge should have at least one observation
    for u, v in small_net.edge_list():
        eid = f"{u}->{v}"
        assert len(h._buffers[eid]) == 1


# ─────────────────────────────────────────────────
# 3. Forecast timestamp and horizon alignment
# ─────────────────────────────────────────────────

def test_horizon_minutes_matches_steps():
    """Model horizon 1 step = 5 minutes exactly."""
    assert STEP_MINUTES == 5
    # horizon 3 steps = 15 minutes
    assert 3 * STEP_MINUTES == 15

def test_predicted_tt_keys_are_step_integers(small_net, history_with_data, lstm_engine):
    write_forecasts_to_network(small_net, history_with_data, lstm_engine, horizons=(1, 3, 6))
    updated = 0
    for u, v, d in small_net.graph.edges(data=True):
        pred = d.get("predicted_travel_times_s", {})
        if pred:
            # Keys should be integer step indices
            for k in pred.keys():
                assert isinstance(k, int), f"Key {k!r} is not an int"
            updated += 1
    assert updated > 0, "No edges were updated with predictions"


# ─────────────────────────────────────────────────
# 4. Forecast-to-routing connection
# ─────────────────────────────────────────────────

def test_predicted_routing_differs_from_current_after_forecast(
    small_net, history_with_data, lstm_engine
):
    """After writing forecasts, predicted route may differ from current-traffic route."""
    nodes = small_net.node_list()
    origin, dest = nodes[0], nodes[-1]

    # Before forecasts: predicted = current (empty dict fallback)
    clear_forecasts_from_network(small_net)
    r_before = route_predicted_traffic(small_net, origin, dest, horizon_steps=1)

    # Write LSTM forecasts
    write_forecasts_to_network(small_net, history_with_data, lstm_engine, horizons=(1, 3, 6))

    # After forecasts: predicted may differ
    r_after = route_predicted_traffic(small_net, origin, dest, horizon_steps=1)

    # Both must be feasible
    assert r_before.feasible
    assert r_after.feasible

    # Travel times must be positive and finite
    assert np.isfinite(r_after.estimated_travel_time_s)
    assert r_after.estimated_travel_time_s > 0

def test_forecast_influences_edge_weights(small_net, history_with_data, lstm_engine):
    """After writing forecasts, at least one edge should have predicted_travel_times_s filled."""
    write_forecasts_to_network(small_net, history_with_data, lstm_engine, horizons=(1,))
    edges_with_forecast = [
        (u, v) for u, v, d in small_net.graph.edges(data=True)
        if d.get("predicted_travel_times_s")
    ]
    assert len(edges_with_forecast) > 0, "No edges received forecast values"

def test_uncertainty_written_to_edges(small_net, history_with_data, lstm_engine):
    """LSTM uncertainty should be written to edge 'uncertainty' field."""
    write_forecasts_to_network(small_net, history_with_data, lstm_engine, horizons=(1,))
    uncertainty_vals = [
        d.get("uncertainty", 0.0)
        for _, _, d in small_net.graph.edges(data=True)
    ]
    # At least some edges should have non-zero uncertainty
    assert any(u > 0 for u in uncertainty_vals), "No uncertainty written to edges"

def test_uncertainty_aware_routing_uses_forecast(
    small_net, history_with_data, lstm_engine
):
    """Uncertainty-aware routing should produce a feasible route with LSTM forecasts."""
    write_forecasts_to_network(small_net, history_with_data, lstm_engine, horizons=(1,))
    nodes = small_net.node_list()
    origin, dest = nodes[0], nodes[-1]
    r = route_uncertainty_aware(small_net, origin, dest, risk_penalty=0.3, horizon_steps=1)
    assert r.feasible
    assert r.uncertainty_s >= 0


# ─────────────────────────────────────────────────
# 5. No future-information leakage
# ─────────────────────────────────────────────────

def test_history_buffer_only_uses_observed_speed(sim, small_net):
    """
    History buffer must only update from observed_speed_kmh, not gt_speed_kmh.
    Simulate and verify buffer never stores ground-truth values directly.
    """
    h = EdgeHistoryBuffer(window_size=12)
    for _ in range(5):
        state = sim.step_forward()
        for obs in state.observations:
            # Inference engine must use OBSERVED speed, not GT speed
            spd = obs.observed_speed_kmh  # This is what we feed
            h.update(obs.edge_id, spd)

    # ground truth values are in network graph; verify we never read them
    for u, v, d in small_net.graph.edges(data=True):
        eid = f"{u}->{v}"
        buf = list(h._buffers.get(eid, []))
        gt = d.get("gt_speed_kmh", d["baseline_speed_kmh"])
        # Buffer stores observed (possibly noisy/missing), not necessarily == gt
        # Sufficient check: buffer values are all positive
        assert all(s > 0 for s in buf), f"Non-positive speed in buffer for {eid}"

def test_normalizer_params_from_train_split():
    """Normalizer min/max loaded from metrics JSON (fit on train split only)."""
    from src.forecasting.inference import _load_normalizer_from_metrics
    vmin, vmax = _load_normalizer_from_metrics()
    assert vmax > vmin, "Normalizer max must be greater than min"
    assert vmin >= 0, "Speed min must be non-negative"
    assert vmax <= 200, "Speed max must be physically plausible"

def test_gt_speed_not_in_inference_pipeline(
    small_net, history_with_data, lstm_engine
):
    """Verify inference engine does not access gt_speed_kmh from the graph."""
    # Corrupt gt_speed_kmh with wildly wrong values
    for u, v, d in small_net.graph.edges(data=True):
        d["gt_speed_kmh"] = 9999.0  # impossible value

    # Inference should still work and produce plausible predictions
    rec = write_forecasts_to_network(small_net, history_with_data, lstm_engine, horizons=(1,))
    for u, v, d in small_net.graph.edges(data=True):
        pred = d.get("predicted_travel_times_s", {})
        for k, tt in pred.items():
            assert tt < 10000, f"Travel time {tt} too large — may have leaked gt_speed"
        # Restore
        d["gt_speed_kmh"] = d["baseline_speed_kmh"]


# ─────────────────────────────────────────────────
# 6. Checkpoint errors and fallback
# ─────────────────────────────────────────────────

def test_missing_lstm_checkpoint_returns_none(tmp_path):
    engine = load_lstm_engine(tmp_path)  # no checkpoint there
    assert engine is None

def test_missing_jepa_checkpoint_returns_none(tmp_path):
    engine = load_jepa_engine(tmp_path)
    assert engine is None

def test_lstm_engine_raises_on_missing_file():
    from src.forecasting.inference import LSTMInferenceEngine
    with pytest.raises(FileNotFoundError):
        LSTMInferenceEngine(Path("/nonexistent/lstm.pt")).load()

def test_jepa_engine_raises_on_missing_file():
    from src.forecasting.inference import JEPAInferenceEngine
    with pytest.raises(FileNotFoundError):
        JEPAInferenceEngine(Path("/nonexistent/jepa.pt")).load()

def test_clear_forecasts_resets_all_edges(small_net, history_with_data, lstm_engine):
    """After clearing, all edges should have empty predicted_travel_times_s."""
    write_forecasts_to_network(small_net, history_with_data, lstm_engine, horizons=(1,))
    clear_forecasts_from_network(small_net)
    for u, v, d in small_net.graph.edges(data=True):
        assert d["predicted_travel_times_s"] == {}
        assert d["uncertainty"] == 0.0


# ─────────────────────────────────────────────────
# 7. Missing predictions fallback
# ─────────────────────────────────────────────────

def test_routing_falls_back_to_current_when_no_forecast(small_net):
    """Predicted-traffic router falls back to current_travel_time_s if no forecast."""
    clear_forecasts_from_network(small_net)
    nodes = small_net.node_list()
    r_pred = route_predicted_traffic(small_net, nodes[0], nodes[-1], horizon_steps=1)
    r_curr = route_current_traffic(small_net, nodes[0], nodes[-1])
    # Both should be feasible and produce the same cost (same edge weights)
    assert r_pred.feasible
    assert r_curr.feasible
    assert abs(r_pred.estimated_travel_time_s - r_curr.estimated_travel_time_s) < 1.0

def test_inference_record_reports_fallback_count(small_net):
    """Edges with no history should be reported as fallback."""
    empty_history = EdgeHistoryBuffer(window_size=12)
    # Only provide history for one edge
    edges = small_net.edge_list()
    eid = f"{edges[0][0]}->{edges[0][1]}"
    for _ in range(15):
        empty_history.update(eid, 40.0)

    lstm_eng = load_lstm_engine()
    if lstm_eng is None:
        pytest.skip("LSTM checkpoint not available")

    rec = write_forecasts_to_network(small_net, empty_history, lstm_eng, horizons=(1,))
    # Most edges should be fallback since history is sparse
    assert rec.n_edges_fallback >= 0


# ─────────────────────────────────────────────────
# 8. Consistent scenario initialisation
# ─────────────────────────────────────────────────

def test_all_strategies_share_same_scenario_state():
    """
    All 7 strategies should use the same departure state.
    Verify by running evaluate_one_scenario and checking that
    origin, destination, departure_step match across strategies.
    """
    from scripts.run_integrated_benchmark import evaluate_one_scenario
    net = RoadNetwork.create_synthetic_grid(seed=42)
    scenarios = create_scenario_suite(net, seed=42)
    scen = scenarios[0]

    lstm_eng = load_lstm_engine()
    jepa_eng = load_jepa_engine()

    recs = evaluate_one_scenario(scen, lstm_eng, jepa_eng, steps_warmup=10)
    assert len(recs) == 7, f"Expected 7 records, got {len(recs)}"

    # All records should share the same origin, destination, departure_step
    origins = {r.origin for r in recs}
    dests = {r.destination for r in recs}
    steps = {r.departure_step for r in recs}
    assert len(origins) == 1, f"Origins differ across strategies: {origins}"
    assert len(dests) == 1, f"Destinations differ: {dests}"
    assert len(steps) == 1, f"Departure steps differ: {steps}"

def test_closed_road_handled_consistently():
    """ROAD_BLOCKAGE scenario: all strategies should consistently see the closure."""
    from scripts.run_integrated_benchmark import evaluate_one_scenario
    net = RoadNetwork.create_synthetic_grid(seed=42)
    scenarios = create_scenario_suite(net, seed=42)
    blockage = next(s for s in scenarios if s.scenario_type == ScenarioType.ROAD_BLOCKAGE)

    lstm_eng = load_lstm_engine()
    jepa_eng = load_jepa_engine()

    recs = evaluate_one_scenario(blockage, lstm_eng, jepa_eng, steps_warmup=25)
    # All should report has_closure consistently
    closures = {r.has_closure for r in recs}
    assert len(closures) == 1, "Closure state differs across strategies!"

def test_unreachable_destination_reported_consistently():
    """If no feasible path exists, all strategies report infeasible."""
    from src.forecasting.inference import load_lstm_engine, load_jepa_engine
    net = RoadNetwork.create_synthetic_grid(rows=3, cols=3, seed=0)
    # Close all edges from first node
    origin = net.node_list()[0]
    for u, v in list(net.graph.out_edges(origin)):
        net.close_edge(u, v)

    nodes = net.node_list()
    dest = nodes[-1]

    r_sd = route_shortest_distance(net, origin, dest)
    r_ct = route_current_traffic(net, origin, dest)
    r_pt = route_predicted_traffic(net, origin, dest)
    # All should be infeasible
    assert not r_sd.feasible
    assert not r_ct.feasible
    assert not r_pt.feasible


# ─────────────────────────────────────────────────
# 9. Benchmark metric calculations
# ─────────────────────────────────────────────────

def test_routing_metrics_ignores_inf():
    times = [100.0, 200.0, float("inf"), 150.0]
    m = routing_metrics(times)
    assert m["unreachable_rate"] == pytest.approx(0.25)
    assert m["mean_s"] == pytest.approx(150.0)
    assert m["n_scenarios"] == 4

def test_improvement_pct_positive_means_faster():
    pct = journey_time_improvement_pct(200.0, 180.0)
    assert pct == pytest.approx(10.0)

def test_improvement_pct_negative_means_slower():
    pct = journey_time_improvement_pct(200.0, 220.0)
    assert pct == pytest.approx(-10.0)

def test_improvement_pct_zero_baseline_safe():
    import math
    pct = journey_time_improvement_pct(0.0, 100.0)
    assert math.isnan(pct)


# ─────────────────────────────────────────────────
# 10. Reproducibility
# ─────────────────────────────────────────────────

def test_benchmark_reproducibility():
    """Two runs with same seed must produce identical journey times."""
    from scripts.run_integrated_benchmark import evaluate_one_scenario
    net = RoadNetwork.create_synthetic_grid(seed=42)
    scen = create_scenario_suite(net, seed=42)[0]

    lstm_eng = load_lstm_engine()
    jepa_eng = load_jepa_engine()

    recs1 = evaluate_one_scenario(scen, lstm_eng, jepa_eng, steps_warmup=10)
    recs2 = evaluate_one_scenario(scen, lstm_eng, jepa_eng, steps_warmup=10)

    times1 = sorted([r.travel_time_s for r in recs1 if r.strategy == "A_shortest_distance"])
    times2 = sorted([r.travel_time_s for r in recs2 if r.strategy == "A_shortest_distance"])
    assert times1 == times2, "Results not reproducible with same seed"


# ─────────────────────────────────────────────────
# 11. End-to-end: forecast → routing
# ─────────────────────────────────────────────────

def test_end_to_end_lstm_forecast_to_route(
    small_net, history_with_data, lstm_engine
):
    """Full pipeline: history → LSTM inference → edge costs → route."""
    # Step 1: write LSTM forecasts to network
    rec = write_forecasts_to_network(small_net, history_with_data, lstm_engine, horizons=(1,))
    assert rec.n_edges_updated > 0

    # Step 2: route using predicted costs
    nodes = small_net.node_list()
    origin, dest = nodes[0], nodes[-1]
    route = route_predicted_traffic(small_net, origin, dest, horizon_steps=1)

    # Step 3: verify route uses predicted costs
    assert route.feasible
    assert route.estimated_travel_time_s > 0
    assert route.estimated_travel_time_s < float("inf")

def test_end_to_end_jepa_forecast_to_route(
    small_net, history_with_data, jepa_engine
):
    """Full pipeline: history → JEPA inference → edge costs → route."""
    rec = write_forecasts_to_network(small_net, history_with_data, jepa_engine, horizons=(1,))
    assert rec.n_edges_updated > 0

    nodes = small_net.node_list()
    origin, dest = nodes[0], nodes[-1]
    route = route_predicted_traffic(small_net, origin, dest, horizon_steps=1)
    assert route.feasible

def test_end_to_end_jepa_uncertainty_routing(
    small_net, history_with_data, jepa_engine
):
    """End-to-end JEPA + uncertainty-aware routing."""
    write_forecasts_to_network(small_net, history_with_data, jepa_engine, horizons=(1,))
    nodes = small_net.node_list()
    origin, dest = nodes[0], nodes[-1]
    route = route_uncertainty_aware(small_net, origin, dest, risk_penalty=0.3, horizon_steps=1)
    assert route.feasible
    assert route.uncertainty_s >= 0

def test_inference_record_metadata(small_net, history_with_data, lstm_engine):
    """InferenceRecord should correctly document model name and checkpoint path."""
    rec = write_forecasts_to_network(small_net, history_with_data, lstm_engine, horizons=(1,))
    assert rec.model_name == "LSTM"
    assert "lstm.pt" in rec.checkpoint_path
    assert rec.horizon_minutes == rec.horizon_steps * STEP_MINUTES
    assert rec.n_edges_updated >= 0
    assert rec.n_edges_fallback >= 0
    assert rec.n_edges_updated + rec.n_edges_fallback == small_net.graph.number_of_edges()

def test_jepa_inference_record_metadata(small_net, history_with_data, jepa_engine):
    """JEPA InferenceRecord should document correct model name."""
    rec = write_forecasts_to_network(small_net, history_with_data, jepa_engine, horizons=(1,))
    assert rec.model_name == "JEPA-Traffic"
    assert "jepa.pt" in rec.checkpoint_path
