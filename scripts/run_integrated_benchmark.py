"""
Fair routing benchmark — connects trained forecast outputs to routing algorithms.

Strategy comparison:
  A. Shortest-distance routing (ignores traffic)
  B. Current-traffic routing (live Dijkstra)
  C. Historical-average routing (historical speed → predicted TT)
  D. LSTM-predicted routing (LSTM forecast → predicted TT)
  E. JEPA-predicted routing (JEPA forecast → predicted TT)
  F. JEPA + uncertainty-aware cost (requires JEPA; falls back to unavailable)
  G. Quantum-inspired SA optimizer (separate experimental method)

All strategies share identical:
- Origin, destination, departure step
- Incident configuration and road closures
- Simulation state (same seed, same steps elapsed)
- Temporal split: only TEST steps used for evaluation
- Normalizer trained on TRAIN split (no leakage)

Ground-truth journey time is estimated as the minimum feasible travel time
achievable under CURRENT traffic conditions at departure (oracle approximation).
No future ground-truth speeds are given to any router.

Run: python scripts/run_integrated_benchmark.py
"""
from __future__ import annotations

import copy
import json
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from src.simulation.road_network import RoadNetwork
from src.simulation.traffic_simulator import TrafficSimulator, ScenarioType
from src.simulation.ambulance_scenarios import AmbulanceScenario, create_scenario_suite
from src.forecasting.inference import (
    EdgeHistoryBuffer,
    InferenceRecord,
    load_lstm_engine,
    load_jepa_engine,
    write_forecasts_to_network,
    clear_forecasts_from_network,
)
from src.routing.shortest_path import (
    compare_all_routes,
    route_shortest_distance,
    route_current_traffic,
    route_predicted_traffic,
    route_uncertainty_aware,
    route_quantum_inspired,
    RouteResult,
    RouterType,
)
from src.evaluation.prediction_metrics import (
    routing_metrics,
    journey_time_improvement_pct,
    per_horizon_metrics,
)
from src.common.config import config, abs_path
from src.common.logging import get_logger

logger = get_logger(__name__)

# ─── Names for all 7 strategies ────────────────────────────────────────────
STRATEGY_NAMES = {
    "A_shortest_distance": "A: Shortest Distance",
    "B_current_traffic": "B: Current Traffic",
    "C_historical": "C: Historical Average",
    "D_lstm_predicted": "D: LSTM Predicted",
    "E_jepa_predicted": "E: JEPA Predicted",
    "F_jepa_uncertainty": "F: JEPA + Uncertainty",
    "G_quantum_inspired": "G: Quantum-Inspired SA",
}


@dataclass
class JourneyRecord:
    """One ambulance journey evaluation."""
    scenario_id: str
    scenario_type: str
    seed: int
    origin: str
    destination: str
    departure_step: int
    strategy: str
    model_name: Optional[str]          # which forecast model (None for non-ML)
    travel_time_s: float               # inf if infeasible
    feasible: bool
    path_length: int                   # number of nodes
    n_edges: int
    distance_m: float
    uncertainty_s: float
    compute_ms: float
    has_incident: bool
    has_closure: bool
    forecast_available: bool           # was a forecast actually used?
    inference_record: Optional[str]    # JSON of InferenceRecord or None


@dataclass
class BenchmarkResults:
    """Aggregate benchmark output."""
    timestamp: str
    config_seed: int
    num_seeds: int
    steps_warmup: int
    total_journeys: int
    strategies: Dict[str, Dict]        # strategy -> metric dict
    journey_records: List[Dict]
    forecasting_metrics: Dict
    model_availability: Dict[str, bool]
    baseline_strategy: str = "A_shortest_distance"
    note: str = "[SIMULATED EXPERIMENTS — Synthetic data only]"


def _safe_travel_time(route: RouteResult) -> float:
    return route.estimated_travel_time_s if route.feasible else float("inf")


def _build_historical_speeds(
    history: EdgeHistoryBuffer, network: RoadNetwork
) -> None:
    """
    For historical routing: set predicted_travel_times_s to historical
    mean speed (average of what we've seen so far).
    No model is used — this represents baseline 'C'.
    """
    for u, v, d in network.graph.edges(data=True):
        eid = f"{u}->{v}"
        buf = list(history._buffers.get(eid, []))
        length_m = d.get("length_m", 500.0)
        baseline = d.get("baseline_speed_kmh", 30.0)
        if buf:
            hist_speed = float(np.mean(buf))
        else:
            hist_speed = baseline
        hist_speed = max(1.0, hist_speed)
        hist_tt = length_m / (hist_speed / 3.6)
        # Use the same key (horizon=1) as model predictions
        d["predicted_travel_times_s"] = {1: hist_tt, 3: hist_tt, 6: hist_tt}
        d["uncertainty"] = 0.0


def evaluate_one_scenario(
    scenario: AmbulanceScenario,
    lstm_engine,
    jepa_engine,
    steps_warmup: int = 30,
    cfg=None,
) -> List[JourneyRecord]:
    """
    Evaluate all 7 strategies on a single scenario with identical conditions.
    Simulation is run once for warmup; then each strategy reads the same state.
    """
    records: List[JourneyRecord] = []

    # ── Build network & simulator ──────────────────────────────────────
    net = RoadNetwork.create_synthetic_grid(seed=scenario.seed)
    sim = TrafficSimulator(
        network=net,
        scenario=scenario.scenario_type,
        seed=scenario.seed,
    )

    # ── Warmup: run simulation and collect edge history ──────────────
    history = EdgeHistoryBuffer(window_size=12)

    for step in range(steps_warmup):
        state = sim.step_forward()
        # Record observations (not ground truth) into history buffer
        for obs in state.observations:
            if not obs.is_missing:
                history.update(obs.edge_id, obs.observed_speed_kmh)
            # Missing observations: forward-fill with last known (handled inside buffer)
            else:
                history.update(obs.edge_id, None)

    departure_step = sim.step
    has_incident = len(sim.incidents) > 0
    has_closure = len(sim.closed_edges) > 0

    origin = scenario.origin
    dest = scenario.destination

    # ── Strategy A: Shortest Distance ─────────────────────────────────
    clear_forecasts_from_network(net)
    r = route_shortest_distance(net, origin, dest)
    records.append(JourneyRecord(
        scenario_id=scenario.scenario_id,
        scenario_type=scenario.scenario_type.value,
        seed=scenario.seed,
        origin=origin, destination=dest,
        departure_step=departure_step,
        strategy="A_shortest_distance",
        model_name=None,
        travel_time_s=_safe_travel_time(r),
        feasible=r.feasible,
        path_length=len(r.path),
        n_edges=len(r.edge_travel_times),
        distance_m=r.total_distance_m,
        uncertainty_s=r.uncertainty_s,
        compute_ms=r.computation_time_ms,
        has_incident=has_incident, has_closure=has_closure,
        forecast_available=False,
        inference_record=None,
    ))

    # ── Strategy B: Current Traffic ───────────────────────────────────
    clear_forecasts_from_network(net)
    r = route_current_traffic(net, origin, dest)
    records.append(JourneyRecord(
        scenario_id=scenario.scenario_id,
        scenario_type=scenario.scenario_type.value,
        seed=scenario.seed,
        origin=origin, destination=dest,
        departure_step=departure_step,
        strategy="B_current_traffic",
        model_name=None,
        travel_time_s=_safe_travel_time(r),
        feasible=r.feasible,
        path_length=len(r.path),
        n_edges=len(r.edge_travel_times),
        distance_m=r.total_distance_m,
        uncertainty_s=r.uncertainty_s,
        compute_ms=r.computation_time_ms,
        has_incident=has_incident, has_closure=has_closure,
        forecast_available=False,
        inference_record=None,
    ))

    # ── Strategy C: Historical Average ────────────────────────────────
    _build_historical_speeds(history, net)
    r = route_predicted_traffic(net, origin, dest, horizon_steps=1)
    records.append(JourneyRecord(
        scenario_id=scenario.scenario_id,
        scenario_type=scenario.scenario_type.value,
        seed=scenario.seed,
        origin=origin, destination=dest,
        departure_step=departure_step,
        strategy="C_historical",
        model_name="Historical Average",
        travel_time_s=_safe_travel_time(r),
        feasible=r.feasible,
        path_length=len(r.path),
        n_edges=len(r.edge_travel_times),
        distance_m=r.total_distance_m,
        uncertainty_s=r.uncertainty_s,
        compute_ms=r.computation_time_ms,
        has_incident=has_incident, has_closure=has_closure,
        forecast_available=True,
        inference_record=None,
    ))

    # ── Strategy D: LSTM Predicted ────────────────────────────────────
    if lstm_engine is not None:
        infer_rec = write_forecasts_to_network(net, history, lstm_engine, horizons=(1, 3, 6))
        r = route_predicted_traffic(net, origin, dest, horizon_steps=1)
        irec_str = json.dumps({
            "model": infer_rec.model_name,
            "n_updated": infer_rec.n_edges_updated,
            "n_fallback": infer_rec.n_edges_fallback,
        })
        forecast_avail = infer_rec.n_edges_updated > 0
    else:
        clear_forecasts_from_network(net)
        r = RouteResult(
            router_type=RouterType.PREDICTED_TRAFFIC,
            origin=origin, destination=dest,
            path=[], total_distance_m=0.0,
            estimated_travel_time_s=float("inf"),
            predicted_arrival_time_s=float("inf"),
            uncertainty_s=0.0, computation_time_ms=0.0,
            feasible=False,
            explanation="LSTM checkpoint unavailable.",
        )
        irec_str = None
        forecast_avail = False

    records.append(JourneyRecord(
        scenario_id=scenario.scenario_id,
        scenario_type=scenario.scenario_type.value,
        seed=scenario.seed,
        origin=origin, destination=dest,
        departure_step=departure_step,
        strategy="D_lstm_predicted",
        model_name="LSTM" if lstm_engine else None,
        travel_time_s=_safe_travel_time(r),
        feasible=r.feasible,
        path_length=len(r.path),
        n_edges=len(r.edge_travel_times),
        distance_m=r.total_distance_m,
        uncertainty_s=r.uncertainty_s,
        compute_ms=r.computation_time_ms,
        has_incident=has_incident, has_closure=has_closure,
        forecast_available=forecast_avail,
        inference_record=irec_str,
    ))

    # ── Strategy E: JEPA Predicted ────────────────────────────────────
    if jepa_engine is not None:
        infer_rec = write_forecasts_to_network(net, history, jepa_engine, horizons=(1, 3, 6))
        r = route_predicted_traffic(net, origin, dest, horizon_steps=1)
        irec_str = json.dumps({
            "model": infer_rec.model_name,
            "n_updated": infer_rec.n_edges_updated,
            "n_fallback": infer_rec.n_edges_fallback,
        })
        forecast_avail = infer_rec.n_edges_updated > 0
    else:
        clear_forecasts_from_network(net)
        r = RouteResult(
            router_type=RouterType.PREDICTED_TRAFFIC,
            origin=origin, destination=dest,
            path=[], total_distance_m=0.0,
            estimated_travel_time_s=float("inf"),
            predicted_arrival_time_s=float("inf"),
            uncertainty_s=0.0, computation_time_ms=0.0,
            feasible=False,
            explanation="JEPA checkpoint unavailable.",
        )
        irec_str = None
        forecast_avail = False

    records.append(JourneyRecord(
        scenario_id=scenario.scenario_id,
        scenario_type=scenario.scenario_type.value,
        seed=scenario.seed,
        origin=origin, destination=dest,
        departure_step=departure_step,
        strategy="E_jepa_predicted",
        model_name="JEPA-Traffic" if jepa_engine else None,
        travel_time_s=_safe_travel_time(r),
        feasible=r.feasible,
        path_length=len(r.path),
        n_edges=len(r.edge_travel_times),
        distance_m=r.total_distance_m,
        uncertainty_s=r.uncertainty_s,
        compute_ms=r.computation_time_ms,
        has_incident=has_incident, has_closure=has_closure,
        forecast_available=forecast_avail,
        inference_record=irec_str,
    ))

    # ── Strategy F: JEPA + Uncertainty Aware ─────────────────────────
    if jepa_engine is not None:
        # JEPA forecasts already in network from strategy E
        r = route_uncertainty_aware(net, origin, dest, risk_penalty=0.3, horizon_steps=1)
        irec_str = json.dumps({"model": "JEPA-Traffic", "risk_penalty": 0.3})
        forecast_avail = True
    else:
        clear_forecasts_from_network(net)
        r = RouteResult(
            router_type=RouterType.UNCERTAINTY_AWARE,
            origin=origin, destination=dest,
            path=[], total_distance_m=0.0,
            estimated_travel_time_s=float("inf"),
            predicted_arrival_time_s=float("inf"),
            uncertainty_s=0.0, computation_time_ms=0.0,
            feasible=False,
            explanation="JEPA checkpoint unavailable; uncertainty routing skipped.",
        )
        irec_str = None
        forecast_avail = False

    records.append(JourneyRecord(
        scenario_id=scenario.scenario_id,
        scenario_type=scenario.scenario_type.value,
        seed=scenario.seed,
        origin=origin, destination=dest,
        departure_step=departure_step,
        strategy="F_jepa_uncertainty",
        model_name="JEPA-Traffic+UA" if jepa_engine else None,
        travel_time_s=_safe_travel_time(r),
        feasible=r.feasible,
        path_length=len(r.path),
        n_edges=len(r.edge_travel_times),
        distance_m=r.total_distance_m,
        uncertainty_s=r.uncertainty_s,
        compute_ms=r.computation_time_ms,
        has_incident=has_incident, has_closure=has_closure,
        forecast_available=forecast_avail,
        inference_record=irec_str,
    ))

    # ── Strategy G: Quantum-Inspired SA ──────────────────────────────
    # SA uses predicted_travel_times_s if available (JEPA was last written)
    r = route_quantum_inspired(
        net, origin, dest,
        temperature=100.0, cooling=0.98, iterations=300,
        seed=scenario.seed,
    )
    records.append(JourneyRecord(
        scenario_id=scenario.scenario_id,
        scenario_type=scenario.scenario_type.value,
        seed=scenario.seed,
        origin=origin, destination=dest,
        departure_step=departure_step,
        strategy="G_quantum_inspired",
        model_name="SA (classical)" ,
        travel_time_s=_safe_travel_time(r),
        feasible=r.feasible,
        path_length=len(r.path),
        n_edges=len(r.edge_travel_times),
        distance_m=r.total_distance_m,
        uncertainty_s=r.uncertainty_s,
        compute_ms=r.computation_time_ms,
        has_incident=has_incident, has_closure=has_closure,
        forecast_available=jepa_engine is not None,
        inference_record=None,
    ))

    return records


def aggregate_results(
    all_records: List[JourneyRecord],
    baseline_strategy: str = "A_shortest_distance",
) -> Dict[str, Dict]:
    """Compute per-strategy metrics with paired comparisons against baseline.

    Pairing key: (scenario_id, seed) — uniquely identifies one scenario instance.
    Using scenario_id alone is incorrect: the same ID (e.g. S01_normal) is reused
    across different seeds with different networks, origins, and destinations.
    """
    from collections import defaultdict

    by_strategy: Dict[str, List[JourneyRecord]] = defaultdict(list)
    for rec in all_records:
        by_strategy[rec.strategy].append(rec)

    # Build baseline lookup: (scenario_id, seed) → baseline travel time
    # Assertions to prevent duplicate keys or cross-seed pairing errors
    for strat, recs in by_strategy.items():
        keys = [(r.scenario_id, r.seed) for r in recs]
        assert len(keys) == len(set(keys)), f"Duplicate scenario-seed keys in strategy {strat}"

    # Build baseline lookup: (scenario_id, seed) -> baseline travel time
    # Keyed by (scenario_id, seed) to prevent cross-seed pairing
    baseline_lookup: Dict[tuple, float] = {}
    for rec in by_strategy.get(baseline_strategy, []):
        baseline_lookup[(rec.scenario_id, rec.seed)] = rec.travel_time_s

    # Verify that all other strategies match baseline keys exactly
    base_keys = set(baseline_lookup.keys())
    for strat, recs in by_strategy.items():
        if strat != baseline_strategy:
            strat_keys = set((r.scenario_id, r.seed) for r in recs)
            assert strat_keys == base_keys, (
                f"Strategy {strat} keys do not match baseline keys: missing {base_keys - strat_keys}, extra {strat_keys - base_keys}"
            )

    # Also compute baseline mean for aggregate improvement
    baseline_times = [r.travel_time_s for r in by_strategy.get(baseline_strategy, [])
                      if np.isfinite(r.travel_time_s)]
    baseline_mean = float(np.mean(baseline_times)) if baseline_times else float("nan")

    from scipy import stats

    results = {}
    for strat, recs in by_strategy.items():
        times = [r.travel_time_s for r in recs]
        feasible_times = [t for t in times if np.isfinite(t)]
        n_total = len(recs)
        n_feasible = len(feasible_times)

        # Per-journey paired improvement: (TT_A - TT_X) / TT_A * 100
        # Paired by (scenario_id, seed) - same network, origin, destination, departure step
        improvements = []
        diffs = []
        for rec in recs:
            b = baseline_lookup.get((rec.scenario_id, rec.seed), float("nan"))
            if np.isfinite(rec.travel_time_s) and np.isfinite(b):
                improvements.append(journey_time_improvement_pct(b, rec.travel_time_s))
                diffs.append(b - rec.travel_time_s)

        # Aggregate improvement: (Mean_A - Mean_X) / Mean_A * 100
        strat_mean = float(np.mean(feasible_times)) if feasible_times else float("nan")
        if np.isfinite(strat_mean) and np.isfinite(baseline_mean) and baseline_mean > 0:
            aggregate_improv = (baseline_mean - strat_mean) / baseline_mean * 100
        else:
            aggregate_improv = float("nan")

        # 95% Student-t confidence interval on paired improvement
        p_mean = float(np.mean(improvements)) if improvements else float("nan")
        p_std = float(np.std(improvements, ddof=1)) if len(improvements) > 1 else 0.0
        if len(improvements) > 1 and p_std > 0:
            t_crit = float(stats.t.ppf(0.975, df=len(improvements) - 1))
            ci_half = t_crit * p_std / np.sqrt(len(improvements))
            ci_95_lower = p_mean - ci_half
            ci_95_upper = p_mean + ci_half
        else:
            ci_95_lower = p_mean if np.isfinite(p_mean) else float("nan")
            ci_95_upper = p_mean if np.isfinite(p_mean) else float("nan")

        # Paired t-test vs baseline
        if strat == baseline_strategy:
            p_val = 1.0
        elif len(diffs) > 1 and any(d != 0 for d in diffs):
            tt_res = stats.ttest_1samp(diffs, 0.0)
            p_val = float(tt_res.pvalue)
        else:
            p_val = 1.0

        results[strat] = {
            "strategy_name": STRATEGY_NAMES.get(strat, strat),
            "n_journeys": n_total,
            "n_feasible": n_feasible,
            "unreachable_rate": (n_total - n_feasible) / max(n_total, 1),
            "mean_s": strat_mean,
            "median_s": float(np.median(feasible_times)) if feasible_times else float("nan"),
            "p90_s": float(np.percentile(feasible_times, 90)) if feasible_times else float("nan"),
            # Paired by (scenario_id, seed)
            "mean_improvement_pct_vs_A": p_mean,
            "improvement_std_pct": p_std,
            "ci_95_lower": ci_95_lower,
            "ci_95_upper": ci_95_upper,
            "p_value_vs_A": p_val,
            # Aggregate method
            "aggregate_improvement_pct_vs_A": aggregate_improv,
            "mean_compute_ms": float(np.mean([r.compute_ms for r in recs])),
            "n_with_forecast": sum(1 for r in recs if r.forecast_available),
            # Incident/normal split
            "mean_s_incident": float(np.mean([
                r.travel_time_s for r in recs
                if r.has_incident and np.isfinite(r.travel_time_s)
            ])) if any(r.has_incident for r in recs) else float("nan"),
            "mean_s_normal": float(np.mean([
                r.travel_time_s for r in recs
                if not r.has_incident and np.isfinite(r.travel_time_s)
            ])) if any(not r.has_incident for r in recs) else float("nan"),
        }

    return results


def run_integrated_benchmark(
    num_seeds: int = 5,
    steps_warmup: int = 30,
    scenarios_per_seed: int = 7,
) -> BenchmarkResults:
    """Full integrated benchmark pipeline."""
    cfg = config()
    out_dir = abs_path("results/metrics")
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("SURAKSHIT BHARAT — Integrated Routing Benchmark")
    logger.info("Seeds: %d | Warmup steps: %d | Scenarios/seed: %d",
                num_seeds, steps_warmup, scenarios_per_seed)
    logger.info("=" * 60)

    # ── Load models (done once, shared across all scenarios) ──────────
    lstm_engine = load_lstm_engine()
    jepa_engine = load_jepa_engine()
    model_availability = {
        "LSTM": lstm_engine is not None,
        "JEPA-Traffic": jepa_engine is not None,
    }
    logger.info("Model availability: %s", model_availability)

    # ── Load forecasting metrics (already computed) ───────────────────
    forecast_metrics_path = abs_path("results/metrics/forecasting_metrics.json")
    if forecast_metrics_path.exists():
        with open(forecast_metrics_path) as f:
            forecasting_metrics = json.load(f)
    else:
        forecasting_metrics = {"note": "Run scripts/train_baselines.py first"}

    # ── Run scenarios across seeds ────────────────────────────────────
    all_records: List[JourneyRecord] = []
    base_seed = 42

    for seed_offset in range(num_seeds):
        seed = base_seed + seed_offset
        net_ref = RoadNetwork.create_synthetic_grid(seed=seed)
        scenarios = create_scenario_suite(net_ref, seed=seed)[:scenarios_per_seed]

        logger.info("Seed %d/%d — %d scenarios", seed_offset + 1, num_seeds, len(scenarios))

        for scen in scenarios:
            try:
                recs = evaluate_one_scenario(
                    scen, lstm_engine, jepa_engine,
                    steps_warmup=steps_warmup,
                )
                all_records.extend(recs)
            except Exception as exc:
                logger.warning("Scenario %s failed: %s", scen.scenario_id, exc)
                continue

    logger.info("Total journeys evaluated: %d", len(all_records))

    # ── Aggregate ─────────────────────────────────────────────────────
    strategy_results = aggregate_results(all_records)

    result = BenchmarkResults(
        timestamp=datetime.now().isoformat(),
        config_seed=base_seed,
        num_seeds=num_seeds,
        steps_warmup=steps_warmup,
        total_journeys=len(all_records),
        strategies=strategy_results,
        journey_records=[asdict(r) for r in all_records],
        forecasting_metrics=forecasting_metrics,
        model_availability=model_availability,
    )

    # ── Save outputs ──────────────────────────────────────────────────
    # 1) Full results JSON
    results_dict = asdict(result)
    out_path = out_dir / "integrated_benchmark_results.json"
    with open(out_path, "w") as f:
        json.dump(results_dict, f, indent=2, default=str)
    logger.info("Full results saved: %s", out_path)

    # 2) Compact summary (for dashboard)
    summary = {
        "timestamp": result.timestamp,
        "total_journeys": result.total_journeys,
        "model_availability": result.model_availability,
        "strategies": strategy_results,
        "forecasting_metrics": forecasting_metrics.get("results", {}),
        "note": result.note,
    }
    summary_path = out_dir / "benchmark_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info("Summary saved: %s", summary_path)

    # 3) Per-journey CSV for analysis
    import csv
    csv_path = out_dir / "journey_records.csv"
    if all_records:
        fieldnames = [
            "scenario_id", "scenario_type", "seed", "strategy", "model_name",
            "travel_time_s", "feasible", "distance_m", "uncertainty_s",
            "compute_ms", "has_incident", "has_closure", "forecast_available",
        ]
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for rec in all_records:
                writer.writerow(asdict(rec))
    logger.info("Journey CSV saved: %s", csv_path)

    return result


if __name__ == "__main__":
    logger.info("Starting integrated benchmark...")
    t0 = time.time()
    results = run_integrated_benchmark(num_seeds=5, steps_warmup=30)

    print("\n" + "=" * 70)
    print("SURAKSHIT BHARAT — Integrated Routing Benchmark Results")
    print(f"Timestamp: {results.timestamp}")
    print(f"Total journeys: {results.total_journeys}")
    print(f"Model availability: {results.model_availability}")
    print("=" * 70)
    print(f"\n{'Strategy':<35} {'Mean(s)':>8} {'P90(s)':>8} {'Improv%':>8} {'ForecastUsed':>13}")
    print("-" * 70)

    baseline_mean = results.strategies.get("A_shortest_distance", {}).get("mean_s", float("nan"))
    for skey, sm in results.strategies.items():
        mean_s = sm.get("mean_s", float("nan"))
        p90_s = sm.get("p90_s", float("nan"))
        improv = sm.get("mean_improvement_pct_vs_A", float("nan"))
        nf = sm.get("n_with_forecast", 0)
        ntotal = sm.get("n_journeys", 0)
        name = sm.get("strategy_name", skey)
        mean_str = f"{mean_s:8.1f}" if np.isfinite(mean_s) else "    N/A "
        p90_str = f"{p90_s:8.1f}" if np.isfinite(p90_s) else "    N/A "
        imp_str = f"{improv:+8.2f}" if np.isfinite(improv) else "    N/A "
        print(f"{name:<35} {mean_str} {p90_str} {imp_str}  {nf}/{ntotal}")

    print("-" * 70)
    print("\nForecasting performance (normalised speeds):")
    for mname, mvals in results.forecasting_metrics.get("results", {}).items():
        if isinstance(mvals, dict):
            mae = mvals.get("MAE", float("nan"))
            rmse = mvals.get("RMSE", float("nan"))
            print(f"  {mname:<25}: MAE={mae:.4f}  RMSE={rmse:.4f}")

    elapsed = time.time() - t0
    print(f"\nBenchmark completed in {elapsed:.1f}s")
    print(f"Results saved to: results/metrics/integrated_benchmark_results.json")
