"""
Benchmark runner — runs all routing experiments and produces metrics.
"""
import sys
import json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from src.simulation.road_network import RoadNetwork
from src.simulation.traffic_simulator import TrafficSimulator, ScenarioType
from src.simulation.ambulance_scenarios import create_scenario_suite
from src.routing.shortest_path import compare_all_routes, RouterType
from src.evaluation.prediction_metrics import routing_metrics, journey_time_improvement_pct
from src.common.config import config, abs_path
from src.common.logging import get_logger

logger = get_logger(__name__)


def run_benchmarks(num_runs: int = 5, steps_per_run: int = 30) -> dict:
    cfg = config()
    out_dir = abs_path("results/metrics")
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}
    for router_type in RouterType:
        all_results[router_type.value] = []

    for run_idx in range(num_runs):
        seed = 42 + run_idx
        net = RoadNetwork.create_synthetic_grid(rows=5, cols=6, seed=seed)
        scenarios = create_scenario_suite(net, seed=seed)

        for scenario in scenarios[:3]:  # use first 3 scenarios
            sim = TrafficSimulator(
                network=net, scenario=scenario.scenario_type, seed=seed
            )
            # Advance a few steps to build traffic state
            for _ in range(steps_per_run):
                sim.step_forward()

            origin = scenario.origin
            dest = scenario.destination
            routes = compare_all_routes(net, origin, dest)

            for route in routes:
                all_results[route.router_type.value].append(
                    route.estimated_travel_time_s if route.feasible else float("inf")
                )

    # Compute metrics per router
    summary = {}
    baseline_times = all_results.get(RouterType.SHORTEST_DISTANCE.value, [])

    for rtype, times in all_results.items():
        m = routing_metrics(times)
        summary[rtype] = m

    # Save
    out_path = out_dir / "benchmark_results.json"
    with open(out_path, "w") as f:
        json.dump({"summary": summary, "note": "[SIMULATED EXPERIMENTS]"}, f, indent=2)

    logger.info("Benchmark results saved: %s", out_path)
    return summary


if __name__ == "__main__":
    logger.info("=== Running Benchmark Experiments ===")
    results = run_benchmarks()
    for router, metrics in results.items():
        print(f"\n{router.upper()}")
        for k, v in metrics.items():
            if isinstance(v, float) and not np.isnan(v):
                print(f"  {k}: {v:.2f}")
