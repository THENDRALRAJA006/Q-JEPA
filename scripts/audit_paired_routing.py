"""
Audits paired journey-time differences across all 45 evaluated scenarios.
Calculates paired differences between:
- Current-traffic vs Time-dependent predicted
- Current-traffic vs Uncertainty-aware
- Shortest-distance vs Time-dependent predicted
- Shortest-distance vs Uncertainty-aware
- Time-dependent predicted vs Uncertainty-aware
"""
import sys
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
from src.simulation.road_network import RoadNetwork
from src.simulation.traffic_simulator import ScenarioType, TrafficSimulator
from src.simulation.incident_generator import IncidentGenerator
from src.forecasting.inference import EdgeHistoryBuffer
from scripts.run_time_dependent_benchmark import (
    load_jepa_engine, write_forecasts_to_network,
    compare_five_strategies, FallbackMode, simulate_trip_execution
)

def run_paired_routing_audit():
    seeds = [42, 43, 44]
    scenarios_per_seed = 15
    warmup_steps = 10
    risk_penalty = 0.5

    results = []

    for seed in seeds:
        network = RoadNetwork.create_synthetic_grid(rows=6, cols=6, seed=seed)
        for u, v, d in network.graph.edges(data=True):
            spd = float(d.get("baseline_speed_kmh", 40.0))
            d["historical_speed_kmh"] = spd
            d["historical_travel_time_s"] = d["length_m"] / (spd / 3.6)

        jepa_engine = load_jepa_engine()
        history = EdgeHistoryBuffer(window_size=12)
        sim = TrafficSimulator(network, scenario=ScenarioType.RUSH_HOUR, seed=seed)
        incidents = IncidentGenerator(network, seed=seed, incident_prob=0.04, closure_prob=0.008)

        for step in range(1, warmup_steps + 1):
            sim.step_forward()
            incidents.step(step)
            history.update_from_network(network)

        if jepa_engine:
            write_forecasts_to_network(network, history, jepa_engine, horizons=list(range(1, 13)))

        rng_seed = np.random.default_rng(seed)
        for u, v, d in network.graph.edges(data=True):
            curr_spd = float(d.get("current_speed_kmh", 40.0))
            base_spd = float(d.get("baseline_speed_kmh", 40.0))
            length_m = float(d.get("length_m", 500.0))
            has_inc = bool(d.get("has_incident", False))
            pred_tts = d.get("predicted_travel_times_s", {})
            pred_spds = d.get("predicted_speeds_kmh", {})
            unc_by_h = d.get("uncertainties_by_horizon", {})
            congestion = max(0.0, 1.0 - curr_spd / max(base_spd, 1.0))
            volatility = 1.0 + 3.0 * congestion + (4.0 if has_inc else 0.0)
            for h in range(1, 13):
                if h not in pred_tts:
                    trend = curr_spd + (base_spd - curr_spd) * (1.0 - np.exp(-0.1 * h))
                    s = float(np.clip(trend + rng_seed.normal(0, 1.5 * np.sqrt(h)), 5.0, 70.0))
                    pred_spds[h] = s
                    pred_tts[h] = length_m / (s / 3.6)
                base_sigma = 2.0 * volatility * (length_m / 500.0)
                unc_by_h[h] = float(base_sigma * np.sqrt(h))
            d["predicted_travel_times_s"] = pred_tts
            d["predicted_speeds_kmh"] = pred_spds
            d["uncertainties_by_horizon"] = unc_by_h

        nodes = list(network.graph.nodes())
        rng_pairs = np.random.default_rng(seed + 999)
        for sc_idx in range(scenarios_per_seed):
            sc_id = f"SEED_{seed}_SCEN_{sc_idx+1:02d}"
            orig = str(rng_pairs.choice(nodes))
            dest = str(rng_pairs.choice([n for n in nodes if n != orig]))

            future_edge_states = {}
            sim_rollout = TrafficSimulator(network, scenario=ScenarioType.RUSH_HOUR, seed=seed + 1000 + sc_idx)
            inc_rollout = IncidentGenerator(sim_rollout.network, seed=seed + 1000 + sc_idx, incident_prob=0.04, closure_prob=0.008)
            for h in range(1, 13):
                sim_rollout.step_forward()
                inc_rollout.step(warmup_steps + h)
                future_edge_states[h] = {
                    (u, v): {
                        "speed_kmh": float(d.get("gt_speed_kmh", d.get("current_speed_kmh", 30.0))),
                        "is_closed": bool(d.get("is_closed", False)),
                    }
                    for u, v, d in sim_rollout.network.graph.edges(data=True)
                }

            routes = compare_five_strategies(
                network, orig, dest,
                departure_time_s=0.0,
                risk_penalty=risk_penalty,
                step_interval_s=300.0,
                max_supported_horizon=12,
                fallback_mode=FallbackMode.CLAMP_MAX_HORIZON,
            )

            tts = {}
            etas = {}
            for k in ["1_shortest_distance", "2_current_traffic", "3_static_historical", "4_time_dependent_predicted", "5_time_dependent_uncertainty"]:
                r = routes[k]
                is_dynamic = k in ("4_time_dependent_predicted", "5_time_dependent_uncertainty")
                realized_tt, reroutes, fallbacks = simulate_trip_execution(
                    network=network, initial_route=r, origin=orig, destination=dest,
                    departure_step=warmup_steps, strategy_key=k,
                    future_edge_states=future_edge_states, enable_dynamic_reroute=is_dynamic,
                    risk_penalty=risk_penalty,
                )
                tts[k] = realized_tt
                etas[k] = r.estimated_travel_time_s

            results.append({
                "scenario_id": sc_id,
                "seed": seed,
                "origin": orig,
                "destination": dest,
                "shortest_time_s": tts["1_shortest_distance"],
                "current_time_s": tts["2_current_traffic"],
                "historical_time_s": tts["3_static_historical"],
                "predicted_time_s": tts["4_time_dependent_predicted"],
                "uncertainty_time_s": tts["5_time_dependent_uncertainty"],
                "diff_curr_minus_pred": tts["2_current_traffic"] - tts["4_time_dependent_predicted"],
                "diff_curr_minus_unc": tts["2_current_traffic"] - tts["5_time_dependent_uncertainty"],
                "diff_short_minus_pred": tts["1_shortest_distance"] - tts["4_time_dependent_predicted"],
                "diff_short_minus_unc": tts["1_shortest_distance"] - tts["5_time_dependent_uncertainty"],
                "diff_pred_minus_unc": tts["4_time_dependent_predicted"] - tts["5_time_dependent_uncertainty"],
                "curr_path": routes["2_current_traffic"].path,
                "pred_path": routes["4_time_dependent_predicted"].path,
                "unc_path": routes["5_time_dependent_uncertainty"].path,
                "routes_differ": routes["2_current_traffic"].path != routes["4_time_dependent_predicted"].path or routes["4_time_dependent_predicted"].path != routes["5_time_dependent_uncertainty"].path
            })

    diffs_pred = [r["diff_curr_minus_pred"] for r in results]
    diffs_unc = [r["diff_curr_minus_unc"] for r in results]
    diffs_short_pred = [r["diff_short_minus_pred"] for r in results]
    diffs_pred_unc = [r["diff_pred_minus_unc"] for r in results]

    print("=" * 80)
    print("PAIRED JOURNEY-TIME AUDIT ACROSS 45 SCENARIOS")
    print("=" * 80)
    print("1. CURRENT-TRAFFIC vs TIME-DEPENDENT PREDICTED (Current - Predicted):")
    print(f"   Mean difference: {np.mean(diffs_pred):+.2f} s")
    print(f"   Median difference: {np.median(diffs_pred):+.2f} s")
    print(f"   Standard deviation: {np.std(diffs_pred):.2f} s")
    print(f"   Scenarios where Predicted was FASTER (Diff > 0): {sum(1 for d in diffs_pred if d > 0.01)} / 45 ({sum(1 for d in diffs_pred if d > 0.01)/45*100:.1f}%)")
    print(f"   Scenarios where Current Traffic was FASTER (Diff < 0): {sum(1 for d in diffs_pred if d < -0.01)} / 45 ({sum(1 for d in diffs_pred if d < -0.01)/45*100:.1f}%)")
    print(f"   Scenarios where Routes TIED (Diff == 0): {sum(1 for d in diffs_pred if abs(d) <= 0.01)} / 45 ({sum(1 for d in diffs_pred if abs(d) <= 0.01)/45*100:.1f}%)")

    print("\n2. CURRENT-TRAFFIC vs TIME-DEPENDENT UNCERTAINTY-AWARE (Current - Uncertainty):")
    print(f"   Mean difference: {np.mean(diffs_unc):+.2f} s")
    print(f"   Median difference: {np.median(diffs_unc):+.2f} s")
    print(f"   Standard deviation: {np.std(diffs_unc):.2f} s")
    print(f"   Scenarios where Uncertainty was FASTER (Diff > 0): {sum(1 for d in diffs_unc if d > 0.01)} / 45 ({sum(1 for d in diffs_unc if d > 0.01)/45*100:.1f}%)")
    print(f"   Scenarios where Current Traffic was FASTER (Diff < 0): {sum(1 for d in diffs_unc if d < -0.01)} / 45 ({sum(1 for d in diffs_unc if d < -0.01)/45*100:.1f}%)")
    print(f"   Scenarios where Routes TIED (Diff == 0): {sum(1 for d in diffs_unc if abs(d) <= 0.01)} / 45 ({sum(1 for d in diffs_unc if abs(d) <= 0.01)/45*100:.1f}%)")

    print("\n3. SHORTEST-DISTANCE vs TIME-DEPENDENT PREDICTED (Shortest - Predicted):")
    print(f"   Mean difference: {np.mean(diffs_short_pred):+.2f} s")
    print(f"   Median difference: {np.median(diffs_short_pred):+.2f} s")
    print(f"   Standard deviation: {np.std(diffs_short_pred):.2f} s")
    print(f"   Scenarios where Predicted was FASTER (Diff > 0): {sum(1 for d in diffs_short_pred if d > 0.01)} / 45 ({sum(1 for d in diffs_short_pred if d > 0.01)/45*100:.1f}%)")
    print(f"   Scenarios where Shortest was FASTER (Diff < 0): {sum(1 for d in diffs_short_pred if d < -0.01)} / 45 ({sum(1 for d in diffs_short_pred if d < -0.01)/45*100:.1f}%)")
    print(f"   Scenarios where Routes TIED (Diff == 0): {sum(1 for d in diffs_short_pred if abs(d) <= 0.01)} / 45 ({sum(1 for d in diffs_short_pred if abs(d) <= 0.01)/45*100:.1f}%)")

    print("\n4. TIME-DEPENDENT PREDICTED vs UNCERTAINTY-AWARE (Predicted - Uncertainty):")
    print(f"   Mean difference: {np.mean(diffs_pred_unc):+.2f} s")
    print(f"   Median difference: {np.median(diffs_pred_unc):+.2f} s")
    print(f"   Scenarios where Uncertainty was FASTER (Diff > 0): {sum(1 for d in diffs_pred_unc if d > 0.01)} / 45")
    print(f"   Scenarios where Predicted was FASTER (Diff < 0): {sum(1 for d in diffs_pred_unc if d < -0.01)} / 45")
    print(f"   Scenarios where Routes TIED (Diff == 0): {sum(1 for d in diffs_pred_unc if abs(d) <= 0.01)} / 45")

    print("\n" + "=" * 80)
    print("DETAILED BREAKDOWN OF DIVERGENT SCENARIOS:")
    print("=" * 80)
    divergent = [r for r in results if r["routes_differ"]]
    for r in divergent:
        print(f"\nScenario: {r['scenario_id']} (Seed {r['seed']}) | {r['origin']} -> {r['destination']}")
        print(f"  Current:     {r['current_time_s']:.1f} s | Path: {r['curr_path']}")
        print(f"  Predicted:   {r['predicted_time_s']:.1f} s | Path: {r['pred_path']} (Diff vs Current: {r['diff_curr_minus_pred']:+.1f} s)")
        print(f"  Uncertainty: {r['uncertainty_time_s']:.1f} s | Path: {r['unc_path']} (Diff vs Current: {r['diff_curr_minus_unc']:+.1f} s)")
        print(f"  Shortest:    {r['shortest_time_s']:.1f} s")

    return results

if __name__ == "__main__":
    run_paired_routing_audit()
