"""
FastAPI backend for Surakshit Bharat Q-JEPA Traffic.
"""
from __future__ import annotations

import json
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.simulation.road_network import RoadNetwork
from src.simulation.traffic_simulator import TrafficSimulator, ScenarioType
from src.routing.shortest_path import (
    compare_all_routes, route_current_traffic,
    RouteResult, RouterType,
)
from src.common.config import config, abs_path
from src.common.logging import get_logger

logger = get_logger(__name__)

# ─── App ───────────────────────────────────────────────────
app = FastAPI(
    title="Surakshit Bharat Q-JEPA Traffic API",
    description="Predict Traffic. Protect Lives. Clear the Way.",
    version="0.1.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

# ─── Global simulation state ───────────────────────────────
_network: Optional[RoadNetwork] = None
_simulator: Optional[TrafficSimulator] = None
_sim_states: List[Dict] = []
_experiments: Dict[str, Dict] = {}


def _get_network() -> RoadNetwork:
    global _network
    if _network is None:
        _network = RoadNetwork.create_synthetic_grid(rows=5, cols=6, seed=42)
    return _network


def _get_simulator(scenario: str = "normal") -> TrafficSimulator:
    global _simulator
    net = _get_network()
    stype = ScenarioType(scenario) if scenario in [s.value for s in ScenarioType] else ScenarioType.NORMAL
    if _simulator is None:
        _simulator = TrafficSimulator(network=net, scenario=stype, seed=42)
    return _simulator


# ─── Schemas ───────────────────────────────────────────────
class RouteRequest(BaseModel):
    origin: str = Field(..., description="Origin node ID")
    destination: str = Field(..., description="Destination node ID")
    risk_penalty: float = Field(0.3, ge=0, le=5.0)


class ForecastRequest(BaseModel):
    edge_ids: Optional[List[str]] = None
    horizon_steps: int = Field(3, ge=1, le=12)


class SimStartRequest(BaseModel):
    scenario: str = "normal"
    seed: int = 42
    rows: int = Field(5, ge=2, le=10)
    cols: int = Field(6, ge=2, le=10)


class IncidentRequest(BaseModel):
    u: str
    v: str
    incident_type: str = "collision"


class ExperimentRequest(BaseModel):
    scenario: str = "normal"
    seed: int = 42
    num_steps: int = Field(50, ge=5, le=288)


# ─── Helper ────────────────────────────────────────────────
def _route_to_dict(r: RouteResult) -> Dict:
    return {
        "router": r.router_type.value,
        "origin": r.origin,
        "destination": r.destination,
        "path": r.path,
        "total_distance_m": r.total_distance_m,
        "estimated_travel_time_s": r.estimated_travel_time_s,
        "uncertainty_s": r.uncertainty_s,
        "computation_time_ms": r.computation_time_ms,
        "feasible": r.feasible,
        "explanation": r.explanation,
        "risk_score": r.risk_score,
    }


# ─── Endpoints ─────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "service": "Surakshit Bharat Q-JEPA Traffic API", "version": "0.1.0"}


@app.get("/system/status")
def system_status():
    net = _get_network()
    summary = net.summary()
    cfg = config()
    ckpt_dir = abs_path(cfg.checkpoints_dir)
    models_available = {
        "lstm": (ckpt_dir / "lstm.pt").exists(),
        "jepa": (ckpt_dir / "jepa.pt").exists(),
    }
    return {
        "network": summary,
        "simulator_running": _simulator is not None,
        "sim_step": _simulator.step if _simulator else 0,
        "models_available": models_available,
        "data_source": "synthetic",
        "note": "All data is synthetically generated. NOT real traffic.",
    }


@app.get("/network")
def get_network():
    net = _get_network()
    nodes = [
        {"id": n, "lat": d["lat"], "lon": d["lon"], "has_signal": d.get("has_traffic_signal", False)}
        for n, d in net.graph.nodes(data=True)
    ]
    edges = [
        {
            "u": u, "v": v,
            "segment_id": d.get("segment_id", f"{u}->{v}"),
            "length_m": d.get("length_m", 0),
            "current_speed_kmh": d.get("current_speed_kmh", 0),
            "congestion_level": d.get("congestion_level", 0),
            "is_closed": d.get("is_closed", False),
            "has_incident": d.get("has_incident", False),
        }
        for u, v, d in net.graph.edges(data=True)
    ]
    return {"nodes": nodes, "edges": edges, "summary": net.summary()}


@app.get("/traffic/current")
def traffic_current():
    net = _get_network()
    edge_states = []
    for u, v, d in net.graph.edges(data=True):
        edge_states.append({
            "edge_id": f"{u}->{v}",
            "current_speed_kmh": d.get("current_speed_kmh", 0),
            "congestion_level": d.get("congestion_level", 0),
            "is_closed": d.get("is_closed", False),
            "has_incident": d.get("has_incident", False),
            "current_travel_time_s": d.get("current_travel_time_s", 0),
        })
    return {"step": _simulator.step if _simulator else 0, "edges": edge_states}


@app.post("/traffic/forecast")
def traffic_forecast(req: ForecastRequest):
    net = _get_network()
    forecasts = []
    for u, v, d in net.graph.edges(data=True):
        eid = f"{u}->{v}"
        if req.edge_ids and eid not in req.edge_ids:
            continue
        pred = d.get("predicted_travel_times_s", {})
        forecasts.append({
            "edge_id": eid,
            "current_travel_time_s": d.get("current_travel_time_s", 0),
            "predicted_travel_time_s": pred.get(req.horizon_steps, d.get("current_travel_time_s", 0)),
            "uncertainty_s": d.get("uncertainty", 0),
            "horizon_steps": req.horizon_steps,
            "note": "Forecast unavailable — showing current value as fallback." if not pred else "",
        })
    return {"horizon_steps": req.horizon_steps, "forecasts": forecasts}


@app.post("/routes/compare")
def routes_compare(req: RouteRequest):
    net = _get_network()
    if req.origin not in net.graph.nodes:
        raise HTTPException(400, f"Origin node '{req.origin}' not in network")
    if req.destination not in net.graph.nodes:
        raise HTTPException(400, f"Destination node '{req.destination}' not in network")
    routes = compare_all_routes(net, req.origin, req.destination, req.risk_penalty)
    return {"routes": [_route_to_dict(r) for r in routes], "data_source": "synthetic"}


@app.post("/routes/ambulance")
def ambulance_route(req: RouteRequest):
    net = _get_network()
    if req.origin not in net.graph.nodes:
        raise HTTPException(400, f"Origin '{req.origin}' not found")
    if req.destination not in net.graph.nodes:
        raise HTTPException(400, f"Destination '{req.destination}' not found")
    route = route_current_traffic(net, req.origin, req.destination)
    return {**_route_to_dict(route), "advisory": "SIMULATED — not for operational use"}


@app.post("/simulation/start")
def simulation_start(req: SimStartRequest):
    global _network, _simulator, _sim_states
    _network = RoadNetwork.create_synthetic_grid(rows=req.rows, cols=req.cols, seed=req.seed)
    stype = ScenarioType(req.scenario) if req.scenario in [s.value for s in ScenarioType] else ScenarioType.NORMAL
    _simulator = TrafficSimulator(network=_network, scenario=stype, seed=req.seed)
    _sim_states = []
    return {"status": "started", "scenario": req.scenario, "nodes": _network.graph.number_of_nodes()}


@app.post("/simulation/step")
def simulation_step():
    sim = _get_simulator()
    state = sim.step_forward()
    brief = {
        "step": state.step, "time_minutes": state.time_minutes,
        "network_summary": state.network_summary,
        "incidents": state.incidents, "closed_edges": state.closed_edges,
        "note": "[SIMULATED DATA]",
    }
    _sim_states.append(brief)
    return brief


@app.post("/simulation/incident")
def simulation_incident(req: IncidentRequest):
    sim = _get_simulator()
    net = _get_network()
    if not net.graph.has_edge(req.u, req.v):
        raise HTTPException(400, f"Edge {req.u}->{req.v} not found")
    sim.add_incident(req.u, req.v)
    return {"status": "incident added", "edge": f"{req.u}->{req.v}"}


@app.get("/experiments")
def list_experiments():
    return {"experiments": list(_experiments.values())}


@app.post("/experiments/run")
def run_experiment(req: ExperimentRequest, background_tasks: BackgroundTasks):
    exp_id = str(uuid.uuid4())[:8]
    _experiments[exp_id] = {
        "id": exp_id, "status": "running",
        "scenario": req.scenario, "seed": req.seed,
        "num_steps": req.num_steps, "results": None,
    }
    background_tasks.add_task(_run_experiment_task, exp_id, req)
    return {"experiment_id": exp_id, "status": "running"}


def _run_experiment_task(exp_id: str, req: ExperimentRequest):
    try:
        net = RoadNetwork.create_synthetic_grid(seed=req.seed)
        stype = ScenarioType(req.scenario) if req.scenario in [s.value for s in ScenarioType] else ScenarioType.NORMAL
        sim = TrafficSimulator(network=net, scenario=stype, seed=req.seed)
        nodes = net.node_list()
        origin, dest = nodes[0], nodes[-1]

        journey_times = {}
        for rtype_str in ["shortest_distance", "current_traffic"]:
            rt = RouterType(rtype_str)
            route = compare_all_routes(net, origin, dest)[list(RouterType).index(rt)]
            journey_times[rtype_str] = route.estimated_travel_time_s

        for _ in range(req.num_steps):
            sim.step_forward()

        _experiments[exp_id]["status"] = "complete"
        _experiments[exp_id]["results"] = {
            "journey_times_s": journey_times,
            "final_network": net.summary(),
            "note": "[SIMULATED EXPERIMENT]",
        }
    except Exception as e:
        _experiments[exp_id]["status"] = "failed"
        _experiments[exp_id]["error"] = str(e)


@app.get("/experiments/{experiment_id}")
def get_experiment(experiment_id: str):
    if experiment_id not in _experiments:
        raise HTTPException(404, f"Experiment {experiment_id} not found")
    return _experiments[experiment_id]


@app.get("/metrics")
def get_metrics():
    metrics_path = abs_path("results/metrics/forecasting_metrics.json")
    if not metrics_path.exists():
        return {"status": "not_available", "message": "Run training first: python scripts/train_baselines.py"}
    with open(metrics_path) as f:
        return json.load(f)


if __name__ == "__main__":
    import uvicorn
    cfg = config()
    uvicorn.run("src.api.main:app", host=cfg.api.host, port=cfg.api.port, reload=cfg.api.reload)
