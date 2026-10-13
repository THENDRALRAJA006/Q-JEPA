"""
Traffic simulator for Surakshit Bharat Q-JEPA.
Generates time-varying speed/flow observations on the road network.
Supports multiple scenario types, incidents, and queue dynamics.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.simulation.road_network import RoadNetwork
from src.common.logging import get_logger

logger = get_logger(__name__)


class ScenarioType(str, Enum):
    NORMAL = "normal"
    RUSH_HOUR = "rush_hour"
    ROAD_BLOCKAGE = "road_blockage"
    QUEUE_SPILLBACK = "queue_spillback"
    MISSING_OBS = "missing_obs"
    NOISY_OBS = "noisy_obs"
    DYNAMIC_REROUTE = "dynamic_reroute"


@dataclass
class TrafficObservation:
    """One observation per edge per timestep (what the model sees)."""
    step: int
    edge_id: str         # "u->v"
    u: str
    v: str
    observed_speed_kmh: Optional[float]   # None = missing
    flow_veh_per_hour: Optional[float]
    congestion_level: Optional[float]
    is_noisy: bool = False
    is_missing: bool = False


@dataclass
class SimulationState:
    """Full ground-truth simulation state at one step."""
    step: int
    time_minutes: float
    network_summary: Dict
    edge_states: Dict[str, Dict]           # edge_id -> state dict
    observations: List[TrafficObservation]
    incidents: List[Dict]
    closed_edges: List[str]


class TrafficSimulator:
    """
    Step-based traffic simulator.

    The simulator maintains ground-truth state separately from observations.
    The gap is used for forecasting experiments.
    """

    def __init__(
        self,
        network: RoadNetwork,
        scenario: ScenarioType = ScenarioType.NORMAL,
        seed: int = 42,
        time_step_minutes: int = 5,
        missing_obs_rate: float = 0.0,
        noise_std_kmh: float = 0.0,
    ):
        self.network = network
        self.scenario = scenario
        self.seed = seed
        self.time_step_minutes = time_step_minutes
        self.missing_obs_rate = missing_obs_rate
        self.noise_std_kmh = noise_std_kmh

        self._rng = np.random.default_rng(seed)
        self.step = 0
        self.incidents: List[Dict] = []
        self.closed_edges: List[str] = []
        self._initialize_speeds()

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def _initialize_speeds(self) -> None:
        """Set initial ground-truth speeds to baseline."""
        for u, v, d in self.network.graph.edges(data=True):
            d["gt_speed_kmh"] = d["baseline_speed_kmh"]

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def step_forward(self) -> SimulationState:
        """Advance simulation by one time step. Returns full state."""
        self.step += 1
        time_min = self.step * self.time_step_minutes

        self._update_ground_truth(time_min)
        observations = self._generate_observations()
        self._sync_observed_to_network()

        state = SimulationState(
            step=self.step,
            time_minutes=time_min,
            network_summary=self.network.summary(),
            edge_states=self._collect_edge_states(),
            observations=observations,
            incidents=list(self.incidents),
            closed_edges=list(self.closed_edges),
        )
        return state

    def _update_ground_truth(self, time_min: float) -> None:
        """Update ground-truth speeds based on scenario and time."""
        hour = (time_min / 60.0) % 24.0

        for u, v, d in self.network.graph.edges(data=True):
            base = d["baseline_speed_kmh"]

            # Time-of-day congestion pattern
            tod_factor = self._time_of_day_factor(hour)

            # Incident / closure override
            incident_factor = 1.0
            eid = f"{u}->{v}"
            if d.get("has_incident"):
                incident_factor = 0.3
            if d.get("is_closed"):
                d["gt_speed_kmh"] = 0.0
                continue

            # Small random walk perturbation
            prev = d.get("gt_speed_kmh", base)
            perturb = float(self._rng.normal(0, 1.5))
            new_speed = prev + 0.3 * (base * tod_factor * incident_factor - prev) + perturb
            d["gt_speed_kmh"] = max(3.0, min(base * 1.1, new_speed))

        # Scenario-specific dynamics
        self._apply_scenario_dynamics(time_min)

    def _time_of_day_factor(self, hour: float) -> float:
        """Speed fraction relative to baseline for time of day."""
        if self.scenario == ScenarioType.RUSH_HOUR:
            if 8 <= hour < 10 or 17 <= hour < 20:
                return 0.35
        # Default gentle rush-hour pattern
        if 8 <= hour < 10:
            return 0.55
        if 17 <= hour < 19:
            return 0.50
        if 0 <= hour < 5:
            return 1.0
        return 0.80

    def _apply_scenario_dynamics(self, time_min: float) -> None:
        step = self.step
        if self.scenario == ScenarioType.ROAD_BLOCKAGE and step == 20:
            # Block a random edge
            edges = self.network.edge_list()
            if edges:
                u, v = edges[len(edges) // 2]
                self.network.close_edge(u, v, step)
                eid = f"{u}->{v}"
                if eid not in self.closed_edges:
                    self.closed_edges.append(eid)
                logger.info("Scenario: Road blockage at %s->%s (step %d)", u, v, step)

        elif self.scenario == ScenarioType.QUEUE_SPILLBACK and 15 <= step <= 50:
            # Reduce speeds on a chain of edges
            edges = self.network.edge_list()
            for i in range(min(5, len(edges))):
                u, v = edges[i]
                d = self.network.graph[u][v]
                d["gt_speed_kmh"] = max(3.0, d["gt_speed_kmh"] * 0.85)

        elif self.scenario == ScenarioType.DYNAMIC_REROUTE and step == 30:
            # An incident appears mid-journey
            edges = self.network.edge_list()
            if len(edges) > 10:
                u, v = edges[10]
                self.network.update_edge_state(u, v, has_incident=True, step=step)
                self.incidents.append({"edge": f"{u}->{v}", "step": step, "type": "collision"})
                logger.info("Scenario: Incident on %s->%s at step %d", u, v, step)

    def _generate_observations(self) -> List[TrafficObservation]:
        """Generate observations (with possible noise/missing) from ground truth."""
        obs_list = []
        for u, v, d in self.network.graph.edges(data=True):
            eid = f"{u}->{v}"
            gt_speed = d.get("gt_speed_kmh", d["baseline_speed_kmh"])
            is_missing = False
            is_noisy = False
            observed_speed: Optional[float] = None
            observed_flow: Optional[float] = None
            cong: Optional[float] = None

            if self.scenario == ScenarioType.MISSING_OBS or self.missing_obs_rate > 0:
                p_miss = 0.3 if self.scenario == ScenarioType.MISSING_OBS else self.missing_obs_rate
                if float(self._rng.random()) < p_miss:
                    is_missing = True

            if not is_missing:
                spd = gt_speed
                if self.scenario == ScenarioType.NOISY_OBS or self.noise_std_kmh > 0:
                    std = 5.0 if self.scenario == ScenarioType.NOISY_OBS else self.noise_std_kmh
                    spd = max(1.0, gt_speed + float(self._rng.normal(0, std)))
                    is_noisy = True
                observed_speed = spd
                flow_factor = max(0.0, 1.0 - (spd / d["baseline_speed_kmh"]))
                observed_flow = float(d["baseline_speed_kmh"] * 10 * (1 - flow_factor))
                cong = max(0.0, min(1.0, 1.0 - spd / d["baseline_speed_kmh"]))

            obs_list.append(
                TrafficObservation(
                    step=self.step,
                    edge_id=eid,
                    u=u,
                    v=v,
                    observed_speed_kmh=observed_speed,
                    flow_veh_per_hour=observed_flow,
                    congestion_level=cong,
                    is_noisy=is_noisy,
                    is_missing=is_missing,
                )
            )
        return obs_list

    def _sync_observed_to_network(self) -> None:
        """Update network state from observations (simulating real sensor fusion)."""
        for u, v, d in self.network.graph.edges(data=True):
            gt_spd = d.get("gt_speed_kmh", d["baseline_speed_kmh"])
            if not d.get("is_closed"):
                self.network.update_edge_state(
                    u, v,
                    current_speed_kmh=gt_spd,
                    step=self.step,
                )

    def _collect_edge_states(self) -> Dict[str, Dict]:
        result = {}
        for u, v, d in self.network.graph.edges(data=True):
            eid = f"{u}->{v}"
            result[eid] = {
                "u": u, "v": v,
                "segment_id": eid,
                "length_m": d["length_m"],
                "baseline_speed_kmh": d["baseline_speed_kmh"],
                "gt_speed_kmh": d.get("gt_speed_kmh", d["baseline_speed_kmh"]),
                "current_speed_kmh": d["current_speed_kmh"],
                "current_travel_time_s": d["current_travel_time_s"],
                "congestion_level": d["congestion_level"],
                "has_incident": d.get("has_incident", False),
                "is_closed": d.get("is_closed", False),
                "uncertainty": d.get("uncertainty", 0.0),
            }
        return result

    # ------------------------------------------------------------------
    # Manual controls
    # ------------------------------------------------------------------

    def add_incident(self, u: str, v: str) -> None:
        self.network.update_edge_state(u, v, has_incident=True, step=self.step)
        self.incidents.append({"edge": f"{u}->{v}", "step": self.step, "type": "manual"})
        logger.info("Manual incident added: %s->%s", u, v)

    def add_closure(self, u: str, v: str) -> None:
        self.network.close_edge(u, v, self.step)
        eid = f"{u}->{v}"
        if eid not in self.closed_edges:
            self.closed_edges.append(eid)
        logger.info("Road closure added: %s->%s", u, v)

    def clear_incident(self, u: str, v: str) -> None:
        self.network.update_edge_state(u, v, has_incident=False, step=self.step)

    def reset(self) -> None:
        self.step = 0
        self.incidents = []
        self.closed_edges = []
        self._rng = np.random.default_rng(self.seed)
        # Reset edge states
        for u, v, d in self.network.graph.edges(data=True):
            d["has_incident"] = False
            d["is_closed"] = False
            d["gt_speed_kmh"] = d["baseline_speed_kmh"]
            d["current_speed_kmh"] = d["baseline_speed_kmh"]
            d["current_travel_time_s"] = d["length_m"] / (d["baseline_speed_kmh"] / 3.6)
            d["congestion_level"] = 0.0
        logger.info("Simulator reset")
